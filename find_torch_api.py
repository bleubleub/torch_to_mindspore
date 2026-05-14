import argparse
import ast
import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin
from urllib.request import pathname2url

import pandas as pd


SCAN_COLUMNS = ["文件", "行号", "接口", "置信度", "类型来源", "推断类型"]
DEFAULT_MAPPING_FILE = "torch_to_mindspore_mapping.xlsx"
PYRIGHT_REQUEST_TIMEOUT_SECONDS = 10.0
_LAST_PROGRESS_LEN = 0
TENSOR_SOURCE_METHODS = {"full_tensor", "to_local"}
TENSOR_CHAIN_METHODS = {"cpu", "cuda", "detach", "float", "half", "to"} | TENSOR_SOURCE_METHODS
TENSOR_RETURNING_PROPERTIES = {"data", "grad"}
NON_TENSOR_RETURNING_PROPERTIES = {"device", "dtype", "itemsize", "ndim", "shape"}
NON_TENSOR_RETURNING_METHODS = {"item", "size", "tolist", "numpy"}
FALLBACK_TENSOR_METHODS = {
    "contiguous", "data", "device", "dtype", "reshape", "shape", "size", "view",
    "abs", "clone", "detach", "flatten", "permute", "squeeze", "transpose", "unsqueeze",
}


def load_tensor_methods(mapping_file: str = DEFAULT_MAPPING_FILE) -> Set[str]:
    methods = set(FALLBACK_TENSOR_METHODS)
    if not os.path.exists(mapping_file):
        return methods
    try:
        df = pd.read_excel(mapping_file, usecols=["Torch_API"])
    except Exception:
        return methods
    for value in df["Torch_API"].dropna():
        api = str(value).strip()
        if api.startswith("torch.Tensor."):
            methods.add(api.rsplit(".", 1)[-1])
    return methods


def find_torch_imports(content: str) -> Dict[str, str]:
    imports: Dict[str, str] = {}
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return imports

    nodes = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    nodes.sort(key=lambda node: (getattr(node, "lineno", 0), getattr(node, "col_offset", 0)))

    def remove_alias(alias_name: str):
        imports.pop(alias_name, None)
        for key in list(imports):
            if key.startswith(f"{alias_name}."):
                imports.pop(key, None)

    for node in nodes:
        if isinstance(node, ast.Import):
            for name in node.names:
                alias = name.asname or name.name
                if name.name.startswith("torch"):
                    imports[alias] = name.name
                    imports[name.name] = name.name
                else:
                    remove_alias(alias)
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("torch"):
                for name in node.names:
                    if name.name == "*":
                        continue
                    alias = name.asname or name.name
                    imports[alias] = f"{node.module}.{name.name}"
            else:
                for name in node.names:
                    if name.name != "*":
                        remove_alias(name.asname or name.name)
    return imports


def _expr_to_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        cur = node
        while isinstance(cur, ast.Attribute):
            parts.insert(0, cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.insert(0, cur.id)
            return ".".join(parts)
    return None


def _file_uri(path: str) -> str:
    return urljoin("file:", pathname2url(os.path.abspath(path)))


def _node_hover_position(node: ast.AST) -> Tuple[int, int]:
    line = max(getattr(node, "lineno", 1) - 1, 0)
    col = getattr(node, "end_col_offset", None)
    if col is None:
        col = getattr(node, "col_offset", 0)
    return line, max(col - 1, getattr(node, "col_offset", 0))


def _node_hover_positions(node: ast.AST) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = []

    def add(pos: Tuple[int, int]):
        if pos not in positions:
            positions.append(pos)

    if hasattr(node, "lineno"):
        line = max(getattr(node, "lineno", 1) - 1, 0)
        add((line, getattr(node, "col_offset", 0)))
        add(_node_hover_position(node))

    # Pyright often does not return hover info on the closing bracket in
    # expressions like `a[0].item()`. Query the base expression as a fallback.
    if isinstance(node, ast.Subscript):
        for pos in _node_hover_positions(node.value):
            add(pos)
    elif isinstance(node, ast.Attribute):
        for pos in _node_hover_positions(node.value):
            add(pos)
    return positions


def _tensor_chain_source(
    node: ast.AST,
    tensor_methods: Set[str],
    tensor_vars: Optional[Set[str]] = None,
) -> Optional[str]:
    tensor_vars = tensor_vars or set()
    if isinstance(node, ast.Name):
        return "variable" if node.id in tensor_vars else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        method = node.func.attr
        if method in TENSOR_SOURCE_METHODS:
            return method
        if method in NON_TENSOR_RETURNING_METHODS:
            return None
        if method in tensor_methods or method in TENSOR_CHAIN_METHODS:
            return method if _tensor_chain_source(node.func.value, tensor_methods, tensor_vars) else None
    if isinstance(node, ast.Attribute):
        attr = node.attr
        if attr in TENSOR_RETURNING_PROPERTIES:
            return attr
        if attr in NON_TENSOR_RETURNING_PROPERTIES:
            return None
        if attr in tensor_methods or attr in TENSOR_CHAIN_METHODS:
            return attr if _tensor_chain_source(node.value, tensor_methods, tensor_vars) else None
    if isinstance(node, ast.Subscript):
        return _tensor_chain_source(node.value, tensor_methods, tensor_vars)
    return None


def _tensor_like_expr_source(node: ast.AST, tensor_methods: Set[str], tensor_vars: Set[str]) -> Optional[str]:
    if isinstance(node, ast.Name):
        return "variable" if node.id in tensor_vars else None
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in TENSOR_SOURCE_METHODS:
                return method
            if method in NON_TENSOR_RETURNING_METHODS:
                return None
            if method in tensor_methods or method in TENSOR_CHAIN_METHODS:
                return method if _tensor_like_expr_source(node.func.value, tensor_methods, tensor_vars) else None
        name = _expr_to_name(node.func)
        if name and name.startswith("torch."):
            return name
    if isinstance(node, ast.Attribute):
        if node.attr in NON_TENSOR_RETURNING_PROPERTIES:
            return None
        if node.attr in TENSOR_RETURNING_PROPERTIES:
            return node.attr
        return _tensor_chain_source(node, tensor_methods, tensor_vars)
    if isinstance(node, ast.Subscript):
        return _tensor_like_expr_source(node.value, tensor_methods, tensor_vars)
    return None


def _is_tensor_like_argument(node: ast.AST, tensor_methods: Set[str]) -> bool:
    return _tensor_like_expr_source(node, tensor_methods, set()) is not None


def _iter_call_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Call):
        return _expr_to_name(node.func)
    return None


def _for_body_range(node: ast.For) -> Tuple[int, int]:
    start = node.body[0].lineno if node.body else node.lineno
    end = getattr(node, "end_lineno", None) or max(
        getattr(child, "lineno", start) for child in ast.walk(node)
    )
    return start, end


def infer_tensor_loop_ranges(tree: ast.AST) -> Dict[str, List[Tuple[int, int]]]:
    ranges: Dict[str, List[Tuple[int, int]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        iter_name = _iter_call_name(node.iter)
        if not iter_name:
            continue

        tensor_targets: List[ast.AST] = []
        if iter_name.endswith(".named_parameters") or iter_name.endswith(".named_buffers"):
            if isinstance(node.target, (ast.Tuple, ast.List)) and len(node.target.elts) >= 2:
                tensor_targets.append(node.target.elts[1])
        elif iter_name.endswith(".parameters") or iter_name.endswith(".buffers"):
            tensor_targets.append(node.target)

        start, end = _for_body_range(node)
        for target in tensor_targets:
            if isinstance(target, ast.Name):
                ranges.setdefault(target.id, []).append((start, end))
    return ranges


def infer_tensor_variable_ranges(tree: ast.AST, tensor_methods: Set[str]) -> Dict[str, List[Tuple[int, int]]]:
    tensor_vars: Set[str] = set()
    active_start: Dict[str, int] = {}
    loop_ranges = infer_tensor_loop_ranges(tree)
    ranges: Dict[str, List[Tuple[int, int]]] = {name: list(items) for name, items in loop_ranges.items()}
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and hasattr(node, "lineno")
    ]
    assignments.sort(key=lambda node: (node.lineno, getattr(node, "col_offset", 0)))

    for node in assignments:
        value = node.value
        if value is None:
            continue
        line_tensor_vars = set(tensor_vars)
        line_tensor_vars.update(
            name for name in loop_ranges
            if _is_tensor_variable_at(name, node.lineno, loop_ranges)
        )
        source = _tensor_like_expr_source(value, tensor_methods, line_tensor_vars)
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id in active_start:
                    ranges.setdefault(target.id, []).append((active_start.pop(target.id), node.lineno))
                if source:
                    tensor_vars.add(target.id)
                    active_start[target.id] = node.lineno
                else:
                    tensor_vars.discard(target.id)
    for name, start in active_start.items():
        ranges.setdefault(name, []).append((start, 10**9))
    return ranges


def _is_tensor_variable_at(name: str, line: int, ranges: Dict[str, List[Tuple[int, int]]]) -> bool:
    return any(start <= line <= end for start, end in ranges.get(name, []))


def _receiver_tensor_variable_source(
    node: ast.AST,
    line: int,
    ranges: Dict[str, List[Tuple[int, int]]],
) -> Optional[str]:
    if isinstance(node, ast.Name) and _is_tensor_variable_at(node.id, line, ranges):
        return node.id
    if isinstance(node, ast.Subscript):
        return _receiver_tensor_variable_source(node.value, line, ranges)
    return None


def infer_tensor_function_params(tree: ast.AST, tensor_methods: Set[str]) -> Set[Tuple[str, str]]:
    function_params: Dict[str, List[str]] = {}
    tensor_params: Set[Tuple[str, str]] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            function_params[node.name] = [arg.arg for arg in node.args.args]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        params = function_params.get(node.func.id)
        if not params:
            continue
        for index, arg in enumerate(node.args):
            if index < len(params) and _is_tensor_like_argument(arg, tensor_methods):
                tensor_params.add((node.func.id, params[index]))
        for keyword in node.keywords:
            if keyword.arg in params and _is_tensor_like_argument(keyword.value, tensor_methods):
                tensor_params.add((node.func.id, keyword.arg))
    return tensor_params


def _hover_text(contents: Any) -> str:
    if contents is None:
        return ""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        return "\n".join(_hover_text(item) for item in contents)
    if isinstance(contents, dict):
        if "value" in contents:
            return str(contents.get("value") or "")
        if "contents" in contents:
            return _hover_text(contents.get("contents"))
    return str(contents)


class PyrightLspClient:
    def __init__(self, root_path: str):
        command = shutil.which("pyright-langserver") or shutil.which("basedpyright-langserver")
        self.available = bool(command)
        self.proc = None
        self.next_id = 1
        self._condition = threading.Condition()
        self._responses: Dict[int, Dict[str, Any]] = {}
        self._reader_error: Optional[Exception] = None
        if not command:
            return
        self.proc = subprocess.Popen(
            [command, "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        try:
            self._request("initialize", {
                "processId": os.getpid(),
                "rootUri": _file_uri(root_path),
                "capabilities": {},
            })
            self._notify("initialized", {})
        except Exception:
            self.available = False
            try:
                self.proc.terminate()
            except Exception:
                pass

    def close(self):
        if not self.proc:
            return
        try:
            self._request("shutdown", {})
            self._notify("exit", {})
        except Exception:
            pass
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _send(self, payload: Dict[str, Any]):
        data = json.dumps(payload).encode("utf-8")
        self.proc.stdin.write(f"Content-Length: {len(data)}\r\n\r\n".encode("ascii") + data)
        self.proc.stdin.flush()

    def _read_message(self) -> Dict[str, Any]:
        headers = {}
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("pyright language server stopped")
            line = line.decode("ascii").strip()
            if not line:
                break
            key, value = line.split(":", 1)
            headers[key.lower()] = value.strip()
        return json.loads(self.proc.stdout.read(int(headers["content-length"])).decode("utf-8"))

    def _reader_loop(self):
        while self.proc and self.proc.poll() is None:
            try:
                msg = self._read_message()
            except Exception as exc:
                with self._condition:
                    self._reader_error = exc
                    self._condition.notify_all()
                return
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            with self._condition:
                self._responses[msg_id] = msg
                self._condition.notify_all()

    def _request(self, method: str, params: Dict[str, Any], timeout: float = PYRIGHT_REQUEST_TIMEOUT_SECONDS) -> Any:
        req_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        with self._condition:
            while req_id not in self._responses:
                if self._reader_error is not None:
                    raise RuntimeError(self._reader_error)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.available = False
                    raise TimeoutError(f"pyright request '{method}' timed out after {timeout:.0f}s")
                self._condition.wait(remaining)
            msg = self._responses.pop(req_id)
        if "error" in msg:
            raise RuntimeError(msg["error"])
        return msg.get("result")

    def _notify(self, method: str, params: Dict[str, Any]):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def open_document(self, file_path: str, content: str):
        self._notify("textDocument/didOpen", {
            "textDocument": {
                "uri": _file_uri(file_path),
                "languageId": "python",
                "version": 1,
                "text": content,
            }
        })

    def hover(self, file_path: str, line: int, col: int) -> str:
        result = self._request("textDocument/hover", {
            "textDocument": {"uri": _file_uri(file_path)},
            "position": {"line": line, "character": col},
        })
        return _hover_text(result.get("contents")) if result else ""


class CandidateVisitor(ast.NodeVisitor):
    def __init__(
        self,
        imports: Dict[str, str],
        tensor_methods: Set[str],
        tensor_params: Set[Tuple[str, str]],
        tensor_var_ranges: Dict[str, List[Tuple[int, int]]],
    ):
        self.imports = imports
        self.tensor_methods = tensor_methods
        self.tensor_params = tensor_params
        self.tensor_var_ranges = tensor_var_ranges
        self.function_stack: List[str] = []
        self.candidates: List[Dict[str, Any]] = []
        self.seen: Set[Tuple[int, int, str]] = set()

    def _resolve_imported(self, node: ast.AST) -> Optional[str]:
        name = _expr_to_name(node)
        if not name:
            return None
        parts = name.split(".")
        for idx in range(len(parts), 0, -1):
            prefix = ".".join(parts[:idx])
            if prefix in self.imports:
                return ".".join([self.imports[prefix]] + parts[idx:])
        return None

    def _add(self, api: str, node: ast.AST, receiver: Optional[ast.AST] = None):
        if not hasattr(node, "lineno"):
            return
        key = (node.lineno, node.col_offset, api)
        if key in self.seen:
            return
        item = {"api": api, "line": node.lineno, "col_offset": node.col_offset}
        if receiver is not None:
            positions = _node_hover_positions(receiver)
            line, col = positions[0]
            tensor_chain_source = _tensor_chain_source(receiver, self.tensor_methods)
            item.update({
                "needs_type": True,
                "receiver_line": line,
                "receiver_col": col,
                "receiver_positions": positions,
            })
            if tensor_chain_source:
                item["tensor_chain_source"] = tensor_chain_source
            if isinstance(receiver, ast.Name) and self.function_stack:
                if (self.function_stack[-1], receiver.id) in self.tensor_params:
                    item["tensor_param_source"] = receiver.id
            tensor_variable_source = _receiver_tensor_variable_source(receiver, node.lineno, self.tensor_var_ranges)
            if tensor_variable_source:
                item["tensor_variable_source"] = tensor_variable_source
        else:
            item["needs_type"] = False
        self.candidates.append(item)
        self.seen.add(key)

    def visit_Call(self, node: ast.Call):
        imported = self._resolve_imported(node.func)
        if imported and imported.startswith("torch."):
            self._add(imported, node.func)
        elif isinstance(node.func, ast.Attribute) and node.func.attr in self.tensor_methods:
            self._add(f"torch.Tensor.{node.func.attr}", node.func, node.func.value)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        parent = getattr(node, "parent", None)
        if isinstance(parent, ast.Call) and parent.func == node:
            self.generic_visit(node)
            return
        if isinstance(parent, ast.Attribute) and parent.value == node:
            self.generic_visit(node)
            return
        imported = self._resolve_imported(node)
        if imported and imported.startswith("torch."):
            self._add(imported, node)
        elif node.attr in self.tensor_methods:
            self._add(f"torch.Tensor.{node.attr}", node, node.value)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_ClassDef(self, node: ast.ClassDef):
        for base in node.bases:
            imported = self._resolve_imported(base)
            if imported and imported.startswith("torch."):
                self._add(imported, base)
        self.generic_visit(node)


def collect_candidates(content: str, imports: Dict[str, str], tensor_methods: Set[str]) -> List[dict]:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    tensor_params = infer_tensor_function_params(tree, tensor_methods)
    tensor_var_ranges = infer_tensor_variable_ranges(tree, tensor_methods)
    visitor = CandidateVisitor(imports, tensor_methods, tensor_params, tensor_var_ranges)
    visitor.visit(tree)
    return visitor.candidates


def _is_known_non_tensor_type(type_text: str) -> bool:
    text = type_text.lower()
    return any(marker in text for marker in [
        "numpy", "ndarray", "enum", "str", "builtins.list", "builtins.dict",
        "list[", "dict[", "tuple[", "set[", "type[",
    ])


def _is_torch_tensor_type(type_text: str) -> bool:
    text = type_text.lower()
    if _is_known_non_tensor_type(text):
        return False
    if "torch" in text and ("tensor" in text or "parameter" in text):
        return True
    padded = f" {text} "
    return " tensor" in padded or " parameter" in padded


def find_torch_usage(
    content: str,
    imports: Dict[str, str],
    file_path: Optional[str] = None,
    resolver: Optional[PyrightLspClient] = None,
    mode: str = "pyright",
    tensor_methods: Optional[Set[str]] = None,
) -> List[dict]:
    tensor_methods = tensor_methods or load_tensor_methods()
    candidates = collect_candidates(content, imports, tensor_methods)
    results = []
    if file_path and resolver and resolver.available:
        resolver.open_document(file_path, content)

    for item in candidates:
        if not item.get("needs_type"):
            results.append({**item, "confidence": "confirmed", "type_source": "static", "inferred_type": ""})
            continue
        if item.get("tensor_chain_source"):
            results.append({
                **item,
                "confidence": "confirmed",
                "type_source": "tensor_chain",
                "inferred_type": f"receiver from Tensor method chain: {item['tensor_chain_source']}",
            })
            continue
        if item.get("tensor_param_source"):
            results.append({
                **item,
                "confidence": "confirmed",
                "type_source": "tensor_param",
                "inferred_type": f"function parameter inferred from Tensor argument: {item['tensor_param_source']}",
            })
            continue
        if item.get("tensor_variable_source"):
            results.append({
                **item,
                "confidence": "confirmed",
                "type_source": "tensor_variable",
                "inferred_type": f"variable inferred from Tensor assignment: {item['tensor_variable_source']}",
            })
            continue
        if mode == "static" or not file_path or not resolver or not resolver.available:
            continue
        inferred = ""
        try:
            for line, col in item.get("receiver_positions") or [(item["receiver_line"], item["receiver_col"])]:
                inferred = resolver.hover(file_path, line, col)
                if _is_torch_tensor_type(inferred):
                    break
        except TimeoutError:
            raise
        except Exception:
            pass
        if _is_torch_tensor_type(inferred):
            results.append({**item, "confidence": "confirmed", "type_source": "pyright", "inferred_type": inferred})
    return results


def _progress_bar(index: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    filled = round(width * index / total)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _truncate_middle(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    left = max_len // 2 - 1
    right = max_len - left - 3
    return f"{text[:left]}...{text[-right:]}"


def _format_progress(
    index: int,
    total_files: int,
    rel_path: str,
    elapsed: float,
    api_count: int,
) -> str:
    percent = index * 100 / total_files if total_files else 100.0
    prefix = f"{_progress_bar(index, total_files)} {percent:5.1f}% ({index}/{total_files}) file="
    suffix = f" elapsed={elapsed:.1f}s apis={api_count}"
    columns = max(shutil.get_terminal_size((100, 20)).columns, 40)
    max_path_len = max(columns - len(prefix) - len(suffix) - 1, 12)
    return f"{prefix}{_truncate_middle(rel_path, max_path_len)}{suffix}"


def _public_api_item(item: Dict[str, Any], file_path: str) -> Dict[str, Any]:
    return {
        "api": item["api"],
        "line": item["line"],
        "col_offset": item.get("col_offset", 0),
        "file": file_path,
        "confidence": item.get("confidence", "confirmed"),
        "type_source": item.get("type_source", "static"),
        "inferred_type": item.get("inferred_type", ""),
    }


def process_file(
    file_path: str,
    mode: str = "pyright",
    resolver: Optional[PyrightLspClient] = None,
    tensor_methods: Optional[Set[str]] = None,
) -> Tuple[List[dict], Optional[str]]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        imports = find_torch_imports(content)
        apis = find_torch_usage(content, imports, file_path, resolver, mode, tensor_methods)
        return [_public_api_item(api_info, file_path) for api_info in apis], None
    except Exception as exc:
        return [], str(exc)


def _print_progress(message: str):
    global _LAST_PROGRESS_LEN
    columns = max(shutil.get_terminal_size((100, 20)).columns, 40)
    message = _truncate_middle(message, columns - 1)
    print(f"\r\033[2K{message}", end="", flush=True)
    _LAST_PROGRESS_LEN = len(message)


def _clear_progress_line():
    global _LAST_PROGRESS_LEN
    if _LAST_PROGRESS_LEN:
        print("\r\033[2K", end="", flush=True)
        _LAST_PROGRESS_LEN = 0


def process_directory(
    directory_path: str,
    mode: str = "pyright",
    resolver: Optional[PyrightLspClient] = None,
    tensor_methods: Optional[Set[str]] = None,
) -> List[dict]:
    all_apis = []
    skip_dirs = {".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", "converted_files"}
    python_files = []
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith(".")]
        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))

    total_files = len(python_files)
    print(f"发现 Python 文件数: {total_files}", flush=True)
    start_time = time.monotonic()
    for index, file_path in enumerate(python_files, start=1):
        rel_path = os.path.relpath(file_path, directory_path)
        elapsed = time.monotonic() - start_time
        _print_progress(_format_progress(index, total_files, rel_path, elapsed, len(all_apis)))
        file_apis, skip_reason = process_file(file_path, mode, resolver, tensor_methods)
        all_apis.extend(file_apis)
        if skip_reason:
            _clear_progress_line()
            print(f"跳过文件 {rel_path}: {skip_reason}", flush=True)
    if total_files:
        _clear_progress_line()
    return all_apis


def main():
    parser = argparse.ArgumentParser(description="Scan Python files for PyTorch API usage.")
    parser.add_argument("target_path", nargs="?", default=".", help="Python file or directory to scan.")
    parser.add_argument("--mode", choices=["pyright", "static"], default="pyright",
                        help="pyright confirms Tensor receivers by type; static only scans explicit torch imports.")
    args = parser.parse_args()

    print(f"开始扫描: {args.target_path}")
    print(f"扫描模式: {args.mode}")

    resolver = None
    if args.mode == "pyright":
        root_path = args.target_path if os.path.isdir(args.target_path) else os.path.dirname(os.path.abspath(args.target_path)) or "."
        resolver = PyrightLspClient(root_path)
        if not resolver.available:
            print("警告: 未找到 pyright-langserver/basedpyright-langserver，仅扫描显式 torch API。")

    try:
        tensor_methods = load_tensor_methods()
        if os.path.isfile(args.target_path):
            results, skip_reason = process_file(args.target_path, args.mode, resolver, tensor_methods)
            if skip_reason:
                print(f"跳过文件 {args.target_path}: {skip_reason}")
        else:
            results = process_directory(args.target_path, args.mode, resolver, tensor_methods)
    finally:
        if resolver:
            resolver.close()

    with open("api_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print("扫描完成，生成的JSON文件已保存到：api_report.json")

    try:
        df = pd.DataFrame(results)
        if results:
            df = df.rename(columns={
                "file": "文件",
                "line": "行号",
                "api": "接口",
                "confidence": "置信度",
                "type_source": "类型来源",
                "inferred_type": "推断类型",
            })
            df = df[SCAN_COLUMNS].sort_values(by=["文件", "行号"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=SCAN_COLUMNS)
        df.to_excel("api_report_check.xlsx", index=False)
        print("为方便核对，Excel报告已保存到: api_report_check.xlsx")
    except Exception as exc:
        print(f"生成Excel报告时出错: {exc}")

    print(f"共找到 {len(results)} 个API调用。")


if __name__ == "__main__":
    main()
