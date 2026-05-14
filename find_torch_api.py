import argparse
import ast
import json
import os
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin
from urllib.request import pathname2url

import pandas as pd


SCAN_COLUMNS = ["文件", "行号", "接口", "置信度", "类型来源", "推断类型"]
DEFAULT_MAPPING_FILE = "torch_to_mindspore_mapping.xlsx"
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
        if not command:
            return
        self.proc = subprocess.Popen(
            [command, "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
        )
        self._request("initialize", {
            "processId": os.getpid(),
            "rootUri": _file_uri(root_path),
            "capabilities": {},
        })
        self._notify("initialized", {})

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

    def _request(self, method: str, params: Dict[str, Any]) -> Any:
        req_id = self.next_id
        self.next_id += 1
        self._send({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        while True:
            msg = self._read_message()
            if msg.get("id") == req_id:
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
    def __init__(self, imports: Dict[str, str], tensor_methods: Set[str]):
        self.imports = imports
        self.tensor_methods = tensor_methods
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
            line, col = _node_hover_position(receiver)
            item.update({"needs_type": True, "receiver_line": line, "receiver_col": col})
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
    visitor = CandidateVisitor(imports, tensor_methods)
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
        if mode == "static" or not file_path or not resolver or not resolver.available:
            continue
        inferred = ""
        try:
            inferred = resolver.hover(file_path, item["receiver_line"], item["receiver_col"])
        except Exception:
            pass
        if _is_torch_tensor_type(inferred):
            results.append({**item, "confidence": "confirmed", "type_source": "pyright", "inferred_type": inferred})
    return results


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
) -> List[dict]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        imports = find_torch_imports(content)
        apis = find_torch_usage(content, imports, file_path, resolver, mode, tensor_methods)
        return [_public_api_item(api_info, file_path) for api_info in apis]
    except Exception as exc:
        print(f"跳过文件 {file_path}: {exc}")
        return []


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
        print(f"[{index}/{total_files}] 扫描 {rel_path}，已耗时 {elapsed:.1f}s，已发现 {len(all_apis)} 个API", flush=True)
        all_apis.extend(process_file(file_path, mode, resolver, tensor_methods))
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
            results = process_file(args.target_path, args.mode, resolver, tensor_methods)
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
