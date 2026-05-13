import ast
import os
import pandas as pd
from typing import Set, Dict, List, Optional, Any

AMBIGUOUS_TENSOR_METHODS = {
    # Common Python/container/path attributes that are also Tensor APIs.
    # Only report these when the receiver can be inferred as a Tensor.
    'data', 'device', 'dtype', 'get', 'items', 'keys', 'name', 'shape',
    'size', 'type', 'values', 'view'
}

TENSOR_CONTEXT_ATTRS = {'shape', 'dtype', 'device'}

FUNCTION_PARAM_TENSOR_HINTS = {'shape', 'dtype', 'device', 'size', 'view'}

TENSOR_METHODS = {
    'H', 'T', '__abs__', '__add__', '__and__', '__array__', '__array_priority__', '__array_wrap__', '__bool__', '__complex__', '__contains__', '__cuda_array_interface__',
    '__deepcopy__', '__delattr__', '__delitem__', '__dict__', '__dir__', '__div__', '__dlpack__', '__dlpack_device__', '__doc__', '__eq__', '__float__', '__floordiv__', '__format__',
    '__ge__', '__getattribute__', '__getitem__', '__gt__', '__hash__', '__iadd__', '__iand__', '__idiv__', '__ifloordiv__', '__ilshift__', '__imod__', '__imul__', '__index__',
    '__int__', '__invert__', '__ior__', '__ipow__', '__irshift__', '__isub__', '__iter__', '__itruediv__', '__ixor__', '__le__', '__len__', '__long__',
    '__lshift__', '__lt__', '__matmul__', '__mod__', '__module__', '__mul__', '__ne__', '__neg__', '__nonzero__', '__or__', '__pos__', '__pow__', '__radd__', '__rand__',
    '__rdiv__', '__reduce__', '__reduce_ex__', '__repr__', '__reversed__', '__rfloordiv__', '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__rpow__', '__rrshift__',
    '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__setitem__', '__setstate__', '__sizeof__', '__str__', '__sub__', '__truediv__', '__xor__',
    '_addmm_activation', '_autocast_to_full_precision', '_autocast_to_reduced_precision', '_backward_hooks', '_base',
    '_cdata', '_coalesced_', '_conj', '_conj_physical', '_dimI', '_dimV', '_fix_weakref', '_grad', '_grad_fn', '_has_symbolic_sizes_strides', '_indices', '_is_all_true', '_is_any_true',
    '_is_view', '_is_zerotensor', '_make_subclass', '_make_wrapper_subclass', '_neg_view', '_nested_tensor_size', '_nested_tensor_storage_offsets', '_nested_tensor_strides', '_nnz',
    '_post_accumulate_grad_hooks', '_python_dispatch', '_reduce_ex_internal', '_sparse_mask_projection', '_to_dense', '_to_sparse', '_to_sparse_bsc', '_to_sparse_bsr', '_to_sparse_csc',
    '_to_sparse_csr', '_typed_storage', '_update_names', '_values', '_version', '_view_func', '_view_func_unsafe', 'abs', 'abs_', 'absolute', 'absolute_', 'acos', 'acos_', 'acosh',
    'acosh_', 'add', 'add_', 'addbmm', 'addbmm_', 'addcdiv', 'addcdiv_', 'addcmul', 'addcmul_', 'addmm', 'addmm_', 'addmv', 'addmv_', 'addr', 'addr_', 'adjoint', 'align_as', 'align_to',
    'all', 'allclose', 'amax', 'amin', 'aminmax', 'angle', 'any', 'apply_', 'arccos', 'arccos_', 'arccosh', 'arccosh_', 'arcsin', 'arcsin_', 'arcsinh', 'arcsinh_', 'arctan', 'arctan2',
    'arctan2_', 'arctan_', 'arctanh', 'arctanh_', 'argmax', 'argmin', 'argsort', 'argwhere', 'as_strided', 'as_strided_', 'as_strided_scatter', 'as_subclass', 'asin', 'asin_', 'asinh',
    'asinh_', 'atan', 'atan2', 'atan2_', 'atan_', 'atanh', 'atanh_', 'backward', 'baddbmm', 'baddbmm_', 'bernoulli', 'bernoulli_', 'bfloat16', 'bincount', 'bitwise_and', 'bitwise_and_',
    'bitwise_left_shift', 'bitwise_left_shift_', 'bitwise_not', 'bitwise_not_', 'bitwise_or', 'bitwise_or_', 'bitwise_right_shift', 'bitwise_right_shift_', 'bitwise_xor', 'bitwise_xor_',
    'bmm', 'bool', 'broadcast_to', 'byte', 'cauchy_', 'ccol_indices', 'cdouble', 'ceil', 'ceil_', 'cfloat', 'chalf', 'char', 'cholesky', 'cholesky_inverse', 'cholesky_solve', 'chunk',
    'clamp', 'clamp_', 'clamp_max', 'clamp_max_', 'clamp_min', 'clamp_min_', 'clip', 'clip_', 'clone', 'coalesce', 'col_indices', 'conj', 'conj_physical', 'conj_physical_', 'contiguous',
    'copy_', 'copysign', 'copysign_', 'corrcoef', 'cos', 'cos_', 'cosh', 'cosh_', 'count_nonzero', 'cov', 'cpu', 'cross', 'crow_indices', 'cuda', 'cummax', 'cummin', 'cumprod', 'cumprod_',
    'cumsum', 'cumsum_', 'data', 'data_ptr', 'deg2rad', 'deg2rad_', 'dense_dim', 'dequantize', 'det', 'detach', 'detach_', 'device', 'diag', 'diag_embed', 'diagflat', 'diagonal',
    'diagonal_scatter', 'diff', 'digamma', 'digamma_', 'dim', 'dim_order', 'dist', 'div', 'div_', 'divide', 'divide_', 'dot', 'double', 'dsplit', 'dtype', 'eig', 'element_size', 'eq',
    'eq_', 'equal', 'erf', 'erf_', 'erfc', 'erfc_', 'erfinv', 'erfinv_', 'exp', 'exp2', 'exp2_', 'exp_', 'expand', 'expand_as', 'expm1', 'expm1_', 'exponential_', 'fill_', 'fill_diagonal_',
    'fix', 'fix_', 'flatten', 'flip', 'fliplr', 'flipud', 'float', 'float_power', 'float_power_', 'floor', 'floor_', 'floor_divide', 'floor_divide_', 'fmax', 'fmin', 'fmod', 'fmod_',
    'frac', 'frac_', 'frexp', 'gather', 'gcd', 'gcd_', 'ge', 'ge_', 'geometric_', 'geqrf', 'ger', 'get_device', 'grad', 'grad_fn', 'greater', 'greater_', 'greater_equal', 'greater_equal_',
    'gt', 'gt_', 'half', 'hardshrink', 'has_names', 'heaviside', 'heaviside_', 'histc', 'histogram', 'hsplit', 'hypot', 'hypot_', 'i0', 'i0_', 'igamma', 'igamma_', 'igammac', 'igammac_',
    'imag', 'index_add', 'index_add_', 'index_copy', 'index_copy_', 'index_fill', 'index_fill_', 'index_put', 'index_put_', 'index_reduce', 'index_reduce_', 'index_select', 'indices',
    'inner', 'int', 'int_repr', 'inverse', 'ipu', 'is_coalesced', 'is_complex', 'is_conj', 'is_contiguous', 'is_cpu', 'is_cuda', 'is_distributed', 'is_floating_point', 'is_inference',
    'is_ipu', 'is_leaf', 'is_meta', 'is_mkldnn', 'is_mps', 'is_mtia', 'is_neg', 'is_nested', 'is_nonzero', 'is_ort', 'is_pinned', 'is_quantized', 'is_same_size', 'is_set_to', 'is_shared',
    'is_signed', 'is_sparse', 'is_sparse_csr', 'is_vulkan', 'is_xla', 'is_xpu', 'isclose', 'isfinite', 'isinf', 'isnan', 'isneginf', 'isposinf', 'isreal', 'istft', 'item', 'itemsize',
    'kron', 'kthvalue', 'layout', 'lcm', 'lcm_', 'ldexp', 'ldexp_', 'le', 'le_', 'lerp', 'lerp_', 'less', 'less_', 'less_equal', 'less_equal_', 'lgamma', 'lgamma_', 'log', 'log10',
    'log10_', 'log1p', 'log1p_', 'log2', 'log2_', 'log_', 'log_normal_', 'log_softmax', 'logaddexp', 'logaddexp2', 'logcumsumexp', 'logdet', 'logical_and', 'logical_and_', 'logical_not',
    'logical_not_', 'logical_or', 'logical_or_', 'logical_xor', 'logical_xor_', 'logit', 'logit_', 'logsumexp', 'long', 'lstsq', 'lt', 'lt_', 'lu', 'lu_solve', 'mH', 'mT', 'map2_',
    'map_', 'masked_fill', 'masked_fill_', 'masked_scatter', 'masked_scatter_', 'masked_select', 'matmul', 'matrix_exp', 'matrix_power', 'max', 'maximum', 'mean', 'median', 'min',
    'minimum', 'mm', 'mode', 'moveaxis', 'movedim', 'msort', 'mul', 'mul_', 'multinomial', 'multiply', 'multiply_', 'mv', 'mvlgamma', 'mvlgamma_', 'name', 'names', 'nan_to_num',
    'nan_to_num_', 'nanmean', 'nanmedian', 'nanquantile', 'nansum', 'narrow', 'narrow_copy', 'nbytes', 'ndim', 'ndimension', 'ne', 'ne_', 'neg', 'neg_', 'negative', 'negative_',
    'nelement', 'new', 'new_empty', 'new_empty_strided', 'new_full', 'new_ones', 'new_tensor', 'new_zeros', 'nextafter', 'nextafter_', 'nonzero', 'nonzero_static', 'norm', 'normal_',
    'not_equal', 'not_equal_', 'numel', 'numpy', 'orgqr', 'ormqr', 'outer', 'output_nr', 'permute', 'pin_memory', 'pinverse', 'polygamma', 'polygamma_', 'positive', 'pow', 'pow_',
    'prelu', 'prod', 'put', 'put_', 'q_per_channel_axis', 'q_per_channel_scales', 'q_per_channel_zero_points', 'q_scale', 'q_zero_point', 'qr', 'qscheme', 'quantile', 'rad2deg',
    'rad2deg_', 'random_', 'ravel', 'real', 'reciprocal', 'reciprocal_', 'record_stream', 'refine_names', 'register_hook', 'register_post_accumulate_grad_hook', 'reinforce', 'relu',
    'relu_', 'remainder', 'remainder_', 'rename', 'rename_', 'renorm', 'renorm_', 'repeat', 'repeat_interleave', 'requires_grad', 'requires_grad_', 'reshape', 'reshape_as', 'resize',
    'resize_', 'resize_as', 'resize_as_', 'resize_as_sparse_', 'resolve_conj', 'resolve_neg', 'retain_grad', 'retains_grad', 'roll', 'rot90', 'round', 'round_', 'row_indices', 'rsqrt',
    'rsqrt_', 'scatter', 'scatter_', 'scatter_add', 'scatter_add_', 'scatter_reduce', 'scatter_reduce_', 'select', 'select_scatter', 'set_', 'sgn', 'sgn_', 'shape', 'share_memory_',
    'short', 'sigmoid', 'sigmoid_', 'sign', 'sign_', 'signbit', 'sin', 'sin_', 'sinc', 'sinc_', 'sinh', 'sinh_', 'size', 'slice_scatter', 'slogdet', 'smm', 'softmax', 'solve', 'sort',
    'sparse_dim', 'sparse_mask', 'sparse_resize_', 'sparse_resize_and_clear_', 'split', 'split_with_sizes', 'sqrt', 'sqrt_', 'square', 'square_', 'squeeze', 'squeeze_', 'sspaddmm',
    'std', 'stft', 'storage', 'storage_offset', 'storage_type', 'stride', 'sub', 'sub_', 'subtract', 'subtract_', 'sum', 'sum_to_size', 'svd', 'swapaxes', 'swapaxes_', 'swapdims',
    'swapdims_', 'symeig', 't', 't_', 'take', 'take_along_dim', 'tan', 'tan_', 'tanh', 'tanh_', 'tensor_split', 'tile', 'to', 'to_dense', 'to_mkldnn', 'to_padded_tensor', 'to_sparse',
    'to_sparse_bsc', 'to_sparse_bsr', 'to_sparse_coo', 'to_sparse_csc', 'to_sparse_csr', 'tolist', 'topk', 'trace', 'transpose', 'transpose_', 'triangular_solve', 'tril', 'tril_',
    'triu', 'triu_', 'true_divide', 'true_divide_', 'trunc', 'trunc_', 'type', 'type_as', 'unbind', 'unflatten', 'unfold', 'uniform_', 'unique', 'unique_consecutive', 'unsafe_chunk',
    'unsafe_split', 'unsafe_split_with_sizes', 'unsqueeze', 'unsqueeze_', 'untyped_storage', 'values', 'var', 'vdot', 'view', 'view_as', 'volatile', 'vsplit', 'where', 'xlogy', 'xlogy_',
    'xpu', 'zero_'
}

def find_torch_imports(content: str) -> Dict[str, str]:
    """查找所有的 torch 相关导入，并建立别名映射"""
    imports = {}
    try:
        tree = ast.parse(content)
        import_nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        import_nodes.sort(key=lambda node: (getattr(node, 'lineno', 0), getattr(node, 'col_offset', 0)))

        def remove_alias(alias_name: str):
            imports.pop(alias_name, None)
            # `import torch.nn as nn` used to add helper aliases such as `nn.nn`.
            for key in list(imports):
                if key.startswith(f"{alias_name}."):
                    imports.pop(key, None)

        for node in import_nodes:
            if isinstance(node, ast.Import):
                for name in node.names:
                    alias = name.asname or name.name
                    if name.name.startswith('torch'):
                        if name.asname:
                            imports[name.asname] = name.name
                            module_name = name.name.split('.')[-1]
                            imports[f"{name.asname}.{module_name}"] = name.name
                        imports[name.name] = name.name
                    else:
                        remove_alias(alias)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith('torch'):
                    for name in node.names:
                        if name.name == '*':
                            continue
                        if name.asname:
                            imports[name.asname] = f"{node.module}.{name.name}"
                        else:
                            imports[name.name] = f"{node.module}.{name.name}"
                        if not name.name.startswith('torch'):
                            imports[name.name] = f"{node.module}.{name.name}"
                        if '.' in name.name:
                            parts = name.name.split('.')
                            imports[parts[-1]] = f"{node.module}.{name.name}"
                        if node.module == 'torch.nn.functional':
                            imports[name.name] = f"torch.nn.functional.{name.name}"
                else:
                    for name in node.names:
                        if name.name == '*':
                            continue
                        remove_alias(name.asname or name.name)
    except SyntaxError:
        pass
    return imports

class TorchApiVisitor(ast.NodeVisitor):
    def __init__(self, imports: Dict[str, str]):
        super().__init__()
        self.imports = imports
        self.torch_apis: List[Dict[str, Any]] = []
        # 使用行号+列号+API名称进行精确去重，Tensor方法额外使用节点id
        self.processed_positions: Set[Any] = set()
        self.tensor_names: Set[str] = set()

    def _expr_to_name(self, node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parts = []
            curr = node
            while isinstance(curr, ast.Attribute):
                parts.insert(0, curr.attr)
                curr = curr.value
            if isinstance(curr, ast.Name):
                parts.insert(0, curr.id)
                return ".".join(parts)
        return None

    def _is_tensor_annotation(self, node: ast.AST) -> bool:
        api = self._resolve_api_from_node(node)
        return api == 'torch.Tensor'

    def _is_tensor_factory_api(self, api: Optional[str]) -> bool:
        if not api or not api.startswith('torch.'):
            return False
        if api == 'torch.Tensor':
            return True
        if api.startswith('torch.nn.functional.'):
            return True
        if api.startswith(('torch.nn.', 'torch.optim.', 'torch.distributed.', 'torch.utils.')):
            return False
        return True

    def _mark_tensor_target(self, target: ast.AST):
        name = self._expr_to_name(target)
        if name:
            self.tensor_names.add(name)
            return True
        return False

    def _mark_named_parameters_targets(self, target: ast.AST):
        if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) < 2:
            return
        self._mark_tensor_target(target.elts[-1])

    def _is_named_parameters_call(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            return False
        return node.func.attr == 'named_parameters'

    def _is_tensor_data_expr(self, node: ast.AST) -> bool:
        if not isinstance(node, ast.Attribute) or node.attr != 'data':
            return False
        return self._is_known_tensor_receiver(node.value)

    def _is_tensor_expr(self, node: ast.AST) -> bool:
        if self._is_tensor_data_expr(node) or self._is_known_tensor_receiver(node):
            return True
        if isinstance(node, ast.Call):
            return self._is_tensor_factory_api(self._resolve_api_from_node(node.func))
        return False

    def _mark_tensor_context_receivers(self, node: ast.AST):
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and child.attr in TENSOR_CONTEXT_ATTRS:
                self._mark_tensor_target(child.value)

    def _is_known_tensor_receiver(self, receiver: ast.AST) -> bool:
        name = self._expr_to_name(receiver)
        if not name:
            return False
        if name in self.tensor_names:
            return True
        parts = name.split('.')
        return any(".".join(parts[:idx]) in self.tensor_names for idx in range(1, len(parts) + 1))

    def _resolve_api_from_node(self, node: ast.AST) -> Optional[str]:
        """
        Resolves a node into a full torch API path using a non-recursive,
        whole-chain analysis to prevent cascading resolution errors.
        """
        if isinstance(node, ast.Name):
            # 处理简单的导入别名，如 `Linear`
            return self.imports.get(node.id)

        if isinstance(node, ast.Attribute):
            # 拆解
            # e.g., `self.bias.data.zero_` -> ['self', 'bias', 'data', 'zero_']
            chain = []
            curr = node
            while isinstance(curr, ast.Attribute):
                chain.insert(0, curr.attr)
                curr = curr.value
            
            if not isinstance(curr, ast.Name):
                # 对于复杂基类（非简单变量名），只有当最终属性是Tensor方法时，才认定为torch.Tensor方法
                if node.attr in TENSOR_METHODS and node.attr not in AMBIGUOUS_TENSOR_METHODS:
                    return f"torch.Tensor.{node.attr}"
                return None
            
            base_name = curr.id
            chain.insert(0, base_name)
            
            # 1. 检查链式调用的基类是否是已知的torch别名。
            if base_name in self.imports:
                module_path = self.imports[base_name]
                result = ".".join([module_path] + chain[1:])
                return result

            # 2. 检查是否是self.xxx的形式，排除直接的self属性访问，但保留链式调用
            if base_name == 'self':
                if len(chain) == 2:
                    # 直接的self.xxx形式（如self.relu, self.conv），排除
                    return None
                elif len(chain) > 2:
                    # 链式调用（如self.weight.abs, self.bias.data.zero_），保留
                    # 检查最后的属性是否为tensor方法
                    last_attr = chain[-1]
                    if last_attr in TENSOR_METHODS:
                        if last_attr in AMBIGUOUS_TENSOR_METHODS and not self._is_known_tensor_receiver(node.value):
                            return None
                        return f"torch.Tensor.{last_attr}"

            # 3. 只有在base不是已知非torch模块时，才检查是否为tensor方法
            # 排除常见的非torch模块，避免误识别
            known_non_torch_modules = {
                'math', 'numpy', 'np', 'random', 'os', 'sys', 'json', 'pickle', 
                'collections', 'itertools', 'functools', 'operator', 'typing',
                'datetime', 'time', 're', 'urllib', 'requests', 'pandas', 'pd'
            }
            
            if base_name not in known_non_torch_modules:
                # 只有在base不是已知非torch模块时，才检查最后的属性是否为tensor方法
                last_attr = chain[-1]
                if last_attr in TENSOR_METHODS:
                    if last_attr in AMBIGUOUS_TENSOR_METHODS and not self._is_known_tensor_receiver(node.value):
                        return None
                    result = f"torch.Tensor.{last_attr}"
                    return result

        return None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        for arg in list(node.args.args) + list(node.args.kwonlyargs):
            if arg.annotation and self._is_tensor_annotation(arg.annotation):
                self.tensor_names.add(arg.arg)
        if node.args.vararg and node.args.vararg.annotation and self._is_tensor_annotation(node.args.vararg.annotation):
            self.tensor_names.add(node.args.vararg.arg)
        if node.args.kwarg and node.args.kwarg.annotation and self._is_tensor_annotation(node.args.kwarg.annotation):
            self.tensor_names.add(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.annotation and self._is_tensor_annotation(node.annotation):
            self._mark_tensor_target(node.target)
        elif node.value is not None and self._is_tensor_factory_api(self._resolve_api_from_node(node.value.func) if isinstance(node.value, ast.Call) else None):
            self._mark_tensor_target(node.target)
        elif node.value is not None and self._is_tensor_data_expr(node.value):
            self._mark_tensor_target(node.target)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        api = self._resolve_api_from_node(node.value.func) if isinstance(node.value, ast.Call) else None
        if self._is_tensor_factory_api(api) or self._is_tensor_data_expr(node.value):
            for target in node.targets:
                self._mark_tensor_target(target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For):
        if self._is_named_parameters_call(node.iter):
            self._mark_named_parameters_targets(node.target)
        self.generic_visit(node)

    def collect_tensor_bindings(self, tree: ast.AST):
        """Pre-pass: infer Tensor parameters from local call sites before visiting function bodies."""
        function_params: Dict[str, List[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                function_params[node.name] = params
                param_names = set(params)
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Attribute) and
                        child.attr in FUNCTION_PARAM_TENSOR_HINTS and
                        isinstance(child.value, ast.Name) and
                        child.value.id in param_names
                    ):
                        self.tensor_names.add(child.value.id)
            elif isinstance(node, ast.For) and self._is_named_parameters_call(node.iter):
                self._mark_named_parameters_targets(node.target)
            elif isinstance(node, ast.Call):
                api = self._resolve_api_from_node(node.func)
                if api and api.startswith('torch.'):
                    for arg in node.args:
                        self._mark_tensor_context_receivers(arg)
                    for keyword in node.keywords:
                        if keyword.value is not None:
                            self._mark_tensor_context_receivers(keyword.value)

        changed = True
        while changed:
            changed = False
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                    continue
                params = function_params.get(node.func.id)
                if not params:
                    continue
                for arg_node, param_name in zip(node.args, params):
                    if self._is_tensor_expr(arg_node) and param_name not in self.tensor_names:
                        self.tensor_names.add(param_name)
                        changed = True

    def add_api(self, api: Optional[str], node: ast.AST):
        # 使用行号+列号+API名称进行精确去重
        if api and hasattr(node, 'lineno') and hasattr(node, 'col_offset'):
            # 对于Tensor方法，使用节点id作为额外的区分
            if api and api.startswith('torch.Tensor.'):
                position_key = (node.lineno, node.col_offset, api, id(node))
            else:
                position_key = (node.lineno, node.col_offset, api)
                
            if position_key not in self.processed_positions:
                self.torch_apis.append({
                    'api': api, 
                    'line': node.lineno,
                    'col_offset': node.col_offset  # 添加列号信息
                })
                self.processed_positions.add(position_key)

    def visit_Call(self, node: ast.Call):
        # 处理当前调用
        api = self._resolve_api_from_node(node.func)
        self.add_api(api, node.func)
        
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # 防止与visit_Call重复计数，但允许独立的属性访问
        parent = getattr(node, 'parent', None)

        # 如果这个attribute是Call的func部分，让visit_Call处理
        if isinstance(parent, ast.Call) and parent.func == node:
            self.generic_visit(node)
            return

        # 如果这是属性链的中间部分，跳过（只处理最终的属性）
        if isinstance(parent, ast.Attribute) and parent.value == node:
            self.generic_visit(node)
            return
        
        # 处理独立的属性访问和属性链的最终部分
        api = self._resolve_api_from_node(node)
        self.add_api(api, node)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        # 处理类定义中的基类
        for base in node.bases:
            api = self._resolve_api_from_node(base)
            self.add_api(api, base)
        
        self.generic_visit(node)

def find_torch_usage(content: str, imports: Dict[str, str]) -> List[dict]:
    """使用 TorchApiVisitor 查找 torch API 的使用情况"""
    try:
        tree = ast.parse(content)
        # Add parent pointers to tree
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node
        
        visitor = TorchApiVisitor(imports)
        visitor.collect_tensor_bindings(tree)
        visitor.visit(tree)
        return visitor.torch_apis
    except SyntaxError:
        return []

def process_file(file_path: str) -> List[dict]:
    """处理单个文件，返回 torch API 使用列表"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        imports = find_torch_imports(content)
        apis = find_torch_usage(content, imports)
        
        for api_info in apis:
            api_info['file'] = file_path
        
        return apis
    except Exception as exc:
        print(f"跳过文件 {file_path}: {exc}")
        return []

def process_directory(directory_path: str) -> List[dict]:
    """处理目录，返回所有文件中的 torch API 使用列表"""
    all_apis = []
    skip_dirs = {'.git', '.hg', '.svn', '__pycache__', '.pytest_cache', '.mypy_cache', 'converted_files'}
    for root, dirs, files in os.walk(directory_path):
        dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                all_apis.extend(process_file(file_path))
    return all_apis

def main():
    """主函数，处理指定目录或文件并输出 JSON 报告和 Excel 报告"""
    import json
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # 默认扫描当前目录下的所有Python文件
        target_path = '.'
    
    print(f"开始扫描: {target_path}")
    
    # 判断是文件还是目录
    if os.path.isfile(target_path):
        results = process_file(target_path)
    else:
        results = process_directory(target_path)
    
    # --- 步骤1: 生成供机器读取的 JSON 报告 ---
    json_output_filename = 'api_report.json'
    with open(json_output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
    print(f"扫描完成，生成的JSON文件已保存到：{json_output_filename}")

    # --- 步骤2: 生成供人工核对的 Excel 文件 ---
    try:
        df = pd.DataFrame(results)
        if results:
            # 重命名并排序，方便阅读
            df = df.rename(columns={'file': '文件', 'line': '行号', 'api': '接口'})
            df = df[['文件', '行号', '接口']]
            df = df.sort_values(by=['文件', '行号']).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=['文件', '行号', '接口'])

        excel_output_filename = 'api_report_check.xlsx'
        df.to_excel(excel_output_filename, index=False)
        print(f"为方便核对，Excel报告已保存到: {excel_output_filename}")
    except Exception as e:
        print(f"生成Excel报告时出错: {e}")

    print(f"共找到 {len(results)} 个API调用。")

if __name__ == "__main__":
    main()
