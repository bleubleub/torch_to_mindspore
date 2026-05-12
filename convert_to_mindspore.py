import libcst as cst
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional, Union
import os
import json
import subprocess
import sys
import shutil


class TorchAliasVisitor(cst.CSTVisitor):
    """
    Visits the CST to find all aliases for PyTorch modules.
    e.g., `import torch.nn as nn` -> {'nn': 'torch.nn'}
    """
    def __init__(self) -> None:
        self.alias_map: Dict[str, str] = {}

    def visit_Import(self, node: cst.Import) -> None:
        for alias in node.names:
            if isinstance(alias.name, cst.Attribute) or alias.name.value == 'torch':
                full_module_path = self.get_full_name_for_node(alias.name)
                if full_module_path.startswith('torch'):
                    self.alias_map[alias.asname.name.value if alias.asname else full_module_path] = full_module_path

    def visit_ImportFrom(self, node: cst.ImportFrom) -> None:
        if node.module:
            module_path = self.get_full_name_for_node(node.module)
            if module_path.startswith('torch'):
                if isinstance(node.names, cst.ImportStar):
                    return
                
                for alias in node.names:
                    imported_name = self.get_full_name_for_node(alias.name)
                    full_api_path = f"{module_path}.{imported_name}"
                    self.alias_map[alias.asname.name.value if alias.asname else imported_name] = full_api_path

    @staticmethod
    def get_full_name_for_node(node: Union[cst.Name, cst.Attribute]) -> str:
        """
        Converts a CST Name or Attribute node to its full string representation.
        e.g., cst.Attribute(value=cst.Name("torch"), attr=cst.Name("nn")) -> "torch.nn"
        """
        if isinstance(node, cst.Name):
            return node.value
        elif isinstance(node, cst.Attribute):
            return f"{TorchAliasVisitor.get_full_name_for_node(node.value)}.{node.attr.value}"
        return ""

class APIConversionTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (cst.metadata.ParentNodeProvider, cst.metadata.PositionProvider)
    DIRECT_IMPORT_NAMES = {'Parameter', 'Cell', 'Tensor'}
    HEALTH_RATE_THRESHOLD = 0.3  # 30%

    def __init__(self, mapping_info: Dict[str, str], full_mapping_info: Dict[str, Any], apis_by_file: Dict[str, List[dict]]):
        super().__init__()
        self.mapping = mapping_info
        self.full_mapping_info = full_mapping_info
        self.modified = False
        self.processed_apis = []
        
        # 将API按(行号, 列号)组织，以便更精确匹配
        self.apis_by_position = {}
        current_file_apis = apis_by_file.get('current_file', [])
        for api_info in current_file_apis:
            line = api_info['line']
            col = api_info.get('col_offset', 0)
            position_key = (line, col)
            if position_key not in self.apis_by_position:
                self.apis_by_position[position_key] = []
            self.apis_by_position[position_key].append(api_info)
        
        self.apis_by_line = {}
        for api_info in current_file_apis:
            line = api_info['line']
            if line not in self.apis_by_line:
                self.apis_by_line[line] = []
            self.apis_by_line[line].append(api_info)
        
        self.needed_imports = set()

    def _should_convert_api(self, torch_api: str) -> tuple[bool, str, str]:
        """
        判断是否应该转换API，返回(是否转换, 可参考API, 备注)
        """
        if torch_api not in self.full_mapping_info:
            return False, "", ""
        
        info = self.full_mapping_info[torch_api]
        mindspore_api = info.get('mindspore_api', '')
        health_rate = info.get('health_rate', 1.0)
        suggested_api = info.get('suggested_api', None)
        remark = info.get('remark', None)
        
        # 判断mindspore_api是否有效
        has_valid_mindspore_api = mindspore_api and pd.notna(mindspore_api) and str(mindspore_api).strip()
        
        if suggested_api and not has_valid_mindspore_api:
            reference_api = suggested_api
            comment = "无可直接替换api，建议参考'其他替换方案'实现功能。"
            if remark:
                comment += remark
            return False, reference_api, comment
        
        # 如果健康度小于xx%，则不替换，可参考API列填入mindspore_api
        if health_rate < self.HEALTH_RATE_THRESHOLD:
            reference_api = mindspore_api if has_valid_mindspore_api else ""
            comment = "因该API健康度不足，未能通过脚本转换，需手动修改适配，建议关注XXX方面（待后期UT完成补完各维度分数）"
            if remark:
                comment += f"。{remark}"
            return False, reference_api, comment
        
        # 健康度符合且有有效的mindspore_api，可以转换
        if has_valid_mindspore_api:
            # 如果同时有可参考API，在备注中提及
            comment = ""
            if suggested_api:
                comment = f"也可参考'其他替换方案'实现功能。"
                if remark:
                    comment += remark
            elif remark:
                comment = remark
            return True, "", comment
        
        # 其他情况不转换，但如果有备注仍然要显示
        comment = remark if remark else ""
        return False, "", comment

    def _create_mindspore_node(self, ms_api: str) -> Optional[cst.BaseExpression]:
        if pd.isna(ms_api) or not isinstance(ms_api, str) or not ms_api:
            return None
        
        parts = ms_api.split('.')
        if len(parts) < 1 or parts[0] != 'mindspore':
            try:
                return cst.parse_expression(ms_api)
            except Exception:
                return None

        # Case 1: from mindspore import Parameter, Cell, Tensor
        if len(parts) == 2 and parts[1] in self.DIRECT_IMPORT_NAMES:
            name = parts[1]
            self.needed_imports.add(('mindspore', name, name))
            return cst.Name(name)

        # Case 2: from mindspore import nn, ops, mint
        if len(parts) > 2:
            alias = parts[1]
            self.needed_imports.add(('mindspore', alias, alias))
            
            node: cst.BaseExpression = cst.Name(alias)
            for part in parts[2:]:
                node = cst.Attribute(node, cst.Name(part))
            return node

        # Case 3: import mindspore -> mindspore.tensor, mindspore.float32, mindspore.Tensor.transpose
        if len(parts) == 2:
            self.needed_imports.add((None, 'mindspore', 'mindspore'))
            return cst.Attribute(value=cst.Name('mindspore'), attr=cst.Name(parts[1]))
        
        # Case 4: mindspore.Tensor.transpose (3 parts: mindspore, Tensor, transpose)
        if len(parts) == 3 and parts[1] == 'Tensor':
            self.needed_imports.add(('mindspore', 'Tensor', 'Tensor'))
            return None
            
        return cst.Name(parts[0])

    def _is_method_to_attribute_conversion(self, torch_api: str, mindspore_api: str) -> bool:
        """
        判断是否是torch.Tensor.size需要特殊转换的情况
        """
        return torch_api == 'torch.Tensor.size' and mindspore_api == 'mindspore.Tensor.shape'
    
    def _convert_tensor_size_call(self, original_node: cst.Call, torch_api: str, func_lineno: int, func_col_offset: int) -> cst.BaseExpression:
        """
        转换torch.Tensor.size调用的统一处理函数
        - size() 无参数 -> .shape (属性访问)
        - size(n) 有参数 -> .shape[n] (索引访问)
        """
        if not isinstance(original_node.func, cst.Attribute):
            return None
            
        # 无参数的size() -> .shape (属性访问)
        if len(original_node.args) == 0:
            new_attr = cst.Attribute(
                value=original_node.func.value,  # 保持原始对象 (如 logit)
                attr=cst.Name("shape")  # 改为 shape 属性
            )
            self.modified = True
            self.needed_imports.add(('mindspore', 'Tensor', 'Tensor'))
            return new_attr
            
        # 单参数的size(n) -> .shape[n] (索引访问)
        elif len(original_node.args) == 1:
            # 创建 obj.shape[n] 的索引访问
            shape_attr = cst.Attribute(
                value=original_node.func.value,  # 保持原始对象 (如 logit)
                attr=cst.Name("shape")  # shape 属性
            )
            # 获取原始参数
            index_arg = original_node.args[0].value
            # 创建索引访问：obj.shape[n]
            new_subscript = cst.Subscript(
                value=shape_attr,
                slice=[cst.SubscriptElement(slice=cst.Index(value=index_arg))]
            )
            self.modified = True
            self.needed_imports.add(('mindspore', 'Tensor', 'Tensor'))
            return new_subscript
            
        # 其他情况（多参数等）暂不转换，保持原样
        else:
            self.needed_imports.add(('mindspore', 'Tensor', 'Tensor'))
            return None

    def leave_Call(self, original_node: cst.Call, updated_node: cst.Call) -> cst.BaseExpression:
        # 获取func的位置，而不是整个Call的位置
        func_pos = self.get_metadata(cst.metadata.PositionProvider, original_node.func)
        func_lineno = func_pos.start.line
        func_col_offset = func_pos.start.column
        
        # 特殊处理：检查是否是super().forward()调用
        if isinstance(original_node.func, cst.Attribute):
            if (isinstance(original_node.func.value, cst.Call) and
                isinstance(original_node.func.value.func, cst.Name) and
                original_node.func.value.func.value == "super" and
                original_node.func.attr.value == "forward"):
                
                # 直接将super().forward()改为super().construct()
                self.modified = True
                new_func = cst.Attribute(
                    value=original_node.func.value,  # super()
                    attr=cst.Name("construct")
                )
                return updated_node.with_changes(func=new_func)
        
        # 统一的API处理逻辑
        if func_lineno in self.apis_by_line:
            for api_info in self.apis_by_line[func_lineno][:]:
                torch_api = api_info['api']
                
                # 检查是否匹配：func匹配或Tensor实例方法匹配
                is_func_match = self._func_matches_api(original_node.func, torch_api)
                is_tensor_method_match = (
                    torch_api.startswith('torch.Tensor.') and 
                    isinstance(original_node.func, cst.Attribute) and
                    original_node.func.attr.value == torch_api.split('.')[-1]
                )
                
                if is_func_match or is_tensor_method_match:
                    # 进行API转换判断
                    match_type = "func_matches" if is_func_match else "tensor_method"
                    api_result = self._process_api_conversion(torch_api, func_lineno, f"leave_Call_{match_type}")
                    self.processed_apis.append({k: v for k, v in api_result.items() if k not in ['is_convertible', 'replacement']})
                    
                    # 如果这个API可以转换
                    if api_result['is_convertible']:
                        # 特殊处理：torch.Tensor.size转换
                        if self._is_method_to_attribute_conversion(torch_api, api_result['replacement']):
                            result = self._convert_tensor_size_call(original_node, torch_api, func_lineno, func_col_offset)
                            if result is not None:
                                # 移除已处理的API
                                self.apis_by_line[func_lineno].remove(api_info)
                                position_key = (func_lineno, func_col_offset)
                                if position_key in self.apis_by_position and api_info in self.apis_by_position[position_key]:
                                    self.apis_by_position[position_key].remove(api_info)
                                return result
                        # 对于其他可转换的API，替换节点
                        elif not (api_result['replacement'] and api_result['replacement'].startswith('mindspore.Tensor.')):
                            self.modified = True
                            new_node = self._create_mindspore_node(api_result['replacement'])
                            if new_node:
                                # 移除已处理的API
                                self.apis_by_line[func_lineno].remove(api_info)
                                position_key = (func_lineno, func_col_offset)
                                if position_key in self.apis_by_position and api_info in self.apis_by_position[position_key]:
                                    self.apis_by_position[position_key].remove(api_info)
                                return updated_node.with_changes(func=new_node)
                        # 对于Tensor实例方法，只添加导入
                        else:
                            self.needed_imports.add(('mindspore', 'Tensor', 'Tensor'))
                    
                    # 移除已处理的API（无论是否转换）
                    self.apis_by_line[func_lineno].remove(api_info)
                    position_key = (func_lineno, func_col_offset)
                    if position_key in self.apis_by_position and api_info in self.apis_by_position[position_key]:
                        self.apis_by_position[position_key].remove(api_info)
                    break  # 每次只处理一个API调用

        return updated_node

    def _process_single_api(self, api_info: dict, lineno: int, source: str) -> Optional[cst.BaseExpression]:
        """处理单个API的公共方法"""
        torch_api = api_info['api']
        
        # 进行API转换判断
        api_result = self._process_api_conversion(torch_api, lineno, source)
        self.processed_apis.append({k: v for k, v in api_result.items() if k not in ['is_convertible', 'replacement']})
        
        # 如果这个API可以转换
        if api_result['is_convertible']:
            # torch.Tensor.size 应该在leave_Call中处理，这里跳过
            if self._is_method_to_attribute_conversion(torch_api, api_result['replacement']):
                return None  # 跳过，让leave_Call处理
            # 对于其他Tensor实例方法，不替换节点，只添加导入
            elif api_result['replacement'] and api_result['replacement'].startswith('mindspore.Tensor.'):
                self.needed_imports.add(('mindspore', 'Tensor', 'Tensor'))
                return None
            else:
                # 对于其他可转换的API，在这里处理独立的属性访问
                self.modified = True
                new_node = self._create_mindspore_node(api_result['replacement'])
                return new_node
        
        return None

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute) -> cst.BaseExpression:
        pos = self.get_metadata(cst.metadata.PositionProvider, original_node)
        lineno = pos.start.line
        col_offset = pos.start.column
        
        # 检查这个Attribute是否是函数调用的一部分（直接或间接）
        # 如果是，则应该由leave_Call处理，这里不处理
        try:
            current = original_node
            parent = self.get_metadata(cst.metadata.ParentNodeProvider, current)
            
            # 向上遍历，检查是否在Call节点的func中
            while parent:
                if isinstance(parent, cst.Call):
                    # 检查当前节点是否在Call的func路径中
                    if self._is_node_in_func_path(original_node, parent.func):
                        # 这是函数调用的一部分，跳过处理
                        return updated_node
                    break
                try:
                    current = parent
                    parent = self.get_metadata(cst.metadata.ParentNodeProvider, current)
                except Exception:
                    break
        except Exception:
            pass
        
        # 检查该位置的API，如果是函数调用相关的，跳过处理
        position_key = (lineno, col_offset)
        if position_key in self.apis_by_position:
            for api_info in self.apis_by_position[position_key]:
                torch_api = api_info['api']
                # 如果是函数调用类型的API，跳过处理，让leave_Call处理
                if any(keyword in torch_api for keyword in ['functional', 'nn.', 'ops.']):
                    return updated_node
        
        # 进行精确位置匹配（行号+列号）
        if position_key in self.apis_by_position:
            for api_info in self.apis_by_position[position_key][:]:
                # 检查这个API是否已经被处理过了（比如在leave_Call中）
                if api_info not in self.apis_by_position[position_key]:
                    continue
                
                result = self._process_single_api(api_info, lineno, "leave_Attribute_position")
                
                if result is not None:
                    # 移除已处理的API
                    self.apis_by_position[position_key].remove(api_info)
                    if lineno in self.apis_by_line and api_info in self.apis_by_line[lineno]:
                        self.apis_by_line[lineno].remove(api_info)
                    return result
                else:
                    # 如果返回None，也移除API避免重复处理
                    self.apis_by_position[position_key].remove(api_info)
                    if lineno in self.apis_by_line and api_info in self.apis_by_line[lineno]:
                        self.apis_by_line[lineno].remove(api_info)
                
                break
                
        return updated_node

    def leave_ClassDef(self, original_node: cst.ClassDef, updated_node: cst.ClassDef) -> cst.CSTNode:
        new_bases = list(updated_node.bases)
        bases_modified = False
        class_name = original_node.name.value

        for i, original_base in enumerate(original_node.bases):
            pos = self.get_metadata(cst.metadata.PositionProvider, original_base.value)
            lineno = pos.start.line
            
            # 处理api_report.json中识别的API
            if lineno in self.apis_by_line:
                # 处理该行的所有API
                for api_info in self.apis_by_line[lineno][:]:
                    torch_api = api_info['api']
                    
                    # 进行API转换判断
                    api_result = self._process_api_conversion(torch_api, lineno, "leave_ClassDef")
                    self.processed_apis.append({k: v for k, v in api_result.items() if k not in ['is_convertible', 'replacement']})

                    if api_result['is_convertible']:
                        self.modified = True
                        bases_modified = True
                        new_node = self._create_mindspore_node(api_result['replacement'])
                        if new_node:
                            new_bases[i] = updated_node.bases[i].with_changes(value=new_node)
                    
                    # 移除已处理的API
                    self.apis_by_line[lineno].remove(api_info)
                    break
        
        if bases_modified:
            return updated_node.with_changes(bases=tuple(new_bases))
            
        return updated_node

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.CSTNode:
        """处理方法定义，将所有forward方法改为construct方法"""
        if original_node.name.value == "forward":
            self.modified = True
            return updated_node.with_changes(name=cst.Name("construct"))
        
        return updated_node

    def _process_api_conversion(self, torch_api: str, lineno: int, naive_name: str = "") -> Dict[str, Any]:
        """处理API转换的通用方法"""
        is_convertible = torch_api in self.mapping
        replacement = self.mapping.get(torch_api)
        
        # 判断是否应该转换
        should_convert, reference_api, remark = self._should_convert_api(torch_api)
        
        # 获取健康度信息
        health_rate = ""
        if torch_api in self.full_mapping_info:
            health_rate_value = self.full_mapping_info[torch_api].get('health_rate', 1.0)
            if pd.notna(health_rate_value):
                health_rate = f"{health_rate_value:.0%}"  # 只显示整数百分比
        
        if is_convertible and not should_convert:
            is_convertible = False

        return {
            '源文件': None, 
            '源文件行': lineno, 
            'Torch API': torch_api,
            '是否替换': is_convertible, 
            '已替换MindSpore_API': replacement if is_convertible else '',
            'API健康度': health_rate,
            '其他替换方案': reference_api,
            '备注': remark,
            'is_convertible': is_convertible,
            'replacement': replacement
        }

    def _func_matches_api(self, func_node: cst.BaseExpression, torch_api: str) -> bool:
        """检查func节点是否匹配指定的torch API"""
        func_str = self._get_func_string(func_node)
        
        # 情况1：完全匹配（torch.log 匹配 torch.log）
        if func_str == torch_api:
            return True
            
        # 情况2：去掉torch.前缀匹配（nn.Linear 匹配 torch.nn.Linear）
        api_without_torch = torch_api.replace('torch.', '')
        if func_str == api_without_torch:
            return True
            
        # 情况3：处理别名情况（F.interpolate 匹配 torch.nn.functional.interpolate）
        # F.interpolate -> nn.functional.interpolate
        if func_str.startswith('F.'):
            expanded_func = func_str.replace('F.', 'nn.functional.')
            if expanded_func == api_without_torch:
                return True
                
        return False
    
    def _get_func_string(self, func_node: cst.BaseExpression) -> str:
        """将func节点转换为字符串表示"""
        if isinstance(func_node, cst.Name):
            return func_node.value
        elif isinstance(func_node, cst.Attribute):
            return f"{self._get_func_string(func_node.value)}.{func_node.attr.value}"
        else:
            return str(func_node)

    def _is_node_in_func_path(self, target_node: cst.CSTNode, func_node: cst.BaseExpression) -> bool:
        """检查target_node是否在func_node的路径中"""
        if target_node == func_node:
            return True
        
        if isinstance(func_node, cst.Attribute):
            return (self._is_node_in_func_path(target_node, func_node.value) or 
                    target_node == func_node.attr)
        
        return False

class APIConverter:
    def __init__(self, mapping_file: str, output_dir: str):
        self.full_mapping_info = {}  # 初始化full_mapping_info
        self.api_mapping = self._read_mapping(mapping_file)
        self.output_dir = Path(output_dir)
        self.all_processed_apis = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_file(): item.unlink()
                else:
                    import shutil
                    shutil.rmtree(item)
        self.output_dir.mkdir(exist_ok=True)

    def _read_mapping(self, file_path: str) -> Dict[str, str]:
        print(f"正在读取映射文件: {file_path}")
        try:
            df = pd.read_excel(file_path)
            required_columns = ['Torch_API', 'MindSpore_API']
            if not all(col in df.columns for col in required_columns):
                print(f"  - 错误: 映射文件 {file_path} 必须包含以下列: {required_columns}")
                print(f"  - 文件中找到的列为: {list(df.columns)}")
                return {}
            
            df.dropna(subset=['Torch_API'], inplace=True)
            
            # 读取完整的映射信息，包括健康度、可参考API、备注等
            self.full_mapping_info = {}
            for _, row in df.iterrows():
                torch_api = row['Torch_API']
                mindspore_api = row['MindSpore_API']
                health_rate = row.get('MindSpore API健康度', 1.0)  # 健康度列
                suggested_api = row.get('其他替代方案', None)  # 其他替代方案列
                remark = row.get('备注', None) if '备注' in df.columns else None  # 备注列（可选）
                
                # 存储完整信息
                self.full_mapping_info[torch_api] = {
                    'mindspore_api': mindspore_api,
                    'health_rate': health_rate if pd.notna(health_rate) else 1.0,
                    'suggested_api': suggested_api if pd.notna(suggested_api) else None,
                    'remark': remark if pd.notna(remark) else None
                }
            
            # 原有的简单映射（为了保持兼容性）
            mapping = df.set_index('Torch_API')['MindSpore_API'].to_dict()
            final_mapping = {k: v for k, v in mapping.items() if isinstance(v, str)}
            
            print(f"读取完成，读取到 {len(self.full_mapping_info)} 个API映射关系")
            return final_mapping
        except FileNotFoundError:
            print(f"警告: 映射文件未找到于 {file_path}, 将使用空映射.")
            return {}

    def convert(self, target_path: str, api_report_file: str):
        target_path = Path(target_path)
        
        # 检查目标路径是否存在
        if not target_path.exists():
            print(f"错误: 目标路径 {target_path} 不存在")
            return
        
        try:
            with open(api_report_file, 'r', encoding='utf-8') as f:
                api_report = json.load(f)
        except FileNotFoundError:
            print(f"错误: API报告文件 {api_report_file} 不存在")
            return
        except json.JSONDecodeError:
            print(f"错误: API报告文件 {api_report_file} 格式错误")
            return

        apis_by_file: Dict[str, List[Dict[str, Any]]] = {}
        for api_info in api_report:
            file = api_info['file']
            if file not in apis_by_file:
                apis_by_file[file] = []
            apis_by_file[file].append(api_info)

        # 获取所有文件（包括非Python文件）
        if target_path.is_file():
            all_files = [target_path]
        else:
            all_files = [p for p in target_path.rglob('*') if p.is_file()]
        
        if not all_files:
            print(f"警告: 在 {target_path} 中没有找到文件")
            return
        
        # 分离Python文件和非Python文件
        py_files = [f for f in all_files if f.suffix == '.py']
        non_py_files = [f for f in all_files if f.suffix != '.py']
        
        total_files = len(all_files)
        print(f"找到 {len(py_files)} 个Python文件需要处理，{len(non_py_files)} 个非Python文件将原样复制")
        
        # 处理Python文件
        for i, file_path in enumerate(py_files):
            progress = i + 1
            bar_length = 40
            filled_length = int(bar_length * progress // len(py_files))
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            # 显示进度
            sys.stdout.write(f'\r文件处理中: [{bar}] {progress}/{len(py_files)} - {file_path.name}      ')
            sys.stdout.flush()
            
            # 获取该文件的API信息
            apis_in_this_file = apis_by_file.get(str(file_path), [])
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()
            except UnicodeDecodeError:
                print(f"\n警告: 无法读取文件 {file_path} (编码问题)，跳过")
                continue
            except Exception as e:
                print(f"\n警告: 无法读取文件 {file_path}: {e}，跳过")
                continue

            # 计算相对路径，保持目录结构
            if target_path.is_file():
                relative_path = file_path.name
            else:
                relative_path = file_path.relative_to(target_path)
            
            output_path = self.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 没有找到API的文件，直接复制
            if not apis_in_this_file:
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(source_code)
                except Exception as e:
                    print(f"\n警告: 无法写入文件 {output_path}: {e}")
                continue

            try:
                source_tree = cst.parse_module(source_code)
                wrapper = cst.metadata.MetadataWrapper(source_tree)
                transformer = APIConversionTransformer(self.api_mapping, self.full_mapping_info, {'current_file': apis_in_this_file.copy()})
                
                modified_tree = wrapper.visit(transformer)
                
                # 更新处理过的API信息
                for api_info in transformer.processed_apis:
                    api_info['源文件'] = str(relative_path)
                self.all_processed_apis.extend(transformer.processed_apis)
                                
                # 生成输出代码
                output_code = source_code
                if transformer.modified:
                    importer = ImportTransformer(transformer.needed_imports)
                    final_tree = cst.metadata.MetadataWrapper(modified_tree).visit(importer)
                    output_code = final_tree.code
                    
                # 写入输出文件
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(output_code)
                except Exception as e:
                    print(f"\n警告: 无法写入文件 {output_path}: {e}")
                    
            except Exception as e:
                print(f"\n警告: 处理文件 {file_path} 时出错: {e}，跳过")
                # 如果处理失败，尝试直接复制原文件
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(source_code)
                except Exception as copy_error:
                    print(f"警告: 无法复制文件 {file_path}: {copy_error}")
                continue
        
        # 处理非Python文件（原样复制并在报告中标注）
        for file_path in non_py_files:
            # 计算相对路径，保持目录结构
            if target_path.is_file():
                relative_path = file_path.name
            else:
                relative_path = file_path.relative_to(target_path)
            
            output_path = self.output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # 原样复制文件
                shutil.copy2(file_path, output_path)
                
                # 在报告中添加记录，标注为"未转换"
                self.all_processed_apis.append({
                    '源文件': str(relative_path),
                    '源文件行': '',
                    'Torch API': '',
                    '是否替换': False,
                    '已替换MindSpore_API': '',
                    'API健康度': '',
                    '其他替换方案': '',
                    '备注': '非Python文件，未转换'
                })
            except Exception as e:
                print(f"\n警告: 无法复制非Python文件 {file_path}: {e}")

        sys.stdout.write('\n')
        sys.stdout.flush()

        self.print_summary()
        self.save_report()

    def save_report(self):
        if not self.all_processed_apis: 
            # 即使没有API，如果有非Python文件记录，也要保存报告
            return
        report_path = self.output_dir / "conversion_report.xlsx"
        df = pd.DataFrame(self.all_processed_apis)
        
        # 确保所有必需的列都存在
        required_columns = ['源文件', '源文件行', 'Torch API', '是否替换', '已替换MindSpore_API', 'API健康度', '其他替换方案', '备注']
        for col in required_columns:
            if col not in df.columns:
                df[col] = ''
        
        df = df[required_columns]
        df = df.fillna('')
        
        # 对于非Python文件，按源文件排序；对于Python文件，按源文件和行号排序
        df['源文件行_num'] = df['源文件行'].apply(lambda x: int(x) if str(x).isdigit() else 0)
        df.sort_values(by=['源文件', '源文件行_num'], inplace=True)
        df.drop('源文件行_num', axis=1, inplace=True)
        df.to_excel(report_path, index=False)
        print(f"详细转换报告已保存到: {report_path}")

    def print_summary(self):
        if not self.all_processed_apis:
            print("\n未检测到任何PyTorch API调用.")
            return
        df = pd.DataFrame(self.all_processed_apis)
        modified_files_count = df[df['是否替换'] == True]['源文件'].nunique()
        print(f"\n总共修改了 {modified_files_count} 个文件.")
        print(f"共检测到 {len(df)} 个PyTorch API调用.")


class ImportTransformer(cst.CSTTransformer):
    """Adds the necessary mindspore imports to the top of the file."""
    def __init__(self, needed_imports: Set[Tuple[str, str, str]]):
        # e.g. {('mindspore', 'nn', 'nn'), ('mindspore', 'Parameter', 'Parameter')}
        self.needed_imports = needed_imports

    def leave_Module(self, original_node: cst.Module, updated_node: cst.Module) -> cst.Module:
        direct_imports = []
        from_imports_from_module: Dict[str, List[cst.ImportAlias]] = {}

        sorted_imports = sorted(list(self.needed_imports), key=lambda x: (x[0] is not None, x[0] or "", x[1]))

        for module, name, alias in sorted_imports:
            if module is None:
                direct_imports.append(cst.SimpleStatementLine(
                    body=[cst.Import(names=[cst.ImportAlias(name=cst.Name(name))])]
                ))
                continue
            
            if module not in from_imports_from_module:
                from_imports_from_module[module] = []
            
            import_alias = cst.ImportAlias(name=cst.Name(name))
            # Only add 'as' if the alias is different from the name
            if alias and alias != name:
                import_alias = import_alias.with_changes(asname=cst.AsName(name=cst.Name(alias)))
            from_imports_from_module[module].append(import_alias)

        from_import_nodes = []
        for module_name, aliases in sorted(from_imports_from_module.items()):
            try:
                # module_name can be "mindspore" or "mindspore.nn"
                module_node = cst.parse_expression(module_name)
                
                from_import_nodes.append(cst.SimpleStatementLine(
                    body=[cst.ImportFrom(
                        module=module_node,
                        # sort for consistent output
                        names=sorted(aliases, key=lambda x: x.name.value) 
                    )]
                ))
            except Exception:
                continue
        
        all_new_imports = direct_imports + from_import_nodes

        insert_idx = 0
        found_non_import = False
        for i, stmt in enumerate(updated_node.body):
            is_import = isinstance(stmt, cst.SimpleStatementLine) and isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))
            is_docstring = (i == 0 and isinstance(stmt.body[0], cst.Expr) and isinstance(stmt.body[0].value, cst.SimpleString))

            if not is_import and not is_docstring:
                 insert_idx = i
                 found_non_import = True
                 break
        
        if not found_non_import:
            insert_idx = len(updated_node.body)

        new_body = list(updated_node.body)
        if insert_idx > 0 and len(all_new_imports) > 0:
            if not isinstance(new_body[insert_idx-1], cst.EmptyLine):
                all_new_imports.insert(0, cst.EmptyLine())
        
        final_imports = all_new_imports

        new_body[insert_idx:insert_idx] = final_imports
        return updated_node.with_changes(body=new_body)


def main():
    # 1. 扫描files_to_be_converted文件夹中的所有Python文件
    scanner_script = 'find_torch_api.py'
    api_report_file = 'api_report.json'
    files_to_convert_dir = 'files_to_be_converted'
    
    print(f"--- 步骤 1: 检测源文件API调用 ({scanner_script}) ---")
    print(f"扫描目录: {files_to_convert_dir}")
    
    # 检查files_to_be_converted目录是否存在
    if not Path(files_to_convert_dir).exists():
        print(f"错误: {files_to_convert_dir} 目录不存在")
        return
    
    # 运行扫描器，扫描整个files_to_be_converted目录
    subprocess.run(['python', scanner_script, files_to_convert_dir], check=True)
    print("=============== 检测完成 ===============")

    # 2. 运行转换器，使用扫描器的报告
    mapping_file = 'torch_to_mindspore_mapping.xlsx'
    output_dir = 'converted_files'
    
    print(f"--- 步骤 2: 替换API (convert_to_mindspore.py) ---")
    converter = APIConverter(mapping_file, output_dir)
    converter.convert(files_to_convert_dir, api_report_file)
    print(f"转换后的文件已保存到：{output_dir} 文件夹中")
    print("=============== 转换完成 ===============")

if __name__ == "__main__":
    main()