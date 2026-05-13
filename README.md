# PyTorch to MindSpore 转换工具

## 功能特性

### 文件结构
```
torch_to_mindspore/
├── convert_to_mindspore.py          # 主转换脚本
├── find_torch_api.py                # API检测脚本
├── torch_to_mindspore_mapping.xlsx  # API映射表
├── files_to_be_converted/           # 待转换文件
│   ├── xxx.py
│   ├── xxx.py
│   └── xxx/
│       ├── __init__.py
│       └── xxx/
│           ├── __init__.py
│           └── xxx.py
└── converted_files/                 # 转换结果目录
    ├── conversion_report.xlsx       # 转换报告
    └── [转换后的文件（保持转换前文件结构）...]
```

## 使用方法

### 1. 准备文件
将要转换的PyTorch文件或文件夹放入 `files_to_be_converted` 目录中。

### 2. 运行转换
```bash
python convert_to_mindspore.py
```

### 3. 查看结果
`converted_files`目录中：   
- 转换完成的代码文件，保持原有命名，并保持原有的目录结构。    
- `conversion_report.xlsx` 为转换报表，包括每个API所在文件路径、所在行数、是否转换、替代方案及备注。请配合代码查看。    


## 转换流程

### 步骤1: API检测
- 扫描 `files_to_be_converted` 目录中的所有Python文件
- 检测PyTorch API调用
- 生成 `api_report.json` 和 `api_report_check.xlsx`（不用关注）

### 步骤2: API转换
- 读取API映射文件 `torch_to_mindspore_mapping.xlsx`
- 根据映射关系替换PyTorch API为MindSpore API
- 自动添加必要的MindSpore导入语句
- 生成转换后的文件


## 依赖包

```bash
pip install libcst pandas openpyxl pyright
```

`find_torch_api.py` 默认使用 Pyright 做类型辅助识别，以减少 `numpy.reshape`、`Enum.name` 等非 PyTorch 对象被误识别为 `torch.Tensor.*` 的情况。安装 `pyright` 后会提供 `pyright` 和 `pyright-langserver` 命令；如果未安装，脚本只扫描显式导入的 `torch.*`、`torch.nn.*`、`torch.nn.functional.*` 等 API，不猜测实例方法归属。

常用扫描命令：

```bash
python find_torch_api.py files_to_be_converted --mode pyright
python find_torch_api.py files_to_be_converted --mode static
```

### 仅根据 Torch API 表生成映射结果

如果已有一个只包含 `接口` 列的 Torch API 表格，可以直接追加 MindSpore 映射信息：

```bash
python map_api_table.py torch_api_list.xlsx
```

默认会在输入文件同目录生成 `torch_api_list_mindspore_mapping.xlsx`，并追加三列：

| 接口 | 可替换MindSpore接口 | 其他替换方案 | 备注 |
|------|---------------------|--------------|------|

也可以指定输出路径：

```bash
python map_api_table.py torch_api_list.xlsx -o output.xlsx
```

### 仅根据 Torch API 表生成映射结果

如果已有一个只包含 `接口` 列的 Torch API 表格，可以直接追加 MindSpore 映射信息：

```bash
python map_api_table.py torch_api_list.xlsx
```

默认会直接写回原表，并追加三列：

| 接口 | 可替换MindSpore接口 | 其他替换方案 | 备注 |
|------|---------------------|--------------|------|

如需保留原表，也可以指定输出路径另存：

```bash
python map_api_table.py torch_api_list.xlsx -o output.xlsx
```

## 注意事项

当前该工具仅支持API接口级别的转换

#### 1. torch.nn.Module与mindspore.nn.Cell中的方法
当前脚本仅对torch.nn.Module.construct()做了转换，转换后为mindspore.nn.Cell.forward()。其他方法需要手动转换。     

例如：在torch.nn.Module中的self.parameters 需要手动改写为 mindspore.nn.Cell中的self.trainable_params

#### 2. API中的参数
参数名不一致、mindspore API中的参数缺失问题，当前工具均不会处理。已知问题已记录在torch_to_mindspor_mapping.xlsx中的备注中。    
转换文件中的相关的API，会被记录在conversion_report中的备注中，请参考并手动修改。

例如：
| 源文件     | 源文件行 | Torch API| 是否替换 | 已替换MindSpore_API | API健康度 | 其他替换方案 | 备注                        |
|---------|------|----------------------|-------------------------------|----------------------|--------|-----------------|---------------------------|
| test.py | 25   | torch.zeros          | TRUE                          | mindspore.mint.zeros | 100%   |                 | ms缺参数                     |
| test.py | 25   | torch.finfo          | FALSE                         |                      | 100%   |                 | 已排进Q3需求，当前建议使用numpy.finfo |
| test.py | 26   | torch.Tensor.squeeze | TRUE  |  mindspore.Tensor.squeeze   | 100% |     |功能一致，参数名不同 |

#### 3. import导入部分

因为大概率会有torch API无可替代ms API的情况，为方便查看，原有的torch相关import不会被删除。只新增mindspore的import。需要手动适配完脚本后删除torch相关import。

#### 4. 代码注释中的API，工具会跳过，不识别也不替换。

#### 5. 小概率Tensor接口识别错误问题
偶尔一些python内置方法会被误识别为Tensor方法，出现在conversation_report中，不会影响脚本替换，请忽略。

例如：loss_dict.values()会被误识别为Tensor.values调用
```python
loss_dict = {}
...
loss = sum(loss_dict.values())
``` 

#### 6. conversion_report中的‘API健康度’列，为功能预埋，暂未启用请忽略。


## 转换示例

### 输入文件 (simple_test.py)
```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 5)
```

### 输出文件
```python
import torch
import torch.nn as nn

from mindspore import mint, nn

class SimpleModel(nn.Cell):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = mint.nn.Linear(10, 5)
``` # torch_to_mindspore
