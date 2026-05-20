  # generic-pytorch-repo 迁移增强计划

  ## Summary

  在 ms-cli 中只增强 migrate-agent 的 generic-pytorch-repo 路由：HF transformers 和 HF diffusers 现有专
  项路线保持不变。新增 Go 原生内置工具完成 PyTorch API 扫描、映射匹配和迁移上下文生成；LLM 不直接调用
  SDK 子流程，而是使用现有 agent 能力读取上下文后通过 read/edit/write 直接修改源码，最后再次扫描验证残留
  Torch API。

  ## Key Changes

  - 新增两个 Go 内置 tool：
      - torch_migration_scan
          - 参数：path，默认 "."；include_tests 默认 true；limit 默认 100。
          - 行为：递归扫描 Python 文件，识别 PyTorch API，匹配内置映射表，生成临时运行目录。
          - 返回：run_id、扫描摘要、映射覆盖率、按文件聚合的 API 列表、临时产物路径。
      - torch_migration_file_context
          - 参数：run_id、file。
          - 行为：读取扫描产物，返回该文件的 API 列表、映射信息、相关源码片段和迁移提示。
  - Go 原生扫描实现：
      - 支持 import torch、import torch as th、import torch.nn as nn、import torch.nn.functional as F、
        from torch import xxx 等常见别名。
      - 识别 torch.*、torch.nn.*、torch.nn.functional.*、已知 torch.Tensor.* 方法。
      - Tensor 方法使用保守启发式，无法确认时标记低置信度，不直接用于自动机械替换。
  - 映射表处理：
      - 将现有 torch_to_mindspore_mapping.xlsx 转成内置 JSON，提交到 ms-cli。
      - JSON 字段固定为：torch_api、mindspore_api、health_rate、alternative、remark。
      - Go tool 只读内置 JSON，不引入 Excel 解析依赖。
  - 迁移流程调整：
      - migrate-agent 路由到 generic-pytorch-repo 时，必须先调用 torch_migration_scan。
      - 对需要修改的文件，调用 torch_migration_file_context 获取文件级迁移上下文。
      - agent 根据源码上下文和映射信息直接编辑源码，不使用旧的机械 convert_to_mindspore.py。
      - 修改后再次调用 torch_migration_scan，报告残留 Torch API、未映射 API、低置信度项和人工处理项。
  - 权限与产物：
      - 新 tool 视为只读分析工具，权限行为与 read/glob 对齐，默认允许。
      - 扫描产物默认写入系统临时目录，例如 mscli-migrate-*，不污染用户项目。
      - 最终回答中列出临时产物路径、修改文件、验证结果和剩余风险。

  ## Implementation Changes

  - 在 tools/ 下新增 migration tool 包，并在 internal/app/wire.go 的 initTools 注册。
  - 更新 permission 规则，使 torch_migration_scan 和 torch_migration_file_context 默认按只读工具处理。
  - 更新 migrate-agent：
      - SKILL.md 保持 top-level router 定位。
      - references/generic-pytorch.md 改成完整流程说明：scan -> file context -> agent edit -> rescan ->
        report。
      - references/migration-routing.md 明确只有 generic-pytorch-repo 路由使用这套工具链。
  - 删除或弱化 generic route 中对旧机械替换脚本的依赖；旧 hf_transformers_auto_convert.py 路线不变。
  - 生成并提交内置映射 JSON，来源记录为当前 torch_to_mindspore_mapping.xlsx。

  ## Test Plan

  - Go 单测：
      - import alias 识别：torch、th、nn、F、from torch import zeros。
      - API 命中和映射：可替换、无直接替换、有其他方案、未映射。
      - Tensor 方法低置信度标记，例如 .view()、.reshape()、.size()。
      - 忽略 .git、__pycache__、虚拟环境目录。
      - 临时产物生成和 run_id 读取。
      - 权限默认允许两个新 tool。
  - Skill 测试：
      - migrate-agent generic route 文档必须包含 torch_migration_scan、torch_migration_file_context、
        rescan 验证要求。
      - HF transformers/diffusers route 文档不被 generic 工具链替换。
  - 集成验证：
      - 构造小型 PyTorch repo，运行 /migrate 的 generic 场景。
      - 期望 agent 先扫描，再编辑，最后重新扫描并报告残留 API。
      - 运行 go test ./tools/... ./permission/... ./integrations/skills/... ./internal/app/...。

  ## Assumptions

  - 第一版不做 Pyright 类型推断，采用 Go 原生保守扫描；低置信度项交给 LLM 和人工判断。
  - 第一版不做程序化自动源码替换；源码改写由现有 agent 使用 read/edit/write 完成。
  - 第一版不新增 Claude/OpenAI 专用 SDK 流程；继续复用 ms-cli 已有 provider 和 agent loop。
  - 扫描产物放临时目录；最终报告中给出路径，但不要求用户项目内生成固定报告文件。
  - 映射 JSON 是第一版的默认唯一映射来源；后续如需热更新映射，再增加外部映射路径参数。