# 01｜工具调用：AI 的“手”

## 先说大白话

大模型本身只会“说”，不会真的查时间、读文件、跑命令。工具调用就是给它一双手：

1. 你告诉它任务：“现在几点？”
2. 它看到工具说明书：`get_current_time` 能查时间，不需要参数。
3. 它决定：我要调用这个工具。
4. 它生成参数：`{}`。
5. 程序执行工具，把结果还给模型。
6. 模型根据结果回答你。

真正的难点不在“调用成功”，而在三个失败场景：

- 选错工具：该查文件，它去算数。
- 参数错误：文件路径不存在、格式不对。
- 执行失败：超时、权限拒绝、临时网络问题。

Agent 系统必须把这三类失败分开处理，不能一失败就无脑重试。

## FluxAgent 里的对应实现

| 概念 | 项目位置 | 作用 |
|---|---|---|
| 工具注册 | `fluxagent/core/tools/base.py` | 定义工具的统一结构 |
| 内置工具 | `fluxagent/core/tools/builtins.py` | 时间、计算器、MCP 安装、任务调度等 |
| 失败分类 | `fluxagent/core/recovery.py` | 区分 transient / permanent / policy |
| 重试决策 | `fluxagent/core/recovery.py` | retry / replan / abort |

重点看这几处：

- `get_current_time`：最简单的无参数工具。
- `calculator`：有参数工具，且强调“安全计算，不用 eval”。
- `install_mcp_server`：高复杂度工具，失败时不能把异常直接甩给用户。
- `classify_failure`：失败后决定重试、重规划还是终止。

## Demo 说明

运行：

```powershell
python learning/01-tool-calling/demo_tool_call.py
```

这个 Demo 不调用真实大模型，而是用代码模拟模型的“选择工具、生成参数、观察结果”过程，方便你先看清机制。

Demo 包含 4 个例子：

1. 查时间：无参数工具。
2. 算数：带参数工具。
3. 模拟网络超时：第一次失败，重试成功。
4. 路径越权：策略拒绝，禁止盲目重试。

## 你要掌握的 3 个面试点

1. **工具不是模型自己执行的**：模型只输出“工具名 + 参数”，真正执行靠外部程序。
2. **失败要分类**：临时错误可以重试；参数错误要重规划；权限错误重试也没用。
3. **工具说明书很重要**：description 写得越准确，模型选择越稳定；写得含糊，工具越多越容易选错。

## 下一课

工具调用只解决“单次动手”。下一课讲智能循环：Agent 如何接任务、制定计划、执行、观察结果、失败后修改计划，直到完成或主动停止。

## 课堂拆解：Demo 输出对应什么？

| Demo 输出 | 真实 Agent 里的含义 |
|---|---|
| `模型决策：调用 get_current_time` | LLM 读完工具 description 后，输出工具名 |
| `生成参数：{}` | LLM 按照 JSON Schema 生成参数，不是手写代码 |
| `执行结果：当前时间...` | 工具由外部 Python 程序执行，模型本身没有系统权限 |
| `观察到临时失败...重试` | transient failure，适合有限次数 retry |
| `越权拦截...禁止盲目重试` | policy failure，重试不会让权限变化，应该终止或换方案 |

## 和 FluxAgent 的逐步对照

1. **工具定义**：FluxAgent 用 `@fluxagent_tool` 把普通函数包装成统一工具，自动提取函数签名和 docstring。对应 Demo 里的 `Tool(name, description, schema, function)`。
2. **参数校验**：真实项目用 Pydantic / JSON Schema 校验；Demo 用 `required` 字段做最小校验。
3. **失败分类**：FluxAgent 的 `classify_failure` 会识别 policy、permanent、transient，再决定 retry、replan、abort。
4. **安全边界**：Demo 里的路径检查对应 `sandbox_tools.py` 的 workspace 限制；真实系统还有 Policy、审批、权限缓存。

## 课后作业

1. 修改 Demo，故意把 `expression` 传成 `"open('secret.txt')"`，观察策略拒绝。
2. 打开 `fluxagent/core/tools/builtins.py`，找到 `get_current_time` 和 `calculator`，比较它们的参数复杂度。
3. 打开 `fluxagent/core/recovery.py`，用大白话解释：为什么“文件不存在”不能靠重试解决？
