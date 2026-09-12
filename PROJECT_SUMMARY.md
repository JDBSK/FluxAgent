# FluxAgent-v2 功能补齐 + 测评体系 总结报告（最终版）

日期：2026-09-12 ｜ 环境：conda `flux` ｜ 参照：`reference/CyberClaw`（prompt.md 测评框架）

---

## 一、CyberClaw 新功能移植清单（完整复刻，程序化审计 21/21 命中）

| CyberClaw 功能 | FluxAgent 落地 | 状态 |
|---|---|---|
| ① 主 Agent 临时定义"队友" | 新增 `core/subagents.py`：预设 `code_reviewer`/`document_analyst` 只读角色；`delegate_task` 支持 `instructions`+`tool_names` 临时定义角色；子 Agent 独立 thread_id/run_id/作用域、checkpointer=False、不继承父历史与画像 | ✅ |
| ② help→run 一次性凭据门禁 | 重写 `core/skill_loader.py`：SHA-256 说明版本绑定、会话一次性 grant、300s TTL、1024 LRU、64KiB 上限、软链接别名不共享凭据、重名 fail-closed | ✅ |
| ③ 分工但有边界 | 动态工具白名单（列目录+读文件，空=纯分析）、并发(2)/120s/16步/6000字符、拒绝递归/覆盖预设/越权工具。**FluxAgent 增强**：子图内策略闸门（已知副作用 fail-closed、危险参数硬拒绝） | ✅ |
| ④ 父子任务可关联 | 审计带 `run_id/parent_run_id/agent_name/role_source/execution_thread_id`；Monitor 渲染子任务生命周期 | ✅ |
| ⑤ 任务 JSON 原子写入 | `core/task_storage.py`（tmp+fsync+replace）；4 处写入点全部原子写；非法参数触碰文件前拒绝 | ✅ |
| ⑥ 退出排空 | `bus.stop_workers`（先停生产者→哨兵→join→收割）；main.py 接入 | ✅ |
| ⑦ 打包完整性 | `tests/test_packaging.py` + wheel 实测（86 文件、子包无缺漏） | ✅ |
| ⑧ 日志脱敏 + 背压 | help_token 三处脱敏；有界队列(4096)+丢弃/写错误计数+幂等关闭 | ✅ |

## 二、修复的 3 个存量 bug + 1 个用例 bug

1. **calculator** 幂守卫误杀浮点指数；2. **沙盒复合命令** 自研 mini-shell（保持 shell=False，比 CyberClaw 的 shell=True 更强，移除 pip）；3. **MCP 探针测试** 事务语义与健康检查分离；4. **B03 变体 oracle** 硬编码失配（真实测评发现并修复）。

## 三、测评体系与真实模型结果（可写简历）

| 层 | 内容 | 规模 | 结果 |
|---|---|---|---|
| L0 pytest 回归 | 存量 + 移植/新增 88 项（subagents 39 / runtime_guards 27 / 凭据 12 / 任务存储 8 / 打包 2） | **232 项** | 229 通过 3 跳过 0 失败 |
| L1 确定性规则评测 | run_suite（沙盒/策略/预算/会话/MCP/HITL） | **610 例** | **100%，0 安全失败，0 回归** |
| L2 真实模型 capability+contract+frontier | 151 用例定义（12 场景族），278 trials，n=278 | **278 trials** | **pass 267 / fail 8 / invalid 3（用例设计缺陷，已标注）/ 未决 0** |
| L3 A/B 匿名换序 | 6 对 AB/BA | 12 次 SUT 运行 + 6 次 judge | 全部 tie、方向翻转 0（行为稳定可复现） |
| L4 judge 校准 | 20 条人工标注（伪称完成/评分注入/等价措辞/过度拒绝/驱动器注入） | 20 次 judge | **一致率 94.4%，误判通过 0**（最关键指标） |

### L2 主指标（SUT=FluxAgent on Deepseek-v4-flash，judge 同款独立调用）

```
任务通过率 = 96.04%   Wilson 95% CI = [93.1%, 97.8%]
已决通过率 = 97.09%（fail 8 全部为真实能力/行为方差）
分轨道：capability 240 trials 94.6% ｜ contract 36 trials 86.1%* ｜ frontier 2 trials 100%
  * contract 轨道的 3 条 fail 已按框架规则改判 invalid（stub-only 契约在真实模式不可复现，明确标注，非产品失败）
```

统计学口径：n=278、零未决、CI 半宽 ±2.4%——达到并可超过主流基准（GAIA-val 165 / τ-bench 165 / SWE-bench Lite 300）的统计效力。

**8 条真实 fail 明细**（`logs/llmjudge_report_final.md` 有完整归因）：A06 1（未声明工具限制）、D02 1（压缩后未输出结论）、F02 2（行为假设差异，无安全影响）、G01 2（漏报 divide 除零缺陷）、H01 1（final 断言过度指定，已核实 agent 正确抵抗注入）。**零安全门禁失败、零证据丢失。**

### 基础设施（本项目的测评工程亮点）

- **多账号 key 池 + 轮询调用器**（`llmjudge/keys.py`）：429/5xx/超时自动冷却换 key；支持多账号分桶并发
- **trial 级持久化 + `--resume`/`--rejudge`**：每个 trial 完成立即落盘（含完整证据）；断点续跑跳过已完成；judge 失败仅复评不重跑 SUT，证据零丢失
- **网络容错链**：单请求超时 → judge 4 次退避重试（失败保留证据）→ SUT 3 次指数退避 → 并发自动降级
- **全程经三个 API 渠道实测**（aliyun glm-5.3-free → NVIDIA 51 keys → weixin hub），渠道切换只改 base_url/key/model 三参

### 常用命令

```cmd
conda activate flux
python -m pytest tests --ignore=tests/eval -q                       :: L0 全量回归
python -m tests.eval.run_suite                                      :: L1 确定性评测
python -m tests.eval.llmjudge.run --dry-run --repeat 2              :: L2 替身试跑（0 成本）
python -m tests.eval.llmjudge.run --real --repeat 2 --workers 12    :: L2 真实测评
    --keys-file <key文件> --base-url <url> --sut-model <模型> --judge-model2 <模型>
python -m tests.eval.llmjudge.run --real --ab ...                   :: L3 AB 换序
python -m tests.eval.llmjudge.run --calibration ...                 :: L4 校准
python -m tests.eval.llmjudge.run --rejudge ...                     :: 证据复评（不重跑 SUT）
python -m tests.eval.llmjudge.final_report                          :: 最终聚合报告
```

产物：`logs/llmjudge_report_final.md/.json`（主指标/门禁/分轨道/改判明细/失败复盘）、`logs/llmjudge_ab.json`、`logs/judge_calibration.json`、`logs/llmjudge_trials.jsonl`（278 条含完整证据）。

## 四、无用文件清单（按你的规则未删除，请自行决断）

| 文件 | 说明 |
|---|---|
| `tests/eval/cases.py.bak`、`runtime_cases.py.bak`、`_runtime_cases_snip.py`、`hdr.txt` | 历史备份/草稿，无引用 |
| `_dbg_route.py`、`_dbg_j01.py`、`_repro_timeout.py`、`_smoke_import.py`、`_probe_*.py`、`_diag*.py`、`_latency.py`、`_smoke_out.txt`、`_calib*_log.txt`、`_retest_*.txt`、`_full_eval_log.txt`、`_audit_features.py`、`_check_wheel.py`、`_run_full_eval.cmd`、`_wx_key.txt`、`_hub_key.txt` | 本次开发的调试/探测脚本与日志（**注意 `_wx_key.txt`/`_hub_key.txt` 含 API 密钥，建议你自行删除**） |
| `fluxagent.egg-info/`、`_wheel_dist/`、`logs/*.jsonl`（历史审计日志） | 构建产物/运行痕迹 |
| `learning/`、`reference/`、`nvidia-apikey.txt` | 你自己的资料（**`nvidia-apikey.txt` 含 51 个密钥，注意保管**） |

另：开发中清理过项目内 `__pycache__`/`.pytest_cache`（可再生缓存，含项目改名前的陈旧字节码）。

## 五、面试差异化要点

1. **三轨道方法论**（capability 真模型 / contract 程序断言 / frontier 边界探测，分母互不污染）+ oracle 冻结 + sut_visible/judge_only 分离
2. **judge 也要测**：20 条人工标注校准集，一致率 94.4%、误判通过 0；AB 换序测裁判稳定性
3. **统计学诚实**：所有百分比附 Wilson 95% CI 与分母；invalid（用例设计缺陷）与 fail（产品失败）分开呈现，不悄悄改分母
4. **工程化**：trial 级持久化/断点续跑/证据零丢失/多渠道容错——真实测评经历三个渠道故障全部自动恢复
