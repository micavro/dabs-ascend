# 昇腾迁移版本地验证记录

验证日期：2026-10-14。执行环境：Windows、Python 3.12.7、httpx 0.28.1、numpy 1.26.4；Shell 入口由 Git Bash 执行。

| 检查 | 结果 | 范围 |
|---|---|---|
| `python tools/run_ascend.py test` | 21 项通过 | 包括原 MuSiQue 2 工作流 / 7 节点的四方法流程、模拟 HTTP、真实 spawn 规划子进程 |
| `PYTHON=... bash run_ascend.sh test` | 同一套 21 项通过 | 验证 Bash 入口、中文项目路径和 UTF-8；Python 仍运行于 Windows |
| `bash -n run_ascend.sh` | 通过 | Bash 语法 |
| 原始冻结源码身份 | 48 个文件匹配 | 使用原 `wfsched.legacy.inspect`，未修改预期哈希 |
| 原始数据及 manifest | 匹配 | `provenance/input_assets.json` 全部文件哈希 |
| Windows 执行 `check` | 按预期非零退出 | 明确报告不是 Linux、缺少 torch-npu/vLLM 和模型，不伪报昇腾成功 |

21 项包含：设备可见性、模型结构/缺失分片、端口占用、固定实验计划、标定数据有效性、原始源码身份、父答案与 UNKNOWN 传递、两代 JSON 约束接口、流结束和 token 数校验、四策略工作流、关键路径分项之和、条件超时、传输失败、过期计划丢弃、部分启动失败清理、只清理本次进程、禁止覆盖旧结果、禁止虚构原生缓存/严格复现成功。

CPU 测试中的生成器是测试替身，不暴露为真实运行选项。测试输出不代表真实 NPU 推理、答案质量或性能数字。

本机没有可用的 910B / CANN / vLLM-Ascend 服务；Docker Linux daemon 也未运行。因此没有执行真实 Linux NPU 服务启动或推理。原历史 `verify_sources.py --test-native` 在 Windows 因缺少 Linux `fcntl` 无法完整运行；其源码/数据校验已由独立只读检查通过，未对历史代码打补丁以绕过这一限制。

现场剩余验证：在原 MFE 容器中运行 `MODEL_PATH=/实际/Qwen2.5-1.5B-Instruct bash run_ascend.sh`，检查生成的 `TEST_REPORT.md`、`acceptance.json` 及 `server_npu*.log`。正式报告中的原生 KV、引擎阶段时间、完整队列重排与 GPU39 性能一致性仍不能由 HTTP 功能迁移版验证，详见 `ascend-linux.md`。
