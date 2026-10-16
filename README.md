# DABS 昇腾复现

## 只记住这两条命令

在 **昇腾 Linux 宿主机的项目目录**执行。

**一行启动测试：**

```bash
bash run_native.sh test
```

自动依次执行：四服务原生能力检查 → 单副本测试 → 四副本测试 → 独立标定 → 原始证据验收。终端会显示当前步骤；遇到失败就停止，并保存原因和下一步命令。只有验收 `pass: true`、`live_verified: true`、`missing: []` 才显示测试 `passed`。

**测试通过后，一行启动全部实验：**

```bash
bash run_native.sh experiment
```

自动使用刚才通过的测试证据，重新核验当前服务，再运行 **全部 96 条件**，生成报告。RR／SLL／Ayo／Ours、种子、DAG 和执行顺序不变；每条件 64 工作流、216 请求、每请求输入 3584 token、总输出 13184 token。没有默认只跑 8 条，也不会接着旧 HTTP 实验续跑。失败、零偏离和负收益照实保留。

## 结果在哪里

每次启动都会打印完整结果路径，并创建新目录，旧结果不覆盖：

| 命令 | 保存位置 |
| --- | --- |
| `test` | `outputs/native_test_时间_编号/` |
| `experiment` | `outputs/native_experiment_时间_编号/` |

**先看该目录下的 `SUMMARY.txt`**：中文状态、失败原因、下一条命令、各步骤日志路径都在这里；`run.json` 保存实际执行命令与时间。实验原始数据和报告在该目录的 `r234_native/` 中，报告为 `TEST_REPORT.md`。所有结果只保存为普通文本／JSON／CSV 等文件；按需复制文字，不打包、不上传。

## 首次使用

从 GitHub 的 **main** 下载到新目录，保留旧项目 `/data/wushibo/dabs-ascend-main-2026101602` 和旧结果。例如：

```bash
git clone -b main https://github.com/micavro/dabs-ascend.git dabs-native-v023 && cd dabs-native-v023
```

环境沿用现场：Linux、Bash、宿主机 Python ≥3.9、可用的 `sudo docker`；四个现有服务 `wushibo-qwen-small-npu0`～`3`、端口 8001～8004。默认模型路径 `/data/model/Qwen2.5-1.5B-Instruct`，不同则先设置 `MODEL_PATH`。无需升级推理栈或启动额外模型服务。

**第一次需要将原生适配加载到四个服务。** 直接运行上面的 `test` 会检查前置条件；未安装、未重启加载或缺少验收证据时，会明确停止并提示。具体安装、手动重启及回滚见 [首次部署与逐步操作](docs/native-operations.md)。已经部署过的目录更新代码时，保留其中的 `outputs/native-control.json` 和已有证据；密钥不要复制到聊天。

重启失效测试需要单独执行，普通 `test` 不会自动重启服务。首次缺少缓存失效证据时，可明确执行以下命令，它会在测试过程中排空并清空 **副本 0 的前缀缓存**：

```bash
bash run_native.sh test --invalidate-cache
```

正式实验必须有全部原生证据才能启动。CPU 测试通过不代表真机验收通过；异步 Host-step 无法提供足够同口径标定样本时会停止，不能用 HTTP 时延替代。实现细节见 [原生机制说明](docs/native-v023.md)。

开发机 CPU 自检：`python tools/run_ascend.py test`。它与上面连接现场服务的 `bash run_native.sh test` 不同。
