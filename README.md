# DABS 昇腾复现

## 第一次运行：按顺序执行这三条命令

在 **昇腾 Linux 宿主机的项目目录**执行。

**1. 自动安装并启动原生适配：**

```bash
bash run_native.sh install --verify-restart
```

先核对四个服务的版本和源码哈希，再按 0→1→2→3 **逐个安装、重启原有容器并等待就绪**。已经安装且加载了相同适配的副本会跳过。`--verify-restart` 明确执行副本 0 的重启失效测试，因此首次安装后副本 0 会额外重启一次。请在这四个服务没有其它任务时运行；检测到忙碌、未知负载或源码不匹配就停止。不会创建新模型服务、升级推理栈或改变显存配置。每副本默认最多等待 900 秒，备份和回滚命令自动保存。

**2. 一行启动测试（首次包含缓存失效验证）：**

```bash
bash run_native.sh test --invalidate-cache
```

自动依次执行：四服务原生能力检查 → 单副本测试 → 四副本测试 → 独立标定 → 清空副本 0 前缀缓存并验证证据失效 → 原始证据验收。终端会显示当前步骤；遇到失败就停止，并保存原因和下一步命令。只有验收 `pass: true`、`live_verified: true`、`missing: []` 才显示测试 `passed`。

**3. 测试通过后，一行启动全部实验：**

```bash
bash run_native.sh experiment
```

自动使用刚才通过的测试证据，重新核验当前服务，再运行 **全部 96 条件**，生成报告。RR／SLL／Ayo／Ours、种子、DAG 和执行顺序不变；每条件 64 工作流、216 请求、每请求输入 3584 token、总输出 13184 token。没有默认只跑 8 条，也不会接着旧 HTTP 实验续跑。失败、零偏离和负收益照实保留。

首次安装和状态失效验证完成后，日常仍是 `bash run_native.sh test`、`bash run_native.sh experiment` 两条命令。普通 `test` 不安装、不重启服务，也不主动清空缓存。

## 结果在哪里

每次启动都会打印完整结果路径，并创建新目录，旧结果不覆盖：

| 命令 | 保存位置 |
| --- | --- |
| `install` | `outputs/native_install_时间_编号/`，回滚命令在 `ROLLBACK.txt` |
| `test` | `outputs/native_test_时间_编号/` |
| `experiment` | `outputs/native_experiment_时间_编号/` |

**先看该目录下的 `SUMMARY.txt`**：中文状态、失败原因、下一条命令、各步骤日志路径都在这里；`run.json` 保存实际执行命令与时间。实验原始数据和报告在该目录的 `r234_native/` 中，报告为 `TEST_REPORT.md`。所有结果只保存为普通文本／JSON／CSV 等文件；按需复制文字，不打包、不上传。

## 首次使用

从 GitHub 的 **main** 下载到新目录，保留旧项目 `/data/wushibo/dabs-ascend-main-2026101602` 和旧结果。例如：

```bash
git clone -b main https://github.com/micavro/dabs-ascend.git dabs-native-v023 && cd dabs-native-v023
```

环境沿用现场：Linux、Bash、宿主机 Python ≥3.9、可用的 `sudo docker`；四个现有服务 `wushibo-qwen-small-npu0`～`3`、端口 8001～8004。默认模型路径 `/data/model/Qwen2.5-1.5B-Instruct`，不同则先设置 `MODEL_PATH`。无需升级推理栈或启动额外模型服务。

**更新代码不等于安装服务端适配。** 第一次先执行本页的 `install --verify-restart`，不要直接跳到 `test`。某个副本失败时，脚本保留已经完成的副本并停止；修复原因后可重新运行安装命令，已正确加载的副本会跳过。已有不同版本的补丁会拒绝覆盖，按该次 `ROLLBACK.txt` 选择对应副本回滚后再安装。单副本操作见 [首次部署与逐步操作](docs/native-operations.md)。已经部署过的目录更新代码时，保留其中的 `outputs/native-control.json` 和已有证据；密钥不要复制到聊天。

正式实验必须有全部原生证据才能启动。CPU 测试通过不代表真机验收通过；异步 Host-step 无法提供足够同口径标定样本时会停止，不能用 HTTP 时延替代。实现细节见 [原生机制说明](docs/native-v023.md)。

开发机 CPU 自检：`python tools/run_ascend.py test`。它与上面连接现场服务的 `bash run_native.sh test` 不同。
