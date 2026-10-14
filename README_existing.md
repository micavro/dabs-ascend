# 连接现场已有 Qwen 服务

在 Linux **宿主机**的项目根目录执行。现场项目路径是 `/data/wushibo/dabs-ascend-main`。

本版本已经合并补丁，不用再次覆盖那三个文件。如果旧目录通过 ZIP 下载，或曾手动添加补丁文件，建议下载最新 ZIP 并解压到新目录，保留原目录和 `outputs/`。这样不会被 Git 的同名未跟踪文件阻挡更新。

```bash
cd /data/wushibo/dabs-ascend-main
bash run_dabs_existing.sh check
bash run_dabs_existing.sh
```

`check` 只检查部署和 HTTP 健康；最后一行默认跑小测试，包含接口检查、JSON canary、tokenization 比对、标定，以及 2 个工作流的 4 种策略。

## 默认现场配置

| 项目 | 默认值 |
|---|---|
| 已运行模型容器 | `wushibo-qwen-small-npu0`～`7` |
| 模型目录（宿主机） | `/data/model/Qwen2.5-1.5B-Instruct` |
| HTTP 地址 | `127.0.0.1:8001`～`8008`，容器端口 `8000` |
| 控制端 Python | `/usr/local/python3.12.13/bin/python`，来自现场镜像 |
| 控制端每副本在途请求 | `2` |
| 服务端并发等参数 | 读取并记录已有配置，不修改；补丁说明中的现场并发上限为 `10` |

宿主机需有 `python3`（≥3.8）和 `sudo docker` 权限。临时控制端使用所选第一张卡对应容器的本地镜像，要求镜像内 Python ≥3.10，已有 `httpx`、`numpy`。模型需有完整权重、配置和 tokenizer，且目录可读。

控制端只挂载项目目录和只读模型目录，不映射 NPU、不挂载 Docker socket，不启动/停止已有模型服务。使用 host 网络访问这些服务；本项目不下载镜像或模型。

## 手敲常用命令

只使用部分卡（卡 4、5 对应端口 8005、8006，不依赖 0 号容器）：

```bash
DEVICE_IDS=4,5 bash run_dabs_existing.sh
```

模型位置不同时，先设置：

```bash
export MODEL_PATH=/实际模型目录
```

小测试通过后，四卡 B1–B4 工作流测试（32 个条件）：

```bash
DEVICE_IDS=0,1,2,3 bash run_dabs_existing.sh benchmark
```

其他参数：

| 用途 | 命令 |
|---|---|
| 代码自检，不连接 HTTP、不要求模型文件 | `bash run_dabs_existing.sh test` |
| 查看四卡网格计划，不执行推理 | `DEVICE_IDS=0,1,2,3 bash run_dabs_existing.sh plan --suite grid` |
| 四卡网格测试（512 个条件） | `DEVICE_IDS=0,1,2,3 bash run_dabs_existing.sh benchmark --suite grid` |
| 提高控制端在途上限（先确认服务承受得住） | `MAX_INFLIGHT=10 bash run_dabs_existing.sh` |
| 镜像内使用普通 Python 路径 | `CONTROLLER_PYTHON=python3 bash run_dabs_existing.sh` |

`test`、`plan` 仍复用本地镜像，因此需至少存在所选第一张卡对应容器，容器可以停止；不要求其余服务健康。开发机不使用 Docker 时可执行 `python tools/run_dabs_existing.py test` 或 `python tools/run_dabs_existing.py plan --devices 4,5 --suite smoke`。

环境变量 `CONTAINER_PREFIX` 可修改容器名前缀，`BASE_PORT` 可修改零号卡对应的起始端口。通过环境变量选择部署参数，脚本会让宿主机采集和控制端选择保持一致。

八卡 benchmark 是扩展条件；固定 `--suite case` 要求四卡，`grid` 仍只扫描 1–4 副本。

## 结果与失败定位

`ASCEND_REPORT=/project/outputs/.../TEST_REPORT.md` 对应宿主机项目 `outputs/.../TEST_REPORT.md`。

- `TEST_REPORT.md`、`acceptance.json`：检查和推理结果。
- `environment.json`、`servers.json`：容器、卡号、端口、模型上下文上限、已有服务参数。
- `error.txt`：控制端失败原因；模型服务错误查看 `sudo docker logs 容器名`。
- `cleanup.json`：确认没有停止已有模型服务。

新运行创建新结果目录，保留失败记录。退出或失败只结束临时控制端；同一项目的控制端锁会阻止重叠选卡的并行测试，但不能阻止其他项目向模型服务发请求。

物理映射证据来自宿主机 Docker 记录，`real_npu_verified=false` 表示控制端没有直接探测 NPU；HTTP 推理通过单独记录为 `http_inference_verified=true`。它不表示模型跑在 CPU 上。HTTP 无法证明服务实际权重和挂载目录逐字节相同，也无法证明硬件独占。
