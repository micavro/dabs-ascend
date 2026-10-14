# DABS GPU39 independent reproduction bundle

## 从 GitHub 下载并在昇腾上运行

在之前 MFE 使用的 Linux / vLLM-Ascend 容器中执行。仓库公开，可免登录下载；模型权重需要已经放在机器上。

```bash
git clone --depth 1 https://github.com/micavro/dabs-ascend.git
cd dabs-ascend
MODEL_PATH=/实际路径/Qwen2.5-1.5B-Instruct bash run_ascend.sh
```

默认使用物理卡 `0,1,2,3`，每卡一个独立模型副本。入口会检查环境、启动服务、执行真实推理和工作流测试，并打印测试记录的路径。当前尚未在真实昇腾机器上验证。

后续更新：在仓库目录执行 `git pull --ff-only`，然后重新运行最后一行。若只缺控制端的 `httpx` / `numpy`，可按 [部署说明](docs/ascend-linux.md) 补充；不需要通过本项目安装或升级 torch、CANN、vLLM。

如果机器没有 Git，可从 [GitHub ZIP 下载入口](https://github.com/micavro/dabs-ascend/archive/refs/heads/main.zip) 下载，解压后进入 `dabs-ascend-main`，执行同一条启动命令。

历史 NVIDIA 工程说明见 **[README_先读我.md](README_先读我.md)**。

## Linux / Ascend 910B functional port

In the existing MFE vLLM-Ascend container, run from this directory:

```bash
MODEL_PATH=/x/Qwen2.5-1.5B-Instruct bash run_ascend.sh
```

Default: four independent single-NPU replicas (`0,1,2,3`), environment checks,
real inference canaries, Ascend-side calibration, then four-method MuSiQue smoke
tests. An eight-card host is sufficient. See [the Ascend guide](docs/ascend-linux.md)
for model paths, full runs, output files, and the explicit differences from the
GPU39 native experiment. No driver, CANN, torch or vLLM installation is performed.

Canonical local checks (no accelerator or model needed):

```bash
python tools/run_ascend.py test
# equivalent Linux entry
bash run_ascend.sh test
```

The portable method is named `DAG-Ascend`; it reuses the original candidate
generator and rollout evaluator but has HTTP-observed state and submission-only
ordering. It is **not** an exact native GPU39/DAG performance reproduction.
The generated acceptance report distinguishes functional tests from unsupported
native KV / engine-phase tests. Historical source archives remain unchanged.

To create a Linux transfer archive without prior outputs or caches:

```bash
python -X utf8 tools/package_ascend.py
```

This project contains exact copies of native scheduling/experiment source archives plus ported fixed-case orchestration, frozen data, offline reanalysis and plotting scripts, and an auditable provenance manifest. The live vLLM serving deployment, model weights, and some full raw logs are not bundled. **RR is not Preble.**
