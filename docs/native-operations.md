# 原生适配部署与专项操作

推荐入口：`bash run_native.sh install` → `bash run_native.sh test` → `bash run_native.sh experiment`。在 Linux 宿主机的新项目目录运行。环境与结果位置见 [README](../README.md)。普通 test 连接现有服务，不重新安装、不重启、不主动清空缓存。

## 安装行为

安装命令先尝试无凭据的 `/dabs/v1/control`，读取真实 EngineCore/worker 能力和快照。可用副本跳过部署；其余副本检查四处实际 hook 插入点，缺少时报告文件与类/函数名。版本、源码和适配文件哈希仅用于排查，不作为兼容白名单。

旧认证接口会返回 403，因此需要一次适配器更新及服务重启。安装命令自动完成这一步，不读取、生成、恢复或复制控制密钥。旧部署整包保留在安装备份中，新适配不使用旧凭据。

待更新副本先检查实际 running/waiting；忙碌或未知时不重启。按 0→1→2→3 顺序更新，每个加载完成才继续。当前适配只要求实际支持 TP/PP/DP=1、单 full-attention KV group、FCFS、async、eager、APC、chunked prefill，且无 speculative/KV connector。资源从引擎读取；1024 步预算、128 块大小、10 服务序列数、4096 上下文、2 客户端在途分别处理。

安装只加四处 hook，备份 vendor 文件与之前的适配包，重复更新不会重复插入。默认每副本等待就绪最多 900 秒。回滚会先另存当前源码，再恢复备份，保留旧结果。可直接使用安装结果目录中的 `ROLLBACK.txt`，选择需要撤销的副本。

## 只操作副本 0

以下低层命令不自动重启，仅在需要分步部署时使用：

```bash
bash run_dabs_r234.sh native-install --replica 0
sudo docker restart wushibo-qwen-small-npu0
```

待健康接口恢复后：

```bash
bash run_dabs_r234.sh native-check
bash run_dabs_r234.sh native-probe-one
```

`native-check` 不发推理，只检查真实接口与配置。`native-probe-one` 才执行推理和队列动作；通过不等于整套 `live_verified`。队列负载仍为 12 个请求，超过服务最大序列数 10；控制端等真实 snapshot 中短 decode 请求达到 14/16 token、至少两条等待请求可修改时发动作，不用客户端提交顺序推断槽位。

回滚（备份路径取本次安装结果）：

```bash
bash run_dabs_r234.sh native-rollback --replica 0 --backup /实际/dabs_native_backup_时间
sudo docker restart wushibo-qwen-small-npu0
```

## 可选的状态破坏专项

普通测试和实验不自动执行以下项目，未运行明确记为 `not_run`。只有需要测试重启失效时才使用：

```bash
bash run_native.sh install --verify-restart
```

该选项在部署后额外对副本 0 保存旧快照、重启、检查旧身份动作被拒绝。之后重新执行普通 test，使标定对应新引擎。

只有需要清空副本 0 前缀缓存并验证失效时才使用：

```bash
bash run_native.sh test --invalidate-cache
```

这会在排空后清空 prefix cache，检查命中转为未命中及旧证据动作失效。普通 test 中的实时缓存重查、快照有效期、实际资源、重复动作及队列安全约束始终有效。

单独保留的入口：`native-probe-four`、`native-calibrate`、`native-invalidate`、`native-restart-arm`、`native-restart-check`、`native-accept`，均通过 `bash run_dabs_r234.sh 命令` 调用。

## 失败怎么处理

查看本次 `SUMMARY.txt` 和失败步骤日志。缺接口会列出 endpoint/op、HTTP 状态或具体字段；补丁结构不适用会列出实际缺少的类/函数。不要改 PASS 标志，不要重新跑旧 prepare。

如果标定提示 `require >=20 ... same host step call`，说明没有足够的同口径批次；保留 `native_profile.json` 的 failures/validation 和少量 `host_calls` 原文。当前统计目标是实际 EngineCore Host-step，不能拿异步 forward 返回时间或 HTTP 时延替代。

源码与文件指纹变化只记溯源。正式实验仍冻结系数、工作负载和有效资源；重启或资源/模型变化需要新验收，不能拼接旧 HTTP 实验。每次运行独立写普通文件，旧目录与失败数据保留，不打包、不上传。
