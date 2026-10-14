# 非英伟达后端适配合同

本文件依据包内真实代码写成，不假定某家SDK或某个目标推理引擎已经具备这些能力。

## 保留层与适配层

优先保留`semantic19/selector.py`、`core18v2`、`joint18v2/actions.py/search.py`、typed workflow/answers/quality以及`wfsched/admission.py/routing.py/worker.py`。适配集中在原生hook、设备身份、tokenize/generate协议、状态采集、控制提交、启动路径和新卡服务时间参数。

原gate含`nvidia_visible_devices`、vLLM0.6.3内部签名和四个服务的检查。它们是已知的接口适配点，不是“任何硬件均支持”的实现。不能只填一个假的NVIDIA UUID来让检查通过。用目标设备真实稳定身份替换**身份的语义**，并在`joint18v2/client.py:gate`和`wfsched/runner.py:environment_contract`中一致检查。

## 1. 模型与请求协议

`ENDPOINTS`目前是四个loopback端口8000..8003。`typed19/workflow.py:ServerTokenizer`也有tokenizer URL。路径、服务名可按复制部署更改；必须确保生成与tokenization使用同一模型版本/tokenizer/chat模板。

需要health、models、tokenize、detokenize和SSE completions；检查实际`typed19/transport.py`请求体，支持prompt token ID输入、usage、结束原因和相同JSON约束。不能将prompt先decode再用另一个模板encode后悄悄当作相同输入。任何无法保持的采样/dtype/结构化生成合同都应明示为配置差异。

## 2. Native状态与请求关联

原协议将每个逻辑副本状态写到`<runtime>/joint18v2/spra-vllm-N.status.json`，控制写到`control.json`，前缀查询写到`prefix_query.json`，事件写到每副本`events.jsonl`。

状态须来自真实引擎，包含：服务/pid/boot身份、采样时刻、running/waiting/swapped、请求ID与prompt_hash、实际prompt/computed/output_done/context、分配块、空闲/已用块、最大序列、实际prefix caching与chunked prefill状态、inflight/step完成状态、缓存查询ACK和证书。

没有观测到缓存块不能填“0命中”并声称合同完整。存在未知应进入明确的保守分支；若只能构造历史亲和性而无法采集原生缓存生命周期，应把它标为另一种降级实验，不是同一主方法的精确复现。

## 3. 固定预算与调度动作

主实验所有cap/total均4096，不执行adaptive batching。原生后端继续自己的continuous batching；排序/路由自然造成实际batch变化是允许的。

动作包括合法的路由/就绪顺序/有界准入。`Publisher`给出snapshot_ns、not_before_ns、action_expires_ns、plan_id与transport lease；续接心跳不能延长算法动作有效期。新后端须记录真正应用时间、实际路由、队列次序、fallback原因和case/request对应关系。

不能把HTTP并发限制当作native队列控制，不能绕过资源hold或提高age阈值掩盖压力问题。保留v2.1准入修复及其回归测试。

## 4. 时钟

原client/controller/hook假定同一Linux boot与time namespace的host monotonic时钟。目标卡位于同一机器时仍需验证；跨主机时不能直接相减两个monotonic值。若改为跨主机，必须实现明确的时钟转换或重新定义不相减的测量边界，并将其作为部署差异记录。

`engine_host_ms`为host观察到的step持续时间，不是设备kernel时间。异步加速器必须在正确的完成观测点记录；只测异步launch调用返回可能得到错误服务时间，不能拿来拟合完整step。

## 5. 缓存与批次审计

原`batchcheck.py`及`tail_check.py`会根据真实状态重算batch预算与cache credit。后端需提供prefill/decode逻辑tokens、原生跳过query tokens、上下文长度、输入prefix/cache块、已完成输出进度。若不同后端的缓存/调度内部语义不同，需调整独立检查器并补测试，而不是只把match字段写成相等。

保持样本未完成或无内容时的排除原因。最终指标从真实请求时刻复算，不能减去规划耗时或替换成预测值。

## 6. 部署差异必须记录

填写`configs/target.example.json`，另存实际设备、driver/SDK、服务端框架版本、内存、dtype、权重与tokenizer哈希、服务映射、启动命令和镜像身份。客户端Python包版本不等于服务端版本。相同负载不意味着不同卡的利用率相同，可另作相同归一化压力的扩展实验，但不能替换本次绝对到达率复现。

所有修改只在迁移副本中进行。源代码`tools/restart*.sh`和stage部署脚本是参考，不要在当前GPU39集群执行它们以“安装迁移包”。本包没有主动安装或重启命令。
