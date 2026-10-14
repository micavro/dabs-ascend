# Stage19 typed semantic handoff v1

本包结束输入四格探索，提供一个统一应用接口：**原生完整chat tokenization + 原生JSON约束生成 + 有类型的真实父答案传递**。它不是新的Joint算法，不重启或修改后端，不覆盖旧Stage19、数据和结果。

## 本轮证据

`analysis/received_results.json`按最新28条原始答文复算。四格最终EM均为0。旧`unknown_all`为5/0/5/0；按明确的开头UNKNOWN或answer/solution标签UNKNOWN作为**事后拒答信号**，分别为6/3/7/0。这个信号不是新准确率，不改写旧EM/F1。原`format_ok`仅检查answer标签，并非整个answer/evidence格式完整。

源代码没有标签时把整段响应作为答案，因而把解释、引用和拒答标记代入后继实体位置。新接口不做这种回退。不同token数组解码为相同文本本身不证明原tokenization有错；这里改用服务器原生完整模板，消除这项输入差异，而不宣称它必然解释了质量问题。

## 默认环境与依赖

在GPU39放在既有`spra_stage19_musique_joint_v1/`旁边。读取该包的执行器、模型内核和选择器；SHA检查全部继承代码。没有新的pip安装、stage或重启入口。

- 数据默认：`../spra_stage19_musique_joint_v1/dataset/compiled/`
- 实验runtime：`/home/wmy/sci/spra_core_exp/joint_e1`
- 原生hook：`78ca1d716c5701a9f83ba7dc0b22669b6b50a38bd54626a56642130cd1684418`
- 原模型、4副本、4K上下文、hard total4096，合法cap保持256/512/1024/2048/4096。
- 如果继承包位置不同，设置`STAGE19_SOURCE=/实际目录`；不会联网寻找或更换源代码。

## 输出与父子接口

请求使用`/v1/completions`，prompt token数组由同服务器`/tokenize`的messages接口返回，包含`add_generation_prompt=true`。不拼接分别编码的片段。

采用vLLM 0.6.3的`guided_json`字段，schema只有一个字符串字段answer：

```json
{"answer":"..."}
```

证据不足使用`{"answer":"UNKNOWN"}`。不要求生成解释或证据段，不设最小生成长度，不忽略EOS，max_tokens仍为96。

约束解码只约束结构，不验证事实正确性。每个实际输出分为：

- `answer`：结构有效的非UNKNOWN字符串，不代表答案正确；
- `abstain`：合法UNKNOWN，value为null；
- `format_error`：JSON/字段错误、重复字段、截断、多行或标签值等。

真实父答文全部保留。后继只绑定`answer.value`；其他两种状态在问题中显示`UNAVAILABLE_STEP_N`，并附带有类型状态。后继依然按原DAG执行，不跳过请求，不注入gold，不进行自动修复或重复生成。即使发生格式错误，也完成当前整条DAG并报告，避免只保留成功工作流。

总体问题不再重复作为另一个问答目标发送；原始子问题与DAG不改变。文档仍使用gold-blind的2700-token等额前缀预算；不按答案/支持标签挑文档。完整文本重新tokenize可能与分段预算的总token数不同，实际长度记录并执行4096硬检查。

条件UUID仍在第一个缓存块中改变token。共享前缀从实际原生token结果的共同部分取得，动态输入逐次验证。原有原生初始冷缓存检查保留，不清缓存、不回填零计数。

## 现在运行：一个接口验收，7个真实请求

```bash
conda activate spra-exp
cd /home/wmy/sci/spra_paper_experiments_github/spra_stage19_typed_handoff_v1
bash run_checks.sh
python -B tools/run_typed.py gate
bash run_smoke.sh
```

复用原smoke的两题，仍是4+3 hop，A/固定1024，不再执行P0/P1×C0/C1。不做额外模型生成预热。该批是接口验收，不是Joint性能实验。

输出：

- `TYPED_RESPONSE`：每个节点的实际答文、解析状态、父绑定和事后EM/F1；
- `TYPED_SEMANTIC_CONDITION_DONE`：含TTFT、TPOT代理、JCT、实际输出量和语义指标；
- `TYPED_HANDOFF_RESULT`：明确分开transport_pass、answer_contract_pass和语义质量；
- `CLEANUP`：排空与源文件检查。

底层继承执行器仍打印一条较早的`CONDITION_DONE`，其semantic_quality尚为null。分析以之后的`TYPED_SEMANTIC_CONDITION_DONE`为准，不重复计数。

如果原生guided_json不受当前镜像支持或其已安装实现报错，请保留HTTP和容器错误，不安装依赖、不悄悄改回自由生成、不重启整个集群。CPU/mock测试没有实际运行原生约束解码器。

`answer_contract_pass=true`只表示交接格式可用，**不是答案准确率通过**。错误答案如`{"answer":"Armenia"}`仍原样评分，不能因为类型合法而称为语义成功。拒答、无效输出不从统计中删除。TPOT样本不足继续N/A，不制造长输出。

## 回到同版Joint：不再换适配器

接口行为核实后，本包已带同适配器的后续入口，不需要再写另一套方法：

```bash
bash run_profile.sh
bash run_pilot.sh
```

两条不自动串联；不要为了得到更多tokens在运行间修改JSON结构或生成上限。

Profile使用既有32个train问题，4blocks×5固定cap=20条件。准备本接口对应的服务曲线和自然输出长度先验。Pilot使用既有64个dev问题，4blocks×A/D/B/Decoupled/Joint=20条件。D/B不作为收益晋级门槛。

旧Stage18模型仅作支持不足阶段的先验。默认路径：
`../spra_stage18_joint_core_v1/runs/CALIBRATE_20261008T152619Z_edfd9e63/host_model.json`。
可用`--legacy-model /实际文件`指定，不重新跑旧标定。旧XML接口的profile不能作为本接口的冻结profile，代码明确拒绝跨接口复用。

每个profile/pilot批次先做原有固定8个服务预热请求，另加**每副本1个JSON路径预热，共4个**。12个请求分别留档、在正式条件前执行，绝不按时延重复。语法编译和约束解码的剩余在线成本、真实tokenization、准入与传输等待仍计入实测JCT，不做事后扣除。

## 重要范围

- 不声称2题的质量代表整个MuSiQue或模型能力；不保证本接口能答对。
- 改动了提示/输出和输入token构造，不将新旧JCT差值当成Joint收益。
- 约束解码不是零成本：所有方法统一使用，需同版比较。
- 在格式失败/全拒答等退化行为未排除时，不能将短JCT包装成有用任务吞吐。
- 最终主目标仍完整workflow JCT；TTFT、TPOT代理、质量和生成量共同记录。
- 通过本包可执行下一步，但本地没有真实GPU测量或真实问题答文。
