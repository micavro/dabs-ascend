# 跨卡复现实验方案（无历史结果）

## 固定项

LLM为Qwen2.5-1.5B-Instruct，使用相同权重/tokenizer/chat模板。原应用为MuSiQue已知annotation DAG，真实父响应作为后继输入；不是用gold答案喂后继。共64个历史问题，四个互不重叠的16题Block。profile另用32题，不用测试胜负拟合服务时间。

执行配置：固定prefill cap和total 4096；max_model_len4096；max_num_seqs64；block_size16；原200ms共同执行循环；原JSON answer-only自然结束，max_output_tokens96；不能强行padding或使用未来真实输出长度。

到达率是系统总workflow/s。wave为1秒活跃、1秒空闲，活跃期以2λ到达，N=16截断。短cohort完成率不代表稳态最大吞吐容量。

## 移植后建议顺序

| 套件 | 条件 | 方法/重复 | 正式条件数 |
|---|---|---|---:|
| `port-smoke` | R4、λ2、B3 | RR/P2C/Affinity/DAG各1次；仅集成检查 | 4 |
| `port-cases` | R4λ2B3、R4λ2B2、R3λ2B4 | 四方法×8轮，固定案例 | 96 |
| `port-load` | R4、B3，λ=1/2/4/8 | 四方法×8轮 | 128 |
| `port-scale` | λ2、B3，R=1/2/3/4 | 四方法×8轮 | 128 |
| `port-grid` | R=1..4，λ=1/2/4/8，B1..B4 | 四方法×2镜像轮 | 512 |
| `port-ablation` | R4、λ2，B1..B4 | Affinity/DAG/DAG-Lite，三方法全6排列 | 72 |

重点案例中，前两个来自已有完整消融中值得优先复现的Block；第三个为已有三副本首轮的探索性候选，证据范围弱于前两个。这里不存历史性能值，也不声称它们是尚未回传条件或目标卡的全局最好案例。

先做CPU/协议检查、目标标定、port-smoke、port-cases；再按需要扩展负载与副本曲线。port-load、port-scale与port-cases有重合工况，但属于新的预声明实验，不把同一条件的最好一次成绩跨套件拼接。port-grid为完整覆盖补充，不要求初次移植把所有套件一口气跑完。

## 顺序与随机量

四方法8轮采用四方法Williams顺序重复两个周期。每个固定案例中，每种方法在每个位置出现2次，每种有序相邻方法对出现2次（仅统计一轮内部）。port-grid仅两镜像轮，不声称完整位置/前序平衡。

每个Block的到达种子为`190500 + block`；packet_slice为`[(block-1)*16, block*16]`；移植套件P2C种子为`940000 + block + 100*repeat`。所有条件都已展开保存在`source/spra_workflow_accept_v1/plans/port-*.json`，不依赖历史结果文件生成计划。

## 报告与验收

主指标是同一R、λ、Block、repeat配对的平均workflow JCT下降，再对该固定Block的预定重复平均。四个Block可以分别报告；不在每次执行结束后更换Block并把最大值当作固定应用均值。

每个被选案例对RR、P2C、Affinity使用同一份DAG执行，所有指标和质量来自同一条件。保留负收益、异常、缺失配对、输出token变化和未完成计划，不能以是否达到某个收益值筛选有效条件。

记录平均/P95 workflow JCT、请求TTFT、客户端TPOT代理、cohort完成率、节点关键路径等待、控制开销、缓存/压力/回退/路由计数、EM/F1、实际输出量。P95是各条件内部的P95，未合并原始样本时不能称为总体P95。GPU token ITL不可用时保持缺失，不用host观测冒充。

原UUID是模型可见的缓存隔离手段，A和D未必有相同prompt tokens；这是保留的源实验限制。直接去掉UUID而不清理缓存，会使后跑方法额外受益，不能作为默认适配。
