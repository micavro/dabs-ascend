# Stage19语义、决策与统计合同

## 数据

源为MuSiQue-Ans v1.0公开train/dev。`#N`引用分解的一基位置，不用全局源question id构造边。只有原生2–4节点、单终点、全节点连通到终点的可执行标注进入样本库。拒绝计数公开。不按答案正确率或serving结果筛选。

运行输入白名单：id/question/paragraphs(idx,title,text)/steps(position,sourceid,question,parents)/graph。评分labels分别保存，条件完成后才打开。服务模型没有答案长度标签；长度先验来自train真实生成。

每问题共享真实document/question前缀。2700-token证据预算对所有文档等额分配前缀，标题也计入；不填充到固定token长度。正文截取均不读support标签。超过后端上下文的动态父输出不悄悄裁剪。

## 时间

静态DAG编译和共同文档tokenization在该批arrival开始前，全部方法相同，单独报告准备时间。实际父响应解析、动态child tokenization发生在依赖完成后，其时间作为input_prepare计入节点完整延迟/JCT。

known-plan问题不计一个并不存在的LLM planner调用；也不凭空添加root/sink。真正ready是max(arrival,parent.finish)。实际发现、准备、提交、生效、到期单独记录。已有动作后果在期限结束后继续保留。

## 搜索

S0为完整Affinity参照；S1从原生当前ready反链与汇合关系形成部分成组迁移、优先级与有限准入备选。对S1放置后的状态提出B1，B0为冻结fixed向量。四格S0B0/S1B0/S0B1/S1B1用同一200ms窗口、相同续行规则完整评分。

B-only不生成S1，在S0状态提出自身预算。D/Decoupled/Joint使用相同S1；Decoupled先选调度行再选该行budget。Joint先要求四格完整，保留参照并统计distinct actions。执行等价动作的memo复用不会虚增distinct完成次数。

80ms CPU接收预算，12ms预留交接；可选评价截止来自父进程统一deadline，没有deadline前无界前置搜索。每个剩余cell均分当时可用时间，未完成无score；可能回退，必须报告，不能据逻辑正确推断实际有足够在线覆盖。

已有decode保护：预计下轮混合步时与其预计剩余完成均相对参照检查。10%+2ms局部纯decode预算、1ms模型余量、3%+2ms剩余完成边界是预声明工程值，不是物理/统计保证。只提出动作不等于采用动作。所有cap保留，不奖励切换次数。

## 公平性

硬件、hook、DAG执行器、context规则、采样、真实缓存证据、提交与计时接口相同。A不做不需要的搜索。各条件prefix nonce和instance标识只作隔离，不造成固定长度填充；实际token量逐请求记录。

自然EOS使实际输出内容与数量可能不同，所有方法允许相同max96，不用min或ignore_eos。质量差异不能从样本剔除。按train选fixed时EM/F1下降最多2百分点；pilot报告EM/F1及TPOT保护失败但继续完成预声明数据。

## 结果

JCT从workflow到实际终点完成，所有等待计入。TTFT为request_start到首非空内容；完整node首内容起于真正ready。TPOT为每请求(return-first)/(actual_N-1)代理，N<=1/缺usage/仅一可见事件等显式排除而非0，排除数报告。

cohort_duration=max(actual_request_finish)-min(workflow_arrival)。runner退出延后单独报告，不计入cohort分母也不扣JCT。主机token进度不是CUDA token生成时间，P95请求TPOT不是P95ITL。

MuSiQue质量按归一化answer EM/tokenF1/aliases；未实现官方support/group充分性指标。本文样本是known-plan、截取context子集，不声称完整排行榜等价。模型与GPU数据当前尚未在本地跑通真实数据源，必须以服务器实际测量为准。
