# 服务时间标定与目标平台版本登记

## 为什么还要标定

LLM权重不变，设备的prefill、decode、mixed服务时间仍可能变化。原DAG通过相位模型比较完整workflow完成时间，因此把GPU39系数直接称为目标卡已校准模型，会混淆算法效果与模型迁移误差。

`profiles/reference/model.json`只有原推理系数（没有旧拟合误差、JCT表或run日志）。`priors.json`只保留已完成校准输出的hop级p75；它们是算法输入，不是本包附送的实验结果。该reference可用于CPU回归或明确标注的“冻结源模型迁移”敏感性对照，但目标主结果应先独立标定。

## 原实现所在位置

- `core18v2/service.py:features`：相位服务特征和系数读取，6维非负模型。
- `joint18v2/service_fit.py:read_rows/nnls/fit_phase_model`：真实native批次与host-step观测，拟合及整块留出。
- `source/spra_stage19_typed_handoff_v1/tools/run_typed.py:freeze_profile`：原语义profile全过程参考。
- `source/spra_stage19_musique_joint_v1/tools/experiment.py`和`tools/run.py`：原基础profiling参考。

原`freeze_profile`会引用旧目录、要求特定B<block>条件命名，还可能回退到旧系数；**不能直接在目标卡使用GPU39的legacy curve作为所有相位的隐式fallback。**适配时由目标卡真实calibration生成新曲线，保留pre/decode/mixed覆盖；缺少相位数据就补校准，不用旧值或零值冒充。

用32个`profile.public.jsonl`问题或原固定计算负载先采集目标服务数据，不用64题主实验中“D赢得最多”的条件拟合。预算扫描若用于离线标定，不是主方法的adaptive batching；所有正式A/D/RR/P2C条件仍固定4096。

## 复制源代码的显式登记

目标卡需要修改native接口后，原`typed19/dependency.py`会阻止偷偷改变依赖，这是正确行为。完成接口代码改动后，在**交接包的复制目录内**执行：

```bash
python -B tools/target_profile.py record-sources \
  --device-label target_card_v1 \
  --reason '只适配目标卡native接口、设备身份与协议；算法/应用不改' \
  --out target_registration/source_v1
```

该命令验证13个关键算法/应用文件未变化，保存原dependency记录与逐文件delta，只更新此复制目录的dependency identity，使适配后的接口可被导入测试。它不会修改GPU39项目，也不代表gate已经通过。如果关键算法变化，工具会拒绝将其登记为“只换硬件”。

完成真实目标标定，得到兼容格式的`model.json`、`priors.json`和独立验证材料后：

```bash
python -B tools/target_profile.py seal-profile \
  --device-label target_card_v1 \
  --reason '使用目标卡独立校准数据；后续正式套件冻结' \
  --model /absolute/path/to/target_calibration/model.json \
  --priors /absolute/path/to/target_calibration/priors.json \
  --evidence /absolute/path/to/target_calibration/validation.json \
  --out profiles/target_card
```

该工具记录模型/先验/代码/证据哈希，不替代人工和原生验证，不根据JCT收益判定校准有效。它检查四个逻辑服务的三个相位系数均存在且有限非负。host曲线文件与真实校准证据应另归档，不可根据主实验结果挑模型。

随后使用`port.py gate/run --profile profiles/target_card`。若硬件接口还未正确适配，native gate仍会明确失败；不能伪造status、ready标记或把本工具的登记回执当成服务就绪证明。

## 数据一致性与输出变化

跨卡保持tokenizer、模板、采样与dtype，仍应记录实际输出量及EM/F1。不同卡的浮点细节可能造成回答变化；不得强制使用旧答案、删除错误问题，或把新的质量差异自动解释成调度收益。原模型可见UUID隔离限制仍存在，任何改动应作为独立协议记录。
