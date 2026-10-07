# M5 纠偏示范与 50k EMA 实验分支

本交接处理已发现的 ACQUIRE 姿态残差：模型能去噪，但在“仍有 3–10° 残差、实际角速度很低”的条件下没有持续给出有效纠偏动作。原训练集的早期 ACQUIRE 覆盖检查没有找到该组合。这个覆盖结论只针对已检查的窗口，不能证明它是全部闭环失败的唯一原因。

新的示范版本独立于正式 M4。原始数据、归一化、源码、配置与 `run_001` 保持原版本。这里提供实验数据与训练交接，不启动训练，不使用 test，不冻结 DP_v1，不进入 M6。

## 数据入库规则

`correction_data.py` 只用独立训练种子 740001–740006，目标为 4 个回合：3° 与 5° 请求各 2 个，最多 6 次尝试。先按原 teacher 抵达 ABOVE 点，再通过真实控制执行低速角脉冲并停稳；不修改机器人位姿、关节状态或观测。停稳后继续原 teacher，执行到完整成功或既有 60 秒时限。

脉冲及停稳动作保存在回放记录中，但动作标签无效；训练只使用 teacher 拥有的有效动作。入库同时要求原纠偏门槛通过、完整流程成功、物理逐步回放通过。失败、超时与未达标尝试保留在 `attempts/`，不进入训练数据目录。新的 `train` 种子及 group 必须与原数据的全部 split 不重叠。

数据报告：`outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json`。仅当 `status == "ready_experimental"`、4 个 `accepted` 的 `replay.status == "passed"` 且 `source_unchanged == true` 时可使用。脚本检查文件覆盖与 SHA256、原 M4 父证据、原 50k 检查点和归一化；报告不能代替正式 M4 冻结证据。

## 承接方式

这是新的实验分支：模型与 EMA 从 `step_50000.pt` 的 **EMA 权重** 初始化，优化器与随机状态重新初始化。它不是恢复旧 `run_001` 的优化器训练；旧 `best.pt` 仍是 10k，不能用于本交接。

保留原架构、扩散训练目标、归一化、优化器超参数、EMA 衰减和梯度保护。每批 75% 原示范按阶段／回合均衡采样；25% 纠偏窗口按新回合／窗口均衡采样，窗口起点限定在真实释放至纠偏完成的 ACQUIRE 区间。动作窗口仍按原规则在无效标签与阶段边界截断。25% 是待验证实验配比，不是已证明的最优值。

为避开尚未在宿主验证的 DataLoader 多进程问题，本工具直接生成批次，实际 workers=0，保留原 CPU 预算。默认 GPU 使用原 BF16 autocast 支持规则；不允许 CUDA 不可用时静默改用 CPU。

## 零更新审计

在仓库根目录执行；每次使用新的输出目录：

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
BASE=outputs/single_bean/v1/m5/dit/run_001/step_50000.pt
DATA_REPORT=outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json
"$PYTHON" tools/m5/train_correction.py audit --checkpoint "$BASE" \
  --data-report "$DATA_REPORT" --device cpu \
  --output outputs/single_bean/v1/m5/dit/correction_warmstart_audit_002
```

审计不创建优化器、不反向传播、不更新权重，只核验数据、EMA 精确承接和一个 4 样本混合批次的有限前向 loss。检查 `report.json`：`status=audit_passed`、`ema_exact=true`、`optimizer_updates=0`、`optimizer_created=false`、`source_unchanged=true`。

## 宿主 GPU 训练

只有在审阅新数据与零更新审计后，由你指定累计目标。这里不推荐一个追加预算。`--updates` 是**累计目标步数**，不是新增步数：从 50k 开始，目标 T 对应新分支更新 T−50000 次；恢复时必须大于检查点当前 step。

以下命令提示你输入目标；输出目录必须尚不存在：

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
BASE=outputs/single_bean/v1/m5/dit/run_001/step_50000.pt
DATA_REPORT=outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json
read -r -p '输入大于 50000 的累计目标步数：' TARGET_UPDATES
"$PYTHON" tools/m5/train_correction.py train --checkpoint "$BASE" \
  --data-report "$DATA_REPORT" --device cuda --updates "$TARGET_UPDATES" \
  --output outputs/single_bean/v1/m5/dit/correction_run_001
```

按原频率计算原 validation split 的 EMA 去噪 loss并保存检查点；不自动调用旧 trailing 闭环评估。保存 `step_T.pt`、`last.pt` 和 `metrics.jsonl`，不按去噪 loss 宣称任务能力或生成 `best.pt`。

恢复必须使用本工具、同一个实验目录及完全相同的基检查点、数据报告、原源码、诊断 Python 文件和归一化：

```bash
read -r -p '输入大于 last.pt 当前 step 的累计目标：' TARGET_UPDATES
"$PYTHON" tools/m5/train_correction.py train --checkpoint "$BASE" \
  --data-report "$DATA_REPORT" --device cuda --updates "$TARGET_UPDATES" \
  --output outputs/single_bean/v1/m5/dit/correction_run_001 \
  --resume outputs/single_bean/v1/m5/dit/correction_run_001/last.pt
```

实验检查点使用独立 schema、`diagnostic=true`、明确数据与工具绑定；原正式训练／评估入口应拒绝它。工具 Python 文件变化会阻止原分支恢复；保留 `tool_snapshot`，需要改版时另建分支并重新审计，不能覆盖旧证据。README／测试修改不影响训练 Python 版本绑定。

## 先看真实拾豆，再完整 validation

训练结束后先用新 EMA、leading＋10 步、噪声种子 0、原保护与执行节拍，检查排序最前的 3 个正常 validation 场景：

```bash
"$PYTHON" tools/m5/eval_correction.py pickup --base-checkpoint "$BASE" \
  --checkpoint outputs/single_bean/v1/m5/dit/correction_run_001/last.pt \
  --data-report "$DATA_REPORT" --device cuda \
  --output outputs/single_bean/v1/m5/dit/correction_pickup_001
```

检查每场 JSON 的真实 `pickup` 事件、阶段进入、失败原因、接触／腕力峰值和预测限幅比例，以及预测／下发／参考／实测速率。先确认模型能给出纠偏和真实拾豆；3 场成功不代表 M5 通过。如果出现非有限值或仍持续限幅，停止扩大物理验证并重新排查采样与条件。

拾豆改善后，执行正常／恢复各 15 场的完整 validation：

```bash
"$PYTHON" tools/m5/eval_correction.py validation --base-checkpoint "$BASE" \
  --checkpoint outputs/single_bean/v1/m5/dit/correction_run_001/last.pt \
  --data-report "$DATA_REPORT" --device cuda \
  --output outputs/single_bean/v1/m5/dit/correction_validation_001
```

结果至少要求 `full.normal` 与 `full.recovery` 各 15 场、完整成功各至少 12/15；所有阶段成功率至少 90%；15 个恢复场景全部真实进入 RECOVER，不能把未进入恢复阶段视为恢复成功。检查报告场景清单及完成场景均为原 validation，所有完整失败原因仍要审阅。

如果尺度恢复、姿态纠偏或 pickup 仍失败，先比较新失败状态与纠偏示范覆盖、训练／推理条件和 mask 行为，再决定补数据或继续训练；不能只因 loss 下降继续增加预算。新示范数量小，只证明记录流程及局部纠偏可行，不能保证训练后闭环成功。

完整 validation 达标后，另行实现正式评估入口的可审计推理修订支持和新数据版本审批。随后才运行正式 test；test 正常／恢复完整成功各至少 12/15、各阶段至少 90%、全部恢复场景真实进入 RECOVER，通过后才允许冻结。

## 本轮实测交付

`correction_dataset_v1_001` 达到 `ready_experimental`，4 次尝试全部入库，没有使用备用种子或放宽门槛：

| 新训练种子 | 实际残差 | 完整成功时间 | 有效 teacher 标签 | 物理／观测回放最大误差 |
| --- | ---: | ---: | ---: | --- |
| 740001 | 3.4946° | 58.346 秒 | 1129 | 0／0 |
| 740002 | 5.5087° | 59.372 秒 | 1136 | 0／0 |
| 740003 | 3.4946° | 58.713 秒 | 1137 | 0／0 |
| 740004 | 5.5087° | 59.323 秒 | 1135 | 0／0 |

共 4537 条有效动作标签；其余 180 条因脉冲、停稳、阶段中断或末尾不完整动作而无效。训练混合器从其中提取 74 个释放至纠偏完成区间的 ACQUIRE 窗口，原训练集保留 113136 个合法窗口。最高接触／腕力峰值为 0.023603／0.023396 N。

实际 CPU FP32 零更新审计位于 `outputs/single_bean/v1/m5/dit/correction_warmstart_audit_001/report.json`：EMA 与正式 50k 权重逐张量完全相同，优化器未创建，更新次数为 0，混合 4 样本前向 loss 为 0.008565913，原 344 项源码哈希与 M4 父证据通过。这个前向值仅验证加载及计算有限，不能作为任务成功证据。

相关测试 90 项通过，1 项 DataLoader 多进程测试留待宿主验证。额外的小模型测试验证连续更新与中断恢复的模型、EMA、随机状态完全一致；没有对实际 80M 模型训练。本环境没有可用 CUDA，宿主 GPU 训练及训练后 DP 闭环仍待执行。四个成功回合来自 teacher，不能视为 DP 已成功拾豆。
