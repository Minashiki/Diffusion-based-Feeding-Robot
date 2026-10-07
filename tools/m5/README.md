# M5：50k EMA 采样诊断与私下验证

固定检查点为 `outputs/single_bean/v1/m5/dit/run_001/step_50000.pt`，SHA256 为
`82f215dd0d1dbcadc9676e14b141851bb5670d51dabb100e45106e1ba240bff5`。
`best.pt` 仍是 10k、闭环分数为 0，不能替代这个候选模型。

当前工具只做独立诊断，不启动优化器、不访问 test 场景、不发布冻结清单。M5 仍 incomplete，DP_v1 未冻结，不能进入 M6。

## 1. 工具、版本与指标

在仓库根目录运行命令，使用已有 feedingrobot 环境，不需要安装依赖。
`tools/m5/` 不进入现有 `input_hashes()` 的 `tools/*.py` 集合，但每次运行都会保存本目录文件快照及 SHA256。
工具先执行未修改的 M4 父证据和原检查点校验，再临时修改 DDIM spacing；训练 DDPM、epsilon、cosine、归一化、条件编码和物理保护保持原版本。
新目录拒绝覆盖；`provenance.json` 保存完整采样配置、timestep、精度、硬件和原始源码哈希；`report.json` 绑定输出证据。
源码校验失败时应查明版本差异，不能关闭校验或改写旧检查点的 source_hashes。

离线对照选择正常 validation 各回合各合法阶段的首窗口、恢复回合首个合法 RECOVER 窗口，共约 105 个窗口，覆盖 30 个回合。
使用种子 0/1/2、batch=8；每个 batch 为两种 spacing 重置同一噪声种子，并保存每个窗口的初始噪声哈希。
这是固定代表性窗口诊断，不是全部 16,937 个 validation 窗口的误差。
推理动作 mask 全有效；标签 mask 仅用于合法前 4 步和完整合法窗口的统计。
线／角 RMSE 定义为三维向量误差平方和的平均值再开平方，单位分别为 m/s、rad/s。
速度使用三维范数；报告 p50/p90/p99/max、超出 0.05 m/s 或 0.5 rad/s 的比例，并按种子、阶段、正常／恢复分组。
`samples.npz` 保留预测、标签、mask 和条件，可复核统计。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest tools/m5/test_diagnose_sampling.py -q
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest tests/test_m5.py -q -k 'not spawn_workers_read_identical_mmap_windows'
```

现有 DataLoader 多进程测试在受限会话中未结束后已中断；请在宿主环境另跑：

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest tests/test_m5.py::test_spawn_workers_read_identical_mmap_windows -q
```

## 2. 本轮 CPU FP32 对照与 3 个完整正常场景

下面保留本轮实际执行配置；这些输出目录已存在，再次运行必须更换编号，并让 `--offline-report` 指向对应的新离线报告。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/diagnose_sampling.py \
  --mode offline --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --output outputs/single_bean/v1/m5/dit/sampler_comparison_50000_001

/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/diagnose_sampling.py \
  --mode pickup --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --offline-report outputs/single_bean/v1/m5/dit/sampler_comparison_50000_001/report.json \
  --output outputs/single_bean/v1/m5/dit/pickup_leading_50000_002
```

pickup 前核验离线报告与证据文件；离线有非有限采样会跳过物理。还会做 leading 有限值预检查；闭环非法命令则停止后续场景。
只运行排序最前的 3 个正常场景，leading＋10 步、噪声种子 0、headless；每个场景执行至终止或原有 60 秒上限，不截断为动作前缀。
逐回合 JSON 保存 pickup/delivery 真实事件、阶段、失败原因、峰值、冲量及预测／下发／参考整形／实测速度；`conditions.json` 保存每次重规划输入。
历史 trailing 基线来自 `run_001/validation_50000`，使用 CUDA/autocast；它与 CPU FP32 没有精度配对，不能将闭环差异全部归因于 spacing。
pickup 的总体验收 status 仍会 failed，因为只有 3 个正常场景；退出码 1 也可能表示完整执行但任务未达标。查看 report、pickup_successes 和失败原因。

若 leading 的线／角 p50 均不超过速度限值，且两类限幅比例均比 trailing 降低，但仍为 pickup 0/3，自动生成 `followup.json`：

- 取每个 validation 回合各阶段首／末短窗口（含 ACQUIRE），做 leading 的同噪声全有效／标签 mask 对照。标签 mask 含未来示范阶段长度，**仅限离线**，不注入物理推理。
- 在同一记录观测上重建推理的因果 32 条观测缓冲，检查 features 数值与有效 mask 一致性。
- 对进入的阶段，取每个 train／validation 回合首／中／末合法窗口，比较失败策略条件到同阶段 train 条件的最近 RMS 距离。

覆盖检查是抽样、未校准的诊断：较大的距离提示值得检查，不能直接证明示范缺失或宣布需要补数据。

## 3. 宿主 GPU 完整 validation 的交接命令

先查看 CPU 对照和 3 个完整场景。只有动作尺度改善并出现真实 pickup，才考虑完整 validation；短时动作合理或 loss 下降不是放行依据。
宿主 GPU 建议先重跑离线对照，以检查实际 CUDA/autocast 精度；工具不会在 CUDA 不可用时静默转 CPU。
各输出目录必须不存在；重复运行请递增末尾编号。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/diagnose_sampling.py \
  --mode offline --device cuda \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --output outputs/single_bean/v1/m5/dit/sampler_comparison_50000_cuda_001

/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/diagnose_sampling.py \
  --mode validation --device cuda \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --offline-report outputs/single_bean/v1/m5/dit/sampler_comparison_50000_cuda_001/report.json \
  --output outputs/single_bean/v1/m5/dit/validation_leading_50000_cuda_001
```

核对 mode/scope、检查点哈希、50k/EMA、leading timesteps `[90,80,70,60,50,40,30,20,10,0]`、source_unchanged、精度及全部 30 个逐回合文件。
正常／恢复各 15 个场景，完整成功各至少 12/15；pickup、delivery、retract、recovery 各至少 90%，阶段分母为实际进入数量，所有 15 个恢复场景必须真实进入 RECOVER 并检查 RECOVER→WAIT_READY。
即使此诊断 validation 的 status=passed，也不会冻结。下一步才是为正式评估入口实现可审计推理修订，绑定原训练版本、实际推理代码／采样配置及验证证据，然后使用 test。
正式 test 同样满足上述门槛才允许发布 DP_v1 并进入 M6；不能直接拿当前 eval_dp 或 freeze 命令当作 leading 评估，它们仍使用原 trailing。

## 4. 如何决定补数据或续训

本轮不追加训练预算。首先区分采样是否恢复合理尺度、pickup 是否真实改善、失败是否集中于短窗口／条件偏移／示范覆盖。
尺度仍不合理时继续查采样；尺度改善但 pickup 失败时先看 followup 和逐回合轨迹，再决定 mask／条件实现修订或针对性补数据。
mask 或条件代码若要改变，必须先设计新版本及旧权重承接证据；不要修改原 run_001 或伪造源码哈希来恢复训练。

未来确认需要续训时，`--updates` 表示累计目标：从 50k 恢复到 60k 是新增 10k，并非新增 60k。
现有 train_dp 要求恢复到原运行目录，配置、数据、归一化和源码均须匹配。仅做工具诊断不会改变原训练版本；原训练内部的 validation 采样也仍是 trailing。
因此不能将“可以恢复旧训练”理解成“已修复正式训练内的评估采样”，也不建议现在按旧配置直接追加更新。

## 本轮实测结果

已完成 CPU FP32 的 105 窗口 × 3 种子 × 2 spacing 对照，覆盖正常／恢复各 15 个回合。所有样本有限，原始 344 项源码输入哈希保持一致，优化器更新为 0。

| 合法前 4 步、三种子合并 | trailing＋10 步 | leading＋10 步 |
| --- | ---: | ---: |
| 线速度中位数（m/s） | 0.49650 | 0.02043 |
| 角速度中位数（rad/s） | 2.85550 | 0.07267 |
| 线速度限幅比例 | 94.68% | 0% |
| 角速度限幅比例 | 95.39% | 0% |
| 线动作向量 RMSE（m/s） | 0.69869 | 0.01075 |
| 角动作向量 RMSE（rad/s） | 6.60529 | 0.10794 |

这是跨阶段的代表窗口统计，与旧报告“正常初始条件”或“全部闭环动作”口径不同。
完整合法 horizon 的 leading 线速度限幅比例仍为 0.507%，角速度为 0%；不能把前 4 步的 0% 扩展为整个预测 horizon 无超限。

leading 的 3 个完整正常场景全部 pickup 失败，均停在 ACQUIRE，因 contact_limit 在 12.897／12.359／13.117 秒终止。
766 条闭环预测的线／角速度中位数为 0.004513 m/s／0.001956 rad/s，限幅比例均为 0%；可见尺度修正没有转化为真实拾豆。
当前结果不支持进入完整 30 场景 validation、直接追加训练、test 或冻结。第 3 节命令保留为达到前置条件后的交接流程。

追加诊断结果：

- 同观测下的因果缓冲特征重建，105 个窗口最大数值差为 0，有效 mask 一致；这不排除闭环观测漂移。
- 196 个阶段首／末短窗口的 mask 补充对照包含 30 个 ACQUIRE 窗口。ACQUIRE 合法前 4 步线 RMSE 由 0.000708 降至 0.000511 m/s，角 RMSE 由 0.007406 降至 0.006766 rad/s。全部阶段合并时角 RMSE反而由 0.03291 升至 0.04423 rad/s，没有一致改善，更不能作为闭环修复证据。
- 抽取 300 个 ACQUIRE train 窗口、45 个 validation 窗口，对比 193 次失败策略重规划条件。策略最近距离 p50／p95 为 0.652／1.322，validation 参考距离 p95 为 0.526。该距离未校准且 train 覆盖只是抽样，只能提示检查状态漂移，不能据此证明数据不足或决定补数据。

证据目录：

- `outputs/single_bean/v1/m5/dit/sampler_comparison_50000_001/report.json`：完整离线对照及原始样本。
- `outputs/single_bean/v1/m5/dit/pickup_leading_50000_002/report.json`：3 个完整正常场景；同目录逐回合 JSON、conditions.json 和 followup.json。
- `outputs/single_bean/v1/m5/dit/mask_followup_50000_001/report.json`：覆盖 ACQUIRE 等全部合法阶段的补充 mask 对照；同目录 run.py、工具快照和哈希使其可追溯，没有新增物理回合。

`pickup_leading_50000_001` 是首次记录字段冲突后的失败运行，已保留 error 报告和首场轨迹；正式诊断结论使用 `_002`。
首场在修正记录字段后重跑，物理结果和预测统计一致。初版 followup 仅含首条件的 WAIT_READY／RECOVER 短窗口，补充报告扩展了覆盖；当前工具自动 followup 已采用扩展规则。
报告中的工具快照保留运行时源码、测试和指南版本，当前指南包含后续实测总结，因此指南哈希与旧快照不同。

## ACQUIRE 密集检查与执行长度对照

`acquire_diagnosis.py` 继续固定 50k EMA、leading＋10 步及原物理保护，只读 validation。
`dense` 使用上述 3 个失败场景，前 15 秒每 200ms 的合法示范窗口，以及已保存的策略重规划条件；种子为 0、1、2，统计合法前 4 步。
示范条件上的动作才有真实标签；策略条件只与同时间教师动作作参考比较，不能称为这些状态上的动作误差。
位姿同时报告同时间教师差值和前 15 秒最近位置轨迹差值，最近位置不保证对应的旋转或任务进度相同。

`cadence` 对排序最前的 3 个正常 validation 场景逐个执行 4 步／1 步配对，每回合保持 H=16，分别每 200ms／50ms 重规划，执行至正常终止或原 60 秒时限。
两组在相同规划 tick 使用相同初始噪声，种子为 `tick//50`（base seed=0）。因此本轮 4 步组也是重新运行，旧 `_002` 的连续 RNG 轨迹仅作历史参考。
记录每次规划的输入、噪声种子、预测动作、原始观测、下发／参考／实测速度、事件、失败接触对和推理耗时。没有将教师动作、未来 mask 或时钟输入模型。
`prefix_rollout.py` 是独立的原评估循环副本，仅增加本次对照的执行长度、噪声配对和记录；测试核对 4 步连续 RNG 行为与原循环一致。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/acquire_diagnosis.py \
  --mode dense --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --previous-report outputs/single_bean/v1/m5/dit/pickup_leading_50000_002/report.json \
  --output outputs/single_bean/v1/m5/dit/acquire_dense_50000_002

/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/acquire_diagnosis.py \
  --mode cadence --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --previous-report outputs/single_bean/v1/m5/dit/acquire_dense_50000_002/report.json \
  --output outputs/single_bean/v1/m5/dit/acquire_cadence_50000_002
```

示例使用新目录 `_002`；已完成的首轮结果在 `_001`。宿主 CUDA 复核可将 device 改为 cuda 并使用新的输出目录，报告会记录实际精度；CPU 推理耗时不能代表 GPU 实时性。

密集离线共 418 个条件窗口 × 3 种子，全部有限。前 2 秒示范条件上的角动作沿教师方向幅度中位数为 91.81%，方向余弦为 0.99999；线动作幅度为 91.58%，方向余弦为 0.99978。
三个旧闭环在约 4 秒相对同时间教师已有 7.18°／9.01°／7.65° 旋转差；最近位置路径比较仍有约 7–9° 差异，提示不能只按时间延迟解释。
这些结果支持继续检查早期动作幅度和姿态误差，但尚不能区分模型拟合偏差、10 步采样近似和闭环状态反馈，更不能据此断言需要补数据或续训。

密集证据：`outputs/single_bean/v1/m5/dit/acquire_dense_50000_001/report.json`，含分时间／条件域统计、配对噪声哈希和原始样本。

完整闭环对照结果如下，六次均因勺与碗接触超限终止，未进入交付或撤离。

| 执行长度 | 真实 pickup | 三场终止时间（仿真秒） | 接触峰值最大值（N） | 推理调用总数 | CPU 推理总耗时（秒） |
| --- | ---: | --- | ---: | ---: | ---: |
| 4 步／200ms | 0/3 | 16.872／16.333／17.357 | 11.356 | 254 | 33.925 |
| 1 步／50ms | 0/3 | 13.821／13.772／13.815 | 18.955 | 827 | 110.122 |

所有预测有限；4 步组线／角速度限幅均为 0，1 步组各场线限幅为 0.36%／0.73%／0.36%，角限幅为 0。
1 步组推理调用和推理耗时约为 4 步组的 3.26／3.25 倍；两组运行时长不同，此比值不是固定每秒开销。单次 CPU 推理平均约 133ms，不能据此承诺 50ms 实时闭环。
核对三对场景的首个预测完全一致，共同规划 tick 的噪声种子一致；之后条件随物理轨迹分化。
本轮不支持将正式执行长度改成 1 步。更早失败和更高接触峰值也不能外推为所有场景必然退化。
闭环证据：`outputs/single_bean/v1/m5/dit/acquire_cadence_50000_001/report.json`，同目录保留六份完整逐回合记录和运行时工具快照。

下一项建议是固定相同早期示范条件和初始噪声，对 leading 的采样步数做小规模离线敏感性检查，区分 10 步近似是否影响旋转幅度；这项尚未执行，不建议直接乘以幅度系数或投入训练预算。
若增加采样步数仍保留相同偏差，再检查同一条件的噪声预测误差与闭环姿态反馈；若误差只在策略偏离后扩大，则优先检查漂移状态的纠偏覆盖。
这些分支都需新证据才能决定代码修订、补数据或续训。当前继续保持 M5 未通过，不启动完整 validation、test、冻结或 M6。

本轮验证：新工具 27 项与原 M5 28 项共 55 项通过，原检查点及 344 项输入哈希未变；多进程 DataLoader 测试单独留待宿主环境执行。

## leading 采样步数敏感性检查

`step_sensitivity.py` 使用上述三个正常 validation 场景前 5 秒每 200ms 的 75 个合法 ACQUIRE 示范窗口，固定 50k EMA、H=16、种子 0／1／2 和逐窗口初始噪声，比较 leading 10／20／50 步。
推理 mask 全有效，误差只统计合法前 4 步；不改训练配置、标签或保护阈值。报告给出分种子、分回合及 0–2／2–5 秒统计，并保存预测与配对噪声哈希。
三个设置起始 timestep 为 90／95／98，epsilon 误差放大系数约 7.08／16.02／64.16，步数和起点同时变化，不能把差异仅归因于步数。

继续闭环的诊断门槛在运行前固定：每个种子角动作 RMSE 至少降低 10%，线动作 RMSE 增幅不超过 5%，线／角限幅比例均不恶化，且所有预测有限。
候选有多个时选择平均角 RMSE 最小的一组。仅用原 4 步执行／200ms 重规划、tick 配对噪声，先跑一个完整场景；真实 pickup 后才扩展到其余两个。这不是 M5 验收或正式选模。
若无候选，自动进行已知前向噪声的 t=20／50／90 epsilon 误差及重建检查；真实噪声、教师动作只用于离线已知噪声探针，不输入条件编码器，也不执行这些重建动作。
该探针衡量示范条件下的去噪误差，不能直接确定闭环失败原因或证明需要续训。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/step_sensitivity.py \
  --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --previous-report outputs/single_bean/v1/m5/dit/acquire_dense_50000_001/report.json \
  --output outputs/single_bean/v1/m5/dit/leading_steps_50000_002
```

首轮证据目录为 `outputs/single_bean/v1/m5/dit/leading_steps_50000_001`；命令使用新的 `_002` 防止覆盖。

首轮 CPU FP32 实测（75 窗口 × 3 种子，合法前 4 步合并）：

| leading 步数 | 线动作 RMSE（m/s） | 角动作 RMSE（rad/s） | 线限幅比例 | 角限幅比例 | 前 2 秒角动作沿教师方向幅度中位数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.010264 | 0.049913 | 0.56% | 0% | 91.81% |
| 20 | 0.008588 | 0.040440 | 9.78% | 6.67% | 96.36% |
| 50 | 0.008108 | 0.038606 | 26.44% | 17.44% | 99.57% |

20／50 步每个种子的角 RMSE 均降低至少 10%，线 RMSE 也降低，但限幅比例均增加，未通过预定诊断门槛，因此本轮没有新增物理场景。
全部采样和已知噪声探针输出有限，原 344 项输入哈希及 50k 检查点不变。当前共 63 项测试通过，多进程 DataLoader 测试仍留待宿主验证。

不能仅用限幅比例把这些设置等同于旧 trailing 的严重尺度爆炸：66% 的教师线动作、43.67% 的角动作速度已达到上限的 99% 以上，接近边界的预测也会计入限幅。
20 步预测最大线／角速度为 0.05510 m/s／0.52376 rad/s，50 步为 0.06444 m/s／0.59679 rad/s。
超过限速 5% 的线／角动作比例，20 步为 0.89%／0%，50 步为 4.00%／6.44%。教师速度略高于阈值的约 1e-8 数值误差不应解释为示范违反保护约束。
单纯速度投影后的角 RMSE 为 10／20／50 步 0.049913／0.040285／0.035841 rad/s；这只是离线速度投影，不包含加速度整形或物理执行，也不能证明 pickup。
后续若调整闭环候选门槛，需要明确考虑绝对超限幅度和保护后动作误差；不应事后更改本轮门槛并宣称已通过。

已知前向噪声探针在 t=20／50／90 的归一化 epsilon 分量 RMSE 为 0.14612／0.09344／0.04022，而角动作重建 RMSE 为 0.00812／0.01263／0.04186 rad/s。
这说明较小的 epsilon 误差不保证较小的动作重建误差，并支持继续关注采样起点和误差放大；不证明采样步数是唯一根因，也不支持直接追加训练。
完整证据见该目录 `report.json`、`samples.npz` 和 `epsilon_probe.npz`，包含实际 timestep、种子、精度、硬件及运行时工具快照。

## 20 步受保护闭环探索

`review_twenty_steps.py` 在采样步数对照之后单独评审速度投影后的误差，再探索 20 步；上一轮“限幅比例不得恶化”的门槛和失败结论保持原样。
三个噪声种子中，20 步线／角投影后 RMSE 均优于 10 步，且超限幅度与推理成本小于 50 步。这是观察结果后选择的探索性配置，不是独立验证或正式选模证据。
评审使用原 `clip_norm`，只代表下发前的速度投影；真实闭环仍执行原加速度整形、参考位姿保护、接触／腕力保护及 60 秒时限。
固定 leading 20 步、50k EMA、4 步执行／200ms 重规划、validation 正常场景、tick 配对种子 0。同一规划 tick 的初始噪声与已有 10 步／4 步参考一致；随后条件会随轨迹分化。
先跑排序第一的完整场景；无真实 pickup 或出现无效命令／非有限状态即停止扩展，真实 pickup 后才继续其余两个场景。未访问 test，也不启动训练或冻结。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/review_twenty_steps.py \
  --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --offline-report outputs/single_bean/v1/m5/dit/leading_steps_50000_001/report.json \
  --baseline-report outputs/single_bean/v1/m5/dit/acquire_cadence_50000_001/report.json \
  --output outputs/single_bean/v1/m5/dit/pickup_leading20_50000_002
```

首轮输出在 `pickup_leading20_50000_001`，示例使用新目录 `_002`。报告同时绑定离线对照、原 10 步参考及其 SHA256，逐回合记录保留原始观测、动作、事件、位姿参考差、失败接触对及推理耗时。

首轮 CPU FP32 单场结果：真实 pickup 0/1，15.487 秒因 contact_limit 终止，未进入交付或撤离；按规则停止扩展，未运行其余两个场景。
接触／腕力峰值为 10.618／8.165 N，失败接触对为 bowl–spoon。309 条预测全部有限，线／角限幅比例均为 2.59%；最大预测速度为 0.05130 m/s／0.51118 rad/s，实际下发仍限制在 0.05 m/s／0.5 rad/s。
20 步共有 78 次推理、推理总耗时 21.004 秒，CPU 单次平均约 269ms，不应视为已满足 200ms 实时重规划要求。

同场景、同 tick 噪声的 10 步参考与 20 步对照：

| 观察项 | 10 步／4 步执行 | 20 步／4 步执行 |
| --- | ---: | ---: |
| 真实 pickup | 否 | 否 |
| 终止仿真时间（秒） | 16.872 | 15.487 |
| 2.05 秒同时间教师旋转差 | 4.42° | 1.88° |
| 4.05 秒同时间教师旋转差 | 7.49° | 5.05° |
| 12.05 秒同时间教师旋转差 | 8.03° | 5.46° |

20 步减小了早期旋转差，但剩余姿态误差在后续持续存在，仍未转化为真实 pickup；不能把单场的更早失败解释为统计意义上的退化。
这些位姿差是教师轨迹参考，不是偏离示范状态下已知的最优目标；下一项应关注约 2–4 秒的旋转误差为何未被纠正，以及该阶段策略条件的纠偏覆盖，而非继续仅凭幅度或 loss 改善增加训练预算。
目前不采用 20 步为正式推理版本，不继续扩大 validation，不启动 test、训练、冻结或 M6。新工具和原 M5 共 66 项测试通过；多进程 DataLoader 测试仍单独留待宿主验证。

## 2–4 秒旋转纠偏与早期示范覆盖

`rotation_feedback.py` 只做离线检查，不新增物理回合。绑定上述失败单场、原 50k EMA 和 leading 20 步，分析 2–5 秒实际记录，并对 15 个重规划条件及对应教师条件使用配对种子 0／1／2 重采样。
已知早期旋转目标由原 acquisition 配置及每个回合的 teacher 参数重建，只用于解释和计算局部教师角控制律参考，不输入条件编码器或执行端。
局部角控制律逐项复核：失败场景 99 个早期教师标签最大差 9.28e-8 rad/s，全部早期训练标签最大差 1.01e-7 rad/s，均小于 1e-6。
这使参考不仅是同时间教师轨迹；它明确度量同一早期目标对当前姿态的角纠偏需求，但仍不是经物理验证的完整偏离状态专家动作。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/rotation_feedback.py \
  --device cpu \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --previous-report outputs/single_bean/v1/m5/dit/pickup_leading20_50000_001/report.json \
  --output outputs/single_bean/v1/m5/dit/rotation_feedback_50000_002
```

首轮结果在 `rotation_feedback_50000_001/report.json`，预测原始数组在 `samples.npz`；示例使用新的 `_002`。
30 个教师／策略条件窗口 × 3 种子全部有限。约 3–4 秒，失败闭环相对早期目标的旋转误差中位数为 5.03°；局部教师角纠偏参考速度中位数 0.2636 rad/s，策略实际下发中位数只有 0.00300 rad/s，实测为 0.00153 rad/s。
对应的三个种子离线重采样角速度中位数分别为 0.00283／0.00176／0.00246 rad/s，沿参考纠偏方向的投影比例中位数均为负（-0.00154／-0.00372／-0.00880）。
原记录中的整形指令与弱下发指令一致；纠偏不足已经出现在策略指令，不能仅用执行器追踪问题解释。整形／实测值是 50ms 区间末端，姿态及参考是区间起点，不应视为同时刻精确追踪误差。

覆盖查询明确限定为早期 ACQUIRE：枚举 100 个 train 回合、tick<=5000 的全部 10,000 个合法窗口；定义“残差慢转”为目标旋转误差 3–10° 且实测角速度<0.05 rad/s。
训练命中数为 0，当前失败轨迹 15 个被检索重规划条件中有 12 个命中。这个描述符是针对观察到的失败提出的探索性查询，不是校准后的 OOD 阈值；结果也不能外推为整个数据集完全没有纠偏示范。
作为对照，15 个正常 validation 回合的 375 个早期窗口到上述 train 条件的完整归一化特征 RMS 最近距离 p50／p95 为 0.02576／0.09532。
失败轨迹 3.05／4.05 秒的最近距离只有 0.08134／0.07472，但最近训练窗口的旋转误差仅 0.105°／0.00465°，实际角速度也已很低。完整特征平均距离会掩盖特定的“残留旋转误差＋慢转”差别，不能仅靠该距离判定纠偏覆盖充分。

当前证据支持的下一步是小规模验证和补充 **ACQUIRE 内部旋转纠偏示范**，而非按原数据直接追加训练。它不是任务的 RECOVER 阶段数据，不能用来替代正式恢复场景验收。

后续纠偏数据验证应满足：

- 使用独立 calibration／train 种子及新输出目录，保留原 M4、50k 检查点、validation 和 test；不得把这个失败 validation 回合直接加入训练。
- 在早期安全高度用真实仿真执行产生约 3／5／8° 的姿态残差和低角速度，再由原教师纠偏；不能只修改观测中的旋转矩阵、伪造 q／dq 或加入未来目标到策略输入。
- 先验证教师是否能在原物理保护下纠正这些状态，再验证完整拾豆；保留接触／腕力、真实事件、失败及拒收记录。几段离线参考动作不能当成有效示范。
- 只有小规模物理验证支持该路径，才形成带父证据的新数据／训练版本，并交给宿主环境训练。现有恢复入口要求原数据、归一化和源码一致，不能把新数据偷偷替换进 run_001 或伪造哈希；需要显式的旧权重承接审计。

本轮没有采集新训练数据、改变策略输入或启动训练；原 344 项输入哈希保持一致，原检查点／M4 证据核验通过。新工具与原 M5 共 70 项测试通过，DataLoader 多进程测试仍单独留待宿主验证。

## 独立 calibration 的教师旋转纠偏实验

`teacher_rotation_calibration.py` 使用新种子 730001／730002／730003，启动前检查不与冻结数据的种子重叠。每个种子使用相同场景做无扰动教师基线和脉冲扰动教师对照。
50k 检查点在本实验中只用于原源码／M4 祖证核验，不执行 DP 推理或训练；所有结果均为教师 calibration 证据，不能作为 M5 策略成功率。

教师在原 `above` 安全点达到位置／姿态容差后，扰动控制保持线速度为零，以 0.05 rad/s 沿早期目标的负 y 轴旋转，通过实际物理控制产生约 3°／5°／8° 残差。
停止阈值考虑原角加速度限制下的短制动弧，并加 0.2° 小余量；记录真实达到的残差，不把请求值当实测值。全过程不直接修改 q／dq、位姿、观测、动力学或保护阈值。
零指令停稳至少 250ms，角速度与整形角速度均低于 0.02 rad/s 后交回原教师。
合格纠偏要求释放时仍在 ACQUIRE，实际残差为 3–10° 且距请求不超过 0.75°；释放后姿态误差<0.01 rad、角速度<0.05 rad/s 持续 200ms。
每回合保留原 60 秒时限、加速度整形、接触和腕力保护。首对基线完整成功、扰动组合格纠偏并发生真实 pickup 后才继续其余两对；任何不满足都保留记录并停止扩展。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/teacher_rotation_calibration.py \
  --checkpoint outputs/single_bean/v1/m5/dit/run_001/step_50000.pt \
  --previous-report outputs/single_bean/v1/m5/dit/rotation_feedback_50000_001/report.json \
  --output outputs/single_bean/v1/m5/dit/teacher_rotation_calibration_003
```

首次 0.1 rad/s 脉冲的记录保留在 `teacher_rotation_calibration_001`：请求 3°，实际 3.99°，教师在 0.85 秒内完成纠偏并完整成功，但超出请求幅度 0.75° 容差，未扩展。未修改该轮门槛或报告。
停稳过程的实测继续转动大于仅按适配器加速度计算的制动弧，后续将脉冲降至 0.05 rad/s，记录在 `teacher_rotation_calibration_002`；命令使用新目录 `_003`。原物理保护与验收容差保持不变。
该实验运行原生 CPU 仿真，不提供训练或 test 模式。
配对检查包含模型签名、几何哈希和初始积分状态哈希；保存完整命令／观测／整形／实测轨迹、真实事件、接触／腕力峰值、实际初始与释放快照。
输出为独立诊断 JSON／pickle，不是 M4 EpisodeWriter 训练数据格式；扰动脉冲和停稳指令不是教师标签，不能把整段轨迹直接加入训练。
若实验成功，下一项才是正式实现新数据版本：只导出合法的教师纠偏标签，保留因果观测、阶段 mask、split 隔离、原归一化承接及父证据，并明确旧 50k 权重的加载审计。当前 train_dp 不支持通过替换旧目录的数据来完成这种版本迁移。

0.05 rad/s 脉冲的三对实测结果：

| calibration 种子 | 实际残差 | 释放角速度（rad/s） | 合格纠偏耗时 | 真实 pickup／delivery | 完整流程 |
| --- | ---: | ---: | ---: | --- | --- |
| 730001 | 3.49° | 0.00465 | 0.80 秒 | 均发生 | 成功，58.677 秒 |
| 730002 | 5.51° | 0.00466 | 0.95 秒 | 均发生 | 成功，59.143 秒 |
| 730003 | 8.53° | 0.00466 | 1.05 秒 | 均发生 | 60 秒 time_limit，尚在 RETRACT |

三组无扰动教师基线均完整成功，初始积分状态／模型／几何配对核验通过；三组扰动均满足实际幅度、慢转残差及持续 200ms 纠偏门槛。
扰动组最高接触／腕力峰值仅为 0.02359／0.02339 N，没有接触超限、受阻或非有限状态。8° 组 pickup／delivery 事件分别在 43.627／58.333 秒，时间上已没有足够撤离余量；保持原 60 秒时限，未延长运行。
完整证据为 `teacher_rotation_calibration_002/report.json` 及六份逐回合 JSON、实际初始／释放快照。早期目标旋转误差字段只适用于本次早期目标，在后续改变目标的阶段不能用于判断执行错误。

结论是：在上方安全点，残留旋转误差与慢转组合可在原保护下由教师纠正，实际 5.51° 这一组还能完整成功，支持以针对性纠偏示范为下一条工作主线。它不证明 DP 已学会纠偏，也不证明只补这一种状态就能解决全部失败。
这些实验在上方安全点形成残差，不完全复现 DP 已开始下探的偏离状态；后续正式数据设计还需覆盖安全范围内不同进度／位置的纠偏，并检查完整闭环。
8° 超时回合只保留为纠偏诊断，不能按当前正常完整成功规则接受为训练回合。若进一步采集，应改进扰动时机／节奏并重新验证时间余量，不应放宽 60 秒限时或把不成功回合标为完整成功。

当前可推进新的纠偏数据版本与旧 50k 权重承接审计，然后交给宿主训练；在这两项完成前，不提供替换旧 M4 目录或直接 resume run_001 的训练命令。
本轮总计 75 项测试通过，DataLoader 多进程测试仍留待宿主环境。原源码、M4 和 50k 检查点未变；没有输出正式训练数据、运行 DP validation／test、训练或冻结。

### 纠偏数据版本与实验训练交接

后续已实现独立 EpisodeWriter 导出：真实脉冲／停稳保留用于回放，但全部标为无效动作标签；仅完整成功、纠偏达标且物理精确回放通过的新训练种子回合进入额外数据版本。保留原 M4 与归一化，实验数据不会替换正式 M4。

操作指南见 [TRAIN_CORRECTION.md](TRAIN_CORRECTION.md)。`correction_data.py` 负责独立采集与数据审计，`train_correction.py audit` 只做 50k EMA 零更新承接核验；宿主训练另用新实验目录、全新优化器／随机状态。实验检查点绑定数据／工具版本，不能按原 `train_dp` 恢复或由原正式评估入口发布。`eval_correction.py` 只支持 leading＋10 步的 validation 拾豆探针与完整 validation。

训练配比是待验证的 75% 原示范与 25% 局部纠偏窗口，训练目标由用户指定；本工具不提供按现配置直接追加预算的结论。新示范只覆盖 ABOVE 安全点局部姿态纠偏，不代表所有失败状态均已覆盖。90 项相关测试通过，原多进程测试仍在宿主待验证；这里不启动实际模型训练。
