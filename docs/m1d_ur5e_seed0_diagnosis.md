# M1-D：UR5e seed 0 / bean_011 顺序排查

日期：2026-10-03。结论：**多豆接触使运动路线分叉，margin 边界的接触开关放大微小状态差异，随后豆—勺—碗局部夹挤产生大力。已完成逐步记录、同状态重放和单豆隔离；四项单变量候选均未通过组合要求，未修改正式参数或轨迹，未进入新的双机器人完整验收或冻结。**

本报告只针对 UR5e `sweep_seed_0` 的 bean_011，不能代替 Panda 或其他 TCP／携带／掉落数值失败的排查，也不宣称已经找到满足全部 M1-D 门槛的修正。M1-D 仍失败，参数保持 candidate，M3/M4 保持 not_verified。

## 1. 逐物理小步记录

入口为 `conda run -n feedingrobot python -m feedingrobot.scripts.diagnose_m1d`。复用 `FeedingTask.step_physics()` 和现有 `sweep_path()/move_pose()`，不新增积分器、不写运行中的 Bean qpos/qvel、不改变保护或验收阈值。正式历史报告保持原样。

从正式 `outputs/beans_native/v1/m1/ur5e/m1d/baseline/sweep_seed_0_states.pkl` 的完整 `episode_reset` 状态开始。三组设置分别为 1 ms / 100 iterations / 1e-8、0.5 ms / 100 / 1e-8、1 ms / 200 / 1e-10。

每小步保存全部 15 豆的位置、四元数、线／角速度、实际接触对象名称、接触位置／距离、世界坐标力向量、接触参数及腕部峰值。区分 applied 接触与推进后的 boundary 接触：`applied_state_time_s = time_s - dt`；比较受力时使用相同边界时刻的 `contacts`，不把两种采样混在一起，也不插值接触力。

原始报告的三组腕部峰值均精确复现。最早显著分歧发生在入豆接触附近，而不是大力峰值时刻：

- 基准 bean_011 首次接触勺子为 **10.468 s**，同时接触碗底及 bean_003、bean_004。
- 半步长与基准在 **10.466 s** 已有约 **0.0377 N** 接触力幅值差，bean_011 位置差约 **0.489 µm**，TCP 位置差约 **9.03 µm**；到 10.467 s 豆位置差超过 1 µm。因此不能仅凭“豆质心接近”认定几何状态相同。
- 基准最大夹挤发生于 **12.675 s**：bean_011 同时接触勺头 prism_086／095、碗底和 wall_01／02。豆—勺接触力幅值合计约 **2.486 N**，全部接触力幅值合计约 **5.190 N**，腕部峰值约 **2.484 N**。这些幅值之和不是合力。

记录保存在 [诊断目录](../outputs/calibration/beans_native/m1d/ur5e_seed0_diagnosis/)，主报告为 [report.json](../outputs/calibration/beans_native/m1d/ur5e_seed0_diagnosis/report.json)，逐步记录为 `baseline_substeps.jsonl`、`dt05ms_substeps.jsonl` 和 `iterations200_substeps.jsonl`。

## 2. 从同一状态重放

基准运行在 **10.40 s** 和 **12.40 s** 保存完整 schema 3 快照，以及逐 1 ms 的实际 actuator ctrl 序列。局部重放保留 robot、spoon、bowl、物理初态和驱动时钟，仅替换三组数值设置。为排除闭环反馈带来的命令差异，在现有 adapter update 位置重放记录的 actuator ctrl；半步长在同一 1 ms 区间内使用相同 ctrl。物理仍只由 `step_physics()` 推进。这是局部物理诊断，不是完整闭环数值验收。

基准设置从两个检查点重放时，豆位置、TCP 位置和接触力均精确复现原记录；10.40 s 的重放持续 3 s、覆盖夹挤峰值。不同设置的完整积分初态均检查为相同。

| 腕部峰值 N | 基准 | 半步长 | 200 iterations |
| --- | ---: | ---: | ---: |
| 原始完整历史运行 | 2.48448 | 0.13751 | 1.66008 |
| 同一 12.40 s 状态，15 豆重放 | 2.48448 | 2.41032 | 2.46308 |
| 同一 12.40 s 状态，单豆重放 | 2.47532 | 2.42700 | 2.46158 |
| 同一 10.40 s 状态，15 豆重放 3 s | 2.48448 | 0.15132 | 1.66073 |
| 同一 10.40 s 状态，单豆重放 3 s | 0.01033 | 0.01036 | 0.01033 |

原始半步长的低峰值不能解释为“对同一晚期夹挤状态算出了较小力”：给它同一 12.40 s 状态后，大力仍出现。分歧前重放和晚期重放需分别解释，不能互相替代。

完整状态和驱动序列见 `checkpoints_and_drivers.pkl`；延长的入豆重放见 [extended_entry_replay.json](../outputs/calibration/beans_native/m1d/ur5e_seed0_diagnosis/extended_entry_replay.json)。

## 3. 单豆隔离及相同几何状态求解

单豆实验只在检查点初态关闭其他 14 颗 Bean 的碰撞掩码，保留 bean_011 的质量、惯量、几何、位置、四元数和速度；保留原勺子、固定碗、机器人和相同驱动序列。没有移动或固定 bean_011，运行中不改 qpos/qvel；每步核验接触中没有其他 Bean。

结果支持两个不同阶段：其他豆参与了“进入夹挤”的路线选择；夹挤状态一旦形成，单豆局部接触就足以维持大力。入豆时直接接触 bean_011 的邻豆为 bean_003 和 bean_004，但本轮没有进一步逐邻豆删除，不能把因果责任单独归给其中一颗。

进一步重建从同一 12.40 s 状态出发、在 **12.416 s** 到达的两组状态，分别对每个状态用全部三组设置重新 `mj_forward`，不推进时间、不改变位置或速度：

| 12.416 s 状态来源 | 用基准求解 N | 用半步长求解 N | 用 200 iterations 求解 N |
| --- | ---: | ---: | ---: |
| 基准路线的状态 | 0.06157251 | 0.06157251 | 0.06156623 |
| 200 iterations 路线的状态 | 0 | 0 | 0 |

这里的力为 bean_011 各接触力幅值之和。两状态质心只差约 **50 nm**、姿态只差约 **14.7 µrad**，但 `mj_geomDistance` 核验的 bean_011—wall_02 距离分别为 **49.99669 µm** 和 **50.00193 µm**，跨过 **50 µm** 接触 margin；实际 boundary 接触列表也显示后一状态已无 wall_02 接触。相同几何状态换求解设置只产生微小差异，换成另一状态则接触和受力一起消失。重新求解的迭代数为 7–9，未达到 100/200 上限。

这些直接证据支持接触激活边界和路线历史放大差异，未证明 MuJoCo 对同一状态的力计算存在明显错误，也没有排除其他时刻或其他用例的数值问题。详见 [contact_activation_boundary.json](../outputs/calibration/beans_native/m1d/ur5e_seed0_diagnosis/contact_activation_boundary.json) 和 `same_state_forward.json`。

![轨迹、同状态重放与单豆对照](../outputs/calibration/beans_native/m1d/ur5e_seed0_diagnosis/diagnosis.png)

## 4. 单变量候选及淘汰证据

各候选都从原方案独立开始，没有把修改叠加，没有写入正式 config／XML，也没有放宽门槛。

| 独立修改 | 检查结果 | 判定 |
| --- | --- | --- |
| 仅 sweep 终点回退 5 mm，壁侧 gap 3 → 8 mm | UR5e seed 0 三组均没有真实拾取；仍有 1.65／0.181／1.33 N 腕部峰值 | 淘汰 |
| 仅入豆起点横向 y 12 → 6 mm | seed 0 三组均拾取，但半步长豆—勺峰值差 0.1088 N 超过 0.0561 N；200 iterations 豆—勺冲量差 0.1758 Ns 超过 0.0481 Ns。基准 seed 1 在 12.862 s 触发 contact_limit，seed 2 无真实拾取 | 淘汰 |
| 仅 Bean solimp[0] 0.99 → 0.95 | UR5e seed 0 穿透约 0.4357 mm，5 s 沉降失败 | 淘汰，未继续舀取 |
| 仅 Bean solimp[0] 0.99 → 0.98 | 基准单豆／双豆／勺头与 seeds 0–9 沉降通过；半步长 seed 6 的 bean_004／012 穿透约 2.0409 mm，虽有低速承托窗口仍失败。基准 seed 0 舀取通过，腕部峰值 0.0681 N；半步长豆—勺峰值差 0.0879 N 超过 0.05 N | 淘汰，不能以舀取改善抵消沉降失败 |

证据分别位于 [sweep_gap_8mm](../outputs/calibration/beans_native/m1d/sweep_gap_8mm/summary.json)、[entry_y6mm](../outputs/calibration/beans_native/m1d/entry_y6mm/summary.json)、[其余 seeds](../outputs/calibration/beans_native/m1d/entry_y6mm/other_seeds/summary.json)、[solimp 0.95](../outputs/calibration/beans_native/m1d/solimp_dmin_095/report.json)、[solimp 0.98 沉降数值对照](../outputs/calibration/beans_native/m1d/solimp_dmin_098/numerical_contact_settling.json) 和 [0.98 舀取](../outputs/calibration/beans_native/m1d/solimp_dmin_098/motion/summary.json)。

## 5. 验证与后续门槛

本轮 86 项相关回归通过：`test_m1d_native.py`、`test_m1c_native.py`、`test_beans_native_runtime.py`、`test_guard_physics.py`；诊断入口语法检查通过。另由实际重放核验基准轨迹和力的精确复现、数值变体完整积分初态一致、单豆无其他豆接触。各试验的源输入在运行前后哈希一致；随后新增的排查文档不是新的正式验收结果。

探针源代码、绘图程序和回归日志保存在诊断目录的 `probes/` 和 `regression_tests.txt`。正式历史 M1-D 报告和冻结保护均未改动，没有生成冻结清单。

下一项修正必须同时做到：在所有 seeds／数值设置下避免多豆把豆子送入勺—碗夹挤、保留真实拾取、穿透不超过 0.4 mm、保持完整沉降窗口，并满足原力／冲量和 TCP 对照。已有证据不支持通过增加迭代数、简单放软接触或仅扩大一个动作阶段的间隙直接放行。局部组合要求通过后，才进入 Panda／UR5e 完整 M1-D 复验；本轮没有满足这一前置条件。

随后实施的抬高、分离动作及侧移试验见 [动作试验报告](m1d_transition_followup.md)。侧移改善了腕部峰值并保留真实取豆，但语义接触峰值／冲量对照仍失败，未采用任何候选；本报告的历史记录保留。
