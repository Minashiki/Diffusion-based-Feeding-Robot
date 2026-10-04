# 单豆 M3：环境、事件与验收契约

物理模型沿用 `single_bean_native_v1`，任务版本为 `single_bean_m3_v1`。只编译 `bean_000`，固定位置／姿态生成后自然沉降。Panda、UR5e 共用餐具、Bean 和物理参数；M4 教师、正式采集及 DP/RL 未放行。正式状态由匹配当前输入的双机器人报告和冻结清单决定，文档本身不提前宣称通过。

## 环境与状态接口

`FeedingTask(robot_id, task_mode=True)` 在沉降完成、时钟归零后初始化 SELECT。`FeedingGymEnv` 和 `FeedingRobot-v0` 使用同一 IK／内置位置伺服／物理推进入口。

- action 为 float32 六维 `Box(-1,1)`，前三维按机器人线速度、后三维按角速度缩放，均在机器人基座系表达；适配器继续执行范数限速、加速度及参考误差保护。
- 默认 reset 是 `beans_in_bowl`；`beans_on_spoon`、`empty` 只作诊断。旧 `food_on_plate/food_on_spoon` 拒绝加载，不提供别名。
- step 最多推进 20 ms；物理 dt 必须整除该周期。任务或故障在子步终止，`elapsed_s` 报告实际时间。外部上限保持 60 s，成功／失败优先于同边界超时。
- `terminated` 表示成功或明确失败，`truncated` 表示外部超时；结束后必须 reset。SB3 自动 reset 后的终态放在 `terminal_observation`。
- 非有限状态返回最后有限观测并置 `observation_valid=False`；这不是异常时刻的有效状态。

观察 schema 2，字段顺序保持 Panda 96／UR5e 94 维（`82+2*n`）：q、dq、TCP 位置／旋转矩阵／世界系 twist、`bean_relative_world`、嘴部相对位置／旋转／开口、三组 F/T、八阶段 one-hot、四项当前交互状态、执行状态 one-hot、frame_age。单豆位置字段在 StateProvider 为 `(1,3)`，Gym 中展开为 3 维。实际 Bean 位姿／速度保留在 snapshot；Bean ID、接触明细、判据及事件位于 oracle，不向 actor 提供未来事件、场景种子或材料参数。

## 几何、阶段与奖励

Bean 为原生椭球。任意坐标系下的半径投影使用 `sqrt(sum((半轴 * 旋转投影)^2))`，不使用包围盒角点替代椭球边界。勺头承载要求实际接触的净世界向上力和 TCP +Z 载荷超过原 `1e-5 N` 门槛、质心在勺头横向界限内且向 −TCP Z 射线命中真实勺头网格。柄部、纯侧向力和悬空不能形成承载。

阶段顺序保持 SELECT／ACQUIRE／TRANSPORT／WAIT_READY／APPROACH／TRANSFER／RETRACT／RECOVER：

| 阶段／事件 | 必需条件 |
| --- | --- |
| SELECT → ACQUIRE | 当前 Bean 目标有效 |
| pickup → TRANSPORT | 整颗 Bean 高于实际碗沿、无碗载荷、真实勺头承载；世界系线速度 <0.001 m/s、角速度 <0.1 rad/s，连续 0.5 s |
| TRANSPORT → WAIT_READY | 到嘴前 6 cm 等待位，位置误差 ≤1 cm、姿态误差 ≤0.1 rad |
| WAIT_READY → APPROACH | 等待位、当前开口和完整勺头／Bean 包络净空连续满足 0.1 s |
| APPROACH → TRANSFER | 餐具进入嘴部交互区；准备条件失效则进入 RECOVER |
| delivery → RETRACT | Bean 完整包络在当前接收区域、有实际嘴部支撑且脱离勺子，连续 0.2 s |
| success | 所有餐具碰撞几何退出、解除嘴接触、Bean 继续留存，连续 0.1 s |
| RECOVER → WAIT_READY | 实际动作返回等待位且退出嘴部接触；阶段机不产生恢复轨迹 |

Bean 接触穿透超过 0.4 mm 立即失败；其余模型保持 3 mm／5 ms 深穿透判据。接触／腕力保护保持 5 N／8 N。无支撑宽限保持 0.1 s；已获取后回落碗内、落到桌面／地面、交付后丢失及未释放就撤离分别记录明确失败。连续窗口中断归零，同子步失败优先，候选里程碑仍记录但不奖励。

奖励保持 pickup +10、delivery +20、success +50、首次 failure −50，另含实际耗时、受保护组 applied 接触冲量及同阶段距离进展。每个里程碑至多奖励一次；阶段切换／终止不产生距离跳变奖励。边界载荷补充峰值与保护，不重复积分冲量。

## 驱动、快照与验收

M3 证明驱动复用 M1 的 65° 前沿入豆和沿有限碗壁回平／抬升路径。运行中只发送统一 twist，不写 Bean 或机器人 qpos/qvel。首次前沿接触必须在 Bean 下半部且力向上，推进阶段 Bean 高度上升；完整轨迹保留实际接触、载荷和穿透。M3 独立驱动在抬升末端减速，5° 入勺落点保持不变，回平结束误差收紧为 0.2 mm／0.002 rad，收尾角速度限制为 0.3 rad/s。路径结束后至少保持 1 s，再在原 pickup 判据成立后运输，并记录运输开始时豆子实际线／角速度；不修改共用 M1 路径。取起后从已越过碗沿的抬升位直接前往嘴前等待位，连续携带，入口后先下降到下颌附近，再在 2.5 s 内将目标旋转向量从零渐变到 `[-1.0,0.4,0] rad`，让 Bean 从前侧勺缘释放。释放目标嘴坐标 XY 为 `[0.008,-0.008] m`，Z 按当前下颌接收平面和完整勺头顶点计算，释放动作的目标勺头保持 1 mm 平面净空（入口准备判据中的 2 mm 包络余量保持不变）。先完成固定的倾倒姿态与目标位置，并由原连续 0.2 s 判据确认交付，再将目标后移 3 mm、侧移 −3 mm、抬高 1 mm，到位后撤离；即使交付事件较早触发，也不会跳过这些动作位置，从而避免按不同的中间倾倒姿态开始撤离。当前嘴部实际线／角速度参与跟踪，不读取未来驱动；嘴部阶段先限制相对动作速度，再叠加嘴部速度，避免跟踪项被动作限幅削弱，底层机器人限速不变。撤离时保留最后释放目标姿态，避免尚在口内时回平碰到上沿。

M3 task snapshot schema 4、event rules 3、Gym snapshot schema 2，包含完整 MuJoCo integration、边界传感状态、控制参考、有效命令、故障、阶段计时器、历史事件、奖励集合、接触统计及随机状态。机器人、编译模型、配置、版本和回合上限不匹配时拒绝恢复。M1 诊断保持 snapshot schema 3／event rules 2。验收专用跨数值回放只允许改变 timestep、solver iterations 和 tolerance，不放宽公开快照检查。

21 类真实物理用例为 bowl、carry、pickup_lift、bowl_return、receiver、receiver_edge、receiver_outside、unsupported、unsupported_recovered、force、contact_safe、contact_force、penetration、shallow、entry、closed、recover、unreleased、early_withdrawal、post_delivery_loss、handle。pickup_lift 从自然碗中初态舀取；其他阶段预置只在 reset 设置并完整记录，不能替代完整流程。失败用例中的腕部外力和交付后丢失脉冲独立标注。

额外必需 full_static 和 full_dynamic：两机器人、seeds 0/1/2 分别从自然 reset 连续完成 `pickup → delivery → success`，各里程碑奖励一次；种子仅验证固定布局重复性。所有 23 类运行均比较 1 ms／100 iterations／1e-8 基准、0.5 ms，以及 1 ms／200 iterations／1e-10，使用共同初态和 20 ms 命令网格。数值对照 schema 3 分开检查准备、入口和接触确认：

- 舀取准备事件（pickup 与进入 ACQUIRE／TRANSPORT）、pickup_lift 结束允许 1 s，与收尾保持时间一致；此时尚未进入嘴部交互，后续动作起步仍受独立检查及 60 s 总上限约束。
- WAIT_READY／APPROACH／TRANSFER 事件和未触发接触确认的动作开始允许 0.4 s。固定动态嘴部为 0.2 Hz，两轴 10 mm 平移和 5° 偏航在 60 mm 等待距离的速度上界约 24.35 mm/s，0.4 s 内位移上界 9.74 mm，小于原 10 mm 等待位误差；40 mm 接收长度和 0.08 rad 下颌振幅的开口变化上界约 4.02 mm/s，同期变化 1.61 mm，小于原 2 mm 净空余量。
- delivery／success 及其候选、进入 RETRACT、release_clear／retract 起步和成功结束允许 1 s，用于接触沉降与连续确认，低于 5 s 动态周期的四分之一。当前下颌驱动始终保持正开口角（0.1±0.08 rad），平勺入口的最大高度包络低于 26 mm，最小开口约 30 mm，因此不会因这段等待错过入口开口窗口。接触阶段 Bean 已进入接收区；每次运行仍独立验证完整包络、实际支撑、脱勺、保护力、穿透、最终留存和餐具退出，且按当前嘴／接收坐标检查 2 mm 路径精度。时间余量不替代这些物理条件。
- 故障事件／故障结束仍为 10 ms。每次运行仍独立验证入口 0.1 s 连续当前开口／净空及失效恢复。

TCP 空间容差保持 2 mm：先比较完整动作段顺序与实际开始时间，再按各动作段归一化累计路程比较路径（段内静止等待不拉伸路径）；入口（包括转入 TRANSFER 后的入口收尾）和撤离使用各采样时刻的当前嘴部坐标，下降、滚转和脱离接触使用当前下颌接收坐标，舀取和运输使用世界坐标。终态、pickup 及等待／入口位姿事件仍检查 2 mm 位置差；delivery／success 及其候选、进入 RETRACT 的瞬时位置差保留为诊断，因为连续接触确认可以在动作路径的不同进度触发，此时按完整对应动作段路径检查 2 mm 空间差。子步证据保留当前嘴／接收位姿与比较坐标，便于复核坐标变换；20 ms 采样网格必须完整；未经对齐的同一绝对时刻世界坐标差仍记录为诊断值，不作为路径判据。力仍为 0.05 N 或 20%、冲量仍为 0.005 N·s 或 20%，逐接触对检查不变。

`python -m feedingrobot.scripts.validate_m3 --robot all` 顺序执行两机器人，完整入口才可冻结；单机器人或 `--cases` 为局部验收。报告包含 checker、几何、快照、物理矩阵、数值对照、完整流程 viewer／截图、完整 M1 回归及哈希复核。缺项、失败或 viewer 不可用返回非零，不改物理参数状态。

正式证据位于 `outputs/single_bean/v1/m3/<robot>/`：report、state_schema、共同初态 pickle、子步 physics.jsonl.gz、trajectory.csv、事件／奖励／载荷及数值对照。双机器人全部通过且输入一致才发布同目录上层 freeze_manifest／freeze_audit。原 M1 冻结清单与 209 个证据保留；新清单关联其 SHA256，另以本轮源码和文档输入执行完整 M1 回归。不得将原 M1 历史输入哈希当作改动后的完整输入签名。
