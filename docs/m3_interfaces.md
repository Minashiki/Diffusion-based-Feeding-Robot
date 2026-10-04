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

M3 证明驱动复用 M1 的 65° 前沿入豆和沿有限碗壁回平／抬升路径。运行中只发送统一 twist，不写 Bean 或机器人 qpos/qvel。首次前沿接触必须在 Bean 下半部且力向上，推进阶段 Bean 高度上升；完整轨迹保留实际接触、载荷和穿透。取起后连续携带，入口后受限侧向滚转（−1.2 rad）释放，目标嘴坐标 `[0.014,-0.008,-0.004] m`；当前嘴部实际线／角速度参与跟踪，不读取未来驱动；撤离时保留释放姿态，避免尚在口内时回平碰到上沿。

M3 task snapshot schema 4、event rules 3、Gym snapshot schema 2，包含完整 MuJoCo integration、边界传感状态、控制参考、有效命令、故障、阶段计时器、历史事件、奖励集合、接触统计及随机状态。机器人、编译模型、配置、版本和回合上限不匹配时拒绝恢复。M1 诊断保持 snapshot schema 3／event rules 2。验收专用跨数值回放只允许改变 timestep、solver iterations 和 tolerance，不放宽公开快照检查。

21 类真实物理用例为 bowl、carry、pickup_lift、bowl_return、receiver、receiver_edge、receiver_outside、unsupported、unsupported_recovered、force、contact_safe、contact_force、penetration、shallow、entry、closed、recover、unreleased、early_withdrawal、post_delivery_loss、handle。pickup_lift 从自然碗中初态舀取；其他阶段预置只在 reset 设置并完整记录，不能替代完整流程。失败用例中的腕部外力和交付后丢失脉冲独立标注。

额外必需 full_static 和 full_dynamic：两机器人、seeds 0/1/2 分别从自然 reset 连续完成 `pickup → delivery → success`，各里程碑奖励一次；种子仅验证固定布局重复性。所有 23 类运行均比较 1 ms／100 iterations／1e-8 基准、0.5 ms，以及 1 ms／200 iterations／1e-10，使用共同初态和 20 ms 命令网格。现有容差保持事件时间 10 ms、TCP 2 mm、力 0.05 N 或 20%、冲量 0.005 N·s 或 20%。

`python -m feedingrobot.scripts.validate_m3 --robot all` 顺序执行两机器人，完整入口才可冻结；单机器人或 `--cases` 为局部验收。报告包含 checker、几何、快照、物理矩阵、数值对照、完整流程 viewer／截图、完整 M1 回归及哈希复核。缺项、失败或 viewer 不可用返回非零，不改物理参数状态。

正式证据位于 `outputs/single_bean/v1/m3/<robot>/`：report、state_schema、共同初态 pickle、子步 physics.jsonl.gz、trajectory.csv、事件／奖励／载荷及数值对照。双机器人全部通过且输入一致才发布同目录上层 freeze_manifest／freeze_audit。原 M1 冻结清单与 209 个证据保留；新清单关联其 SHA256，另以本轮源码和文档输入执行完整 M1 回归。不得将原 M1 历史输入哈希当作改动后的完整输入签名。
