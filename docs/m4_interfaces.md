# M4 教师、采集与物理重放接口

新餐具 M1/M3 v2 已正式放行，证据保留在 [验收报告](acceptance.md)。M4 正在重建：几何教师、验收门禁和显示接口已实现，候选参数仍在物理标定，尚未取得稳定真实取餐、完整教师放行或正式数据。没有 DP/SAC 训练。旧教师参数与成功证据不能用于新餐具采集。

## 教师与执行

`Teacher(robot_config, config)`、`Teacher.reset(parameters, *, geometry=...)`、`Teacher.act(policy_obs)`：输入为 `StateProvider.observe()["policy_obs"]` 字典；输出六维基座系 TCP twist，单位 m/s、rad/s。教师维护阶段内路径进度和从当前／过去嘴部位置估计的速度，仅使用当前位姿、交互标记和补偿 F/T。它不接收 oracle、未来驱动计划或随机种子，不修改 MuJoCo 物理状态、任务阶段或结果标记。

原始教师提议和下发命令分开；命令先受教师速度约束，再进入现有 RobotAdapter 的范数限速、变化率限制、目标积分、Mink IK 与 MuJoCo 位置伺服。`check_waypoints(task, waypoints)` 使用独立 Mink Configuration，不写回模拟器。离线可达不证明接触任务成功。

环境 reset 后通过 `teacher_geometry(task)` 提取盘面坐标、盘沿、真实勺唇、承载网格、全餐具包络、食物尺寸及关节范围的只读副本。教师持有该副本，不持有模拟器引用；动态信息仅来自当前／过去的 `policy_obs`。Gym 观测维度不变。

真实勺头为下凹曲面，不能沿用平铲的水平扫取。取餐路径在盘面坐标系按真实网格计算最低点与 TCP 高度，依次转勺、接近、下降、有限弧形舀取、转勺承托和抬升；行程锚定初始食物位置。承托反馈只控制路径推进，真实 `pickup` 仍由 M3 判定。转勺、运输、释放和恢复的参数尚未物理冻结，不把短暂接触视为取餐通过。

M4 的非故障阶段取消使用 `RobotAdapter.stop(hold_reference=True)`：清除旧命令，保留负载下的伺服参考，并按既有加速度限制将参考速度降到零。减速继续经过原有限速、参考偏差、IK、碰撞及物理保护。默认 `stop()` 和故障停止仍清零参考速度并在实际关节位置重新锁定目标；不改变保护阈值或伺服增益。

`FeedingGymEnv.observe_policy()` 返回与原 Gym 一致的观测向量副本。Gym `step()` 的 20 ms 步长、归一化动作、Panda 96／UR5e 94 维 schema 保持兼容。采集器直接复用唯一物理入口 `FeedingTask.step_physics()`，不以交替 2／3 个 Gym 步近似 50 ms 动作周期。

`run_episode(..., viewer=True)` 和 `collect --viewer` 使用独立进程中的被动 MuJoCo viewer，显示模型与积分状态的副本。刷新按墙钟最多 30 FPS，单帧队列满时丢弃显示帧，不暂停或丢弃物理子步、动作或观测。窗口关闭或启动失败只关闭显示，回合继续运行；manifest 的可选 `visualization` 元数据记录请求、状态、原因及帧数。原回合数组、命令、快照与 schema 不变。教师验收和正式采集默认显示，`--headless` 显式关闭；并行验收只显示一个执行环境。显示标注阶段与结果。`benchmark_visualization` 在预热后执行三组等工作量对照，以中位耗时之比判定；超过 1.5 倍时 `apply_training_budget` 关闭显示并保存测量。等工作量和恢复初态由训练入口保证；M5 的当前模型闭环评估及 M6 的单个实际采样环境接入尚未实施。

## 失败后继续仿真的观察入口

在桌面终端进入仓库，使用已保存的固定场景回合及匹配的冻结版本：

```bash
cd /home/minashiki/FeedingRobot_DPRL
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/observe_m4.py \
  --episode outputs/calibration/new_tableware/m4/rebuild_check_01/calibration/fixed_0 \
  --source-root outputs/calibration/new_tableware/m4/rebuild_check_01/frozen_inputs \
  --viewer --diagnostics
```

这是原命令的真实物理重放，同时核对教师历史、事件与结局。工具使用冻结源码和配置；归档网格路径映射到经 SHA256 验证的原资产路径，保持原模型签名检查。原资产缺失或变化会拒绝重放。每次复盘产生独立的 `outputs/inspection/m4/<时间戳>/` 目录。

已验证的复盘摘要位于 `outputs/inspection/m4/20261002T074914_931714Z/summary.json`，教师命令、观测及物理参考误差均为 0；同目录 `diagnostics.jsonl.gz` 保存逐物理步诊断，`diagnostic_commands.jsonl` 保存命令与执行器切换。

此回合仿真 60 秒，在 ACQUIRE 时间截断，无 pickup。墙钟播放可能更久；当前入口没有暂停、逐帧或时间滑块，关闭 viewer 后原命令重放继续。已有诊断试验的 `trace.jsonl` 可逐时刻查看，但这些 trace 不具备完整快照及命令记录，不能作为严格重放入口。

`--post-failure-seconds` 默认 `0`，接受任意有限非负秒数，包括 `2`、`5`、`2.5`。正值须同时指定 `--viewer`。`--case` 必须与 `--cases-file` 一起使用，按该文件的场景及教师参数重新运行当前 Panda 候选；它与保存回合重放模式互斥。`--source-root` 默认当前仓库；复盘此历史回合须显式使用上述冻结目录。

保存回合仍通过原重放器严格核对输入哈希、命令、观测、物理日志、事件和结局。观察入口在每个原命令边界用实时观测重建教师历史，并核对教师命令，重放物理仍执行保存的原命令。新工具位于原输入哈希集合之外，不修改冻结源码、教师、配置或事件判据。

任务失败 `food_dropped`、`food_lost_after_delivery`、`withdrawal_before_release`、`food_missing` 发生时，在故障停止改写命令和承重参考之前捕获完整状态。原回合先正常结束，再由另一个环境实例恢复该状态，继续教师动作；仅观察分支允许任务失败后继续，所有执行保护保持原行为。原回合因执行保护失败、成功或超时时不追加观察。分支到达指定仿真时长、成功、执行保护触发或窗口关闭时结束。窗口不可用时原回合继续无界面运行，并记录跳过续仿真的原因。

物理步长、20 Hz 动作网格、50 Hz 观测网格及阶段取消的承重参考保持原样；持续时间允许一个物理步的舍入差异。原回合不固定暂停；延长段按墙钟约正常速度播放，显示仍使用独立的只读被动 viewer，最多 30 FPS。

证据自动保存到 `outputs/inspection/m4/<UTC时间戳>/`：总 `summary.json` 保留原失败及分支结局；重新运行模式包含原 `episode/`；`branch/` 包含停止前的受信任本地快照、逐物理子步日志、动作与观测日志及分支报告。延长段明确标记为 `observation_branch`，后续成功不改写原失败，不计入验收或训练数据。控制台打印证据目录、原失败时间、实际观察时长和提前结束原因。

可选 `--diagnostics` 为原段和观察分支分别增加同格式的 `diagnostics.jsonl.gz` 与 `diagnostic_commands.jsonl`，不修改原回合数组或命令格式。日志记录教师子段和目标、参考/实际 TCP、速度与差分加速度、食物局部运动、倾角、最低角点高度、支撑与接触、关节余量，以及命令和停止前后的执行器状态。参考 twist 由执行器整形速度转换到世界坐标；差分量只在时间严格前进时有效，首行或零时间增量故障行的差分量为 null。故障停止前后的状态用于识别参考重置，不能把重置后的关节余量当作故障前证据。该日志只用于诊断，不作为教师输入或训练样本。

## 场景与模型版本

当前实现的共享承接判定 `receiver_xy_tolerance_m` 为 0.0005 m：食物全部角点须处于扩展后的水平承接边界内，并具有真实食物—嘴部接触。容差不扩展嘴部净空宽度，也不放宽高度、底面、释放、持续承接或退出条件。该几何判据已随新模型 M3 v2 放行，M4 保持原值。

`env.reset(seed=..., options={"scenario": parameters})` 新增可选场景参数：食物质量、滑动摩擦、盘内偏移；头部原点、有限行程内偏移、幅度、频率、初相位；`recover`。数值范围在 `sim/scenarios.py` 校验，默认 reset 仍使用原 M3 场景。模型质量、惯量、摩擦和头部原点仅在 reset 更改；随后由原有限力驱动推进。

恢复场景将下颌闭合范围扩展至 −0.65 rad，APPROACH 时以原有力矩限制驱动一次 −0.6 rad、1.5 秒闭合，然后恢复正常运动。闭合安排和触发时刻仅在 scenario_state，当前开口与接触才进入 policy_obs。默认下颌范围不变。测试中构造阶段初态只验证驱动接口，不算完整恢复示范。

所有入口现统一使用新勺子与新盘子，工具选择参数及旧变体已移除。工具质量为 0.035 kg；TCP、负载补偿和承载区域已按新模型适配。教师参数尚待 M4 物理标定，不把接口回归通过视为完整教师放行。

M4 配置使用已放行的 M3 默认头部原点 `[0.55, 0.12, 0.35]`。旧 M4 的远端布局不再用于本次教师放行。不同场景模型签名的数据与快照不可混用。资产、模型、配置及源码哈希保存在每个 episode。

任务快照签名升级至 **v2**：在积分状态之外保存边界 `qacc` 和 `sensordata`，使恢复后立即读取的 F/T 与原始边界一致。外层 Gym 快照和观测 schema 仍为 v1。旧任务 v1 快照被拒绝，不把重新计算过的边界传感器冒充原缓存。

## 磁盘格式与时间契约

每回合目录含完整初态 `initial_state.pkl`、`manifest.json`、`commands.json` 和 `.npy` 数组。pickle 只用于本项目生成的受信任本地文件。

- `actions.npy`、`proposals.npy`、`action_observations.npy`：20 Hz 下发 twist、原始提议与动作之前的观测。动作使用 float64 保持命令重放精度；它们不是实测 TCP 速度。
- `action_ticks.npy`、`action_end_ticks.npy`、`action_phases.npy`、`action_mask.npy`：实际执行区间 `[start_tick, end_tick)`；阶段变化或提前结束会取消旧命令并屏蔽被截断区间。
- `observations.npy`、`observation_ticks.npy`、`observation_phases.npy`、`observation_valid.npy`：50 Hz float32 Gym 观测，以及非网格终止边界。数值失败保留最后有效观测并标记无效，不能当新状态用于训练。
- `physics.npy`：每个物理子步的 float64 实际执行参考、整形速度、实测速度、q/dq、三份 F/T、接触／腕力峰值、累计冲量及超限时间。参考姿态保存 wxyz+xyz；六维实测速度为世界系 TCP twist。具体字段顺序在 manifest。
- `commands.json`：精确 tick、下发命令和有效期，以及独立的 stop 操作。非故障阶段 stop 新增可选 `hold_reference: true`，重放同样执行保留参考的减速；未带该字段的 stop 保持原语义。stop 不伪装为正常零动作标签。

预分配 `.npy` 是 mmap 文件；只能使用 manifest 中的 `physics_rows`、`observation_rows` 有效前缀，预留尾部不是训练样本。`load_episode()` 校验各文件 SHA256，返回只读 mmap 数组。一次只处理一个回合，高频日志不汇总到 RAM。

1 ms 下动作网格为每 50 tick、观测每 20 tick；0.5 ms 下分别为 100／40 tick。实际仿真时间与墙钟耗时分别记录。重放重新载入相同模型、场景和完整初态，执行记录命令而不调用教师；逐子步比较参考／物理日志、逐帧比较观测并逐项比较事件与结局。源码／配置／模型不兼容或文件损坏时拒绝重放；只允许重放读取器 `data/replay.py` 自身修复后读取旧记录，结果同时记录旧／新读取器哈希，其他输入仍必须逐项一致。故障可以不推进 tick；同 tick 的网格观测和故障观测保持各自的物理执行顺序。

## 筛选、划分与采集门槛

完整成功回合才作为常规示范；所有失败尝试仍保留。恢复片段必须有真实 `RECOVER → WAIT_READY` 事件且标注与事件一致，失败回合中只有这样的恢复片段可纳入恢复训练。阶段边界的 mask 禁止跨阶段拼接。

布局、食物质量／摩擦、头部运动参数按离散组合生成 group_id，group_id 不含种子；哈希 fold 预先隔离 acceptance／train／validation／test。种子空间也分离，整个回合及其恢复片段只属于一个 split。

常规与恢复分别按 train/validation/test 的 100/15/15 配额采集。每类每 split 最多尝试配额的三倍；不足则保存统计并非零退出。恢复计数是已标注完成片段的回合数，不是截取任意动作的数量。`normalization.json` 只统计 train 的合法动作观测；常量字段使用尺度 1，并保留原始 std。

`collect` 核对 Panda 正常至少 95/100、恢复 10/10，UR5e 正常和恢复各 5/5，双机器人全部物理检查、教师配置及完整输入哈希。每机器人收敛预先固定正常／恢复各 5 个，不得替换失败用例；比较 1 ms／100 次迭代、0.5 ms／100 次迭代、1 ms／200 次迭代且求解容差缩小十倍，保持 M3 的事件、TCP、力和冲量容差。放行状态与整体 M4 验收状态分开，避免要求先有数据才能生成数据。当前教师若未通过门槛，采集命令会拒绝创建正式数据集。
