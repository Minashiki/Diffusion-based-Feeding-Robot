# 当前单豆喂餐原型范围

单豆 M3 尚未迁移，`task_mode=True` 当前显式拒绝。后续必须从碗中自然初态验证舀取、携带、移到嘴前、释放、撤离、留存及掉落／失败；阶段预置诊断不替代完整流程。

下文接口和成绩属于历史盘子／单块食物版本，不作为单豆放行依据。

# M3 环境、state schema、phase schema 与 reward specification

M3 使用 M1 的同一物理入口和保护阈值。默认 `FeedingTask(robot_id)` 仍是 M1 诊断；`FeedingTask(robot_id, task_mode=True)` 开启 M3 事件，`FeedingGymEnv` 在其上实现奖励与 50 Hz 交互。M2 已取消。这里定义的是 P0 刚体仿真任务，不是人体安全标准或已训练的喂餐策略。

## Gym 接口与时间

```python
import gymnasium as gym
import numpy as np
import feedingrobot  # 注册 FeedingRobot-v0

env = gym.make("FeedingRobot-v0", robot_id="panda")  # 或 ur5e
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
state = env.unwrapped.get_state()
# 在同型号、同模型/配置的环境中恢复：
env.unwrapped.set_state(state)
env.close()
```

- `action_space = Box(-1, 1, (6,), float32)`。前三维乘机器人 `linear_speed_limit`，后三维乘 `angular_speed_limit`，再由原适配器执行范数限速、变化率限制、IK 和保护。不是关节动作，也还不是 SAC residual；相同对角动作经范数限制后不会超出总速度上限。
- `reset(seed=None, options=None)` 默认食物在盘；`options={"preset": "food_on_spoon"}` 是承载诊断初态，不能当作完整取餐成绩。`empty` 仅用于失败诊断。非法 options 报错。
- `step()` 使用仿真时间，每次最多推进 20 ms。默认 1 ms 子步；可用 0.5 ms。物理 dt 必须整除 20 ms，故障或任务结束立即中止子步循环。`info.elapsed_s` 是实际推进时间，故障在动作适配阶段发生时可以为 0。
- `terminated`：完整成功或明确失败；`truncated`：达到外部时间上限，默认 60 秒。上限在首个达到它的物理边界触发；非物理步整数倍的上限最多向上量化一个子步。成功／失败优先于同边界超时。外部上限不作为有限时域任务目标，因此不加入剩余时间观测。
- 结束后再次 `step()` 抛 `RuntimeError`，必须 reset。有限值检查、动作 shape 或范围不合法会清空命令、锁存故障并抛 `ValueError`。
- 数值异常返回最后一份有限观测，并明确 `observation_valid=False`；此观测不是当前异常状态的估计，必须处理终止而不再执行。
- 原生 Gym 不自动 reset。SB3 `DummyVecEnv` 返回的 done 时观测已是下一回合，结束观测在 `info["terminal_observation"]`；`TimeLimit.truncated` 区分超时和任务终止。
- `render_mode="human"` 复用 M1 passive viewer；默认无渲染。`close()` 关闭窗口并等待渲染线程退出。

## State schema v1

Gym 观测是单一一维 float32 Box，Panda 96 维，UR5e 94 维（`82 + 2*n`）。字段按下表顺序拼接；旋转矩阵按行展开，所有返回数组均为副本。真实物理值使用无穷 Box 边界以避免虚假截断，因此 Gym checker 会产生两条建议有限边界的 warning；这不是检查失败。

| 字段 | 长度 | 单位／含义 |
| --- | ---: | --- |
| q、dq | 各 n | rad、rad/s；配置定义的关节顺序 |
| tcp_position | 3 | 世界系 m |
| tcp_rotation | 9 | 工具到世界旋转矩阵 |
| tcp_twist_world | 6 | TCP 原点的世界系线速度 m/s、角速度 rad/s |
| food_relative_world | 3 | 食物质心减 TCP，世界系 m |
| mouth_relative_world | 3 | mouth_entry 减 TCP，世界系 m |
| mouth_rotation | 9 | 嘴入口到世界旋转矩阵 |
| mouth_aperture_m | 1 | 当前上沿与下颌的保守垂直开口 m |
| raw_wrench_sensor | 6 | 原始传感器系 F/T，N、N·m |
| wrench_world_at_tcp | 6 | 外界对工具的世界系 F/T，力矩原点为 TCP |
| compensated_wrench | 6 | 减去刚体工具重力／惯性后的 F/T |
| stage | 8 | 下节阶段顺序的 one-hot |
| interaction | 4 | 当前勺支撑、嘴支撑、餐具—嘴接触、ready |
| execution_status | 17 | 下列执行状态顺序的 one-hot |
| frame_age_s | 1 | 当前真值出口为 0 秒 |

执行状态依次为 `idle, active, expired, stopped, success, blocked, ik_failure, workspace_limit, invalid_command, contact_limit, joint_speed_limit, nonfinite_state, model_penetration, food_dropped, food_lost_after_delivery, withdrawal_before_release, food_missing`。

`env.schema` 和验收目录 `state_schema.json` 提供机器可读顺序、维度、单位及 robot_id。robot_id 不作为字符串混入数值观测；不同关节数量分别使用自己的 schema，不支持 checkpoint 自动跨机器人复用。

`StateProvider.observe()` 仍保留三个出口：`policy_obs` 为当前状态；`oracle_info` 为真实接触、终止原因、阶段计时器及事件；`scenario_state` 为种子和场景驱动容器。未来事件、材料参数、真实接触明细不拼入 Gym actor 输入。当前 mouth_support/ready 由当前接触和几何计算，不读取未来下颌驱动目标。

Gym `info` 包括时间、实际步长、phase、success、failure_reason、observation_valid、本步 events、reward_terms、step_metrics 与 oracle_info。`step_metrics` 给出本步物理子步的接触峰值、腕力峰值、接触冲量和超限时长；oracle 中同名接触统计为回合累计。冲量只使用积分器实际载荷，每子步积一次；边界载荷用于补充峰值／保护，不重复积冲量。

## Phase schema v1 与事件

顺序固定为 `SELECT / ACQUIRE / TRANSPORT / WAIT_READY / APPROACH / TRANSFER / RETRACT / RECOVER`。事件含仿真 `time`、`name`；阶段事件另含 previous/phase，失败事件含 reason。配置集中在 `configs/task.json`。

| 阶段 | 转换条件 |
| --- | --- |
| SELECT | 食物目标有效，下一物理边界进入 ACQUIRE |
| ACQUIRE | 离盘且连续勺支撑 0.1 秒，进入 TRANSPORT，发 pickup |
| TRANSPORT | 到达嘴前等待位且姿态对齐，进入 WAIT_READY |
| WAIT_READY | 仍在等待位，开口及对齐条件连续满足 0.1 秒，进入 APPROACH |
| APPROACH | 条件失效进入 RECOVER；否则餐具进入嘴部交互区域后进入 TRANSFER |
| TRANSFER | 食物脱勺、在嘴内接收区域受支撑连续 0.2 秒，进入 RETRACT，发 delivery |
| RETRACT | 所有餐具几何退出、解除嘴接触，食物继续留在嘴中 0.1 秒，发 success 并终止 |
| RECOVER | 外部动作撤回等待位且退出口部接触后进入 WAIT_READY |

状态机只切换阶段和报告事件，不生成路径或恢复轨迹。阶段变化不清除硬故障，也不绕过既有参考连续性／限速规则。

取餐同时要求全部食物角点在 `plate_frame` 局部法向上的最低点高于有效盘面 2 mm，且无真实盘面载荷。勺支撑要求新勺头实际接触沿 TCP +Z 的净承托载荷大于 `1e-5 N`、质心横向位于实际网格界限内、从质心沿 −TCP Z 的射线命中新勺头碰撞网格。仅有侧向载荷、包围盒空角或勺柄接触不能形成承载；旧 `spoon_support_min/max` 不再决定 M3 pickup，仍用于 M1 几何滑出诊断。食物接触地面／桌面或已获取后回落盘内是掉落；离盘／承载后失去有效支撑超过 0.1 秒也是掉落。该宽限允许从勺到嘴的短暂自由运动。

接收判据要求全部食物角点在嘴内横向和顶部范围内、位于当前下颌支撑面上方容差内，存在食物—嘴载荷且完全没有食物—勺载荷。区域随头部和下颌实时变换。食物在嘴旁、仍压在勺上或空勺到嘴都不满足交付。

等待位是 mouth_entry 局部负 X 方向 6 cm；位置容差 1 cm，姿态容差 0.1 rad。当前开口须容纳当前勺碗／食物投影高度和宽度，留 2 mm 余量。餐具与嘴部交互盒用全部餐具几何的包围盒检测；圆柱用保守包围盒，不会因只检查 TCP 点而提前宣称撤离。

连续判据被打断时对应计时器归零；成功的阶段里程碑保持已达成历史。同子步遇到失败和里程碑候选，保留 `*_candidate` 与所有 failure 事件，但不发成功奖励。主要 failure_reason 优先为执行／物理故障，其次穿透，再食物结果与非法撤离。

失败原因包括 `food_missing`、`food_dropped`、`food_lost_after_delivery`、`withdrawal_before_release`、`blocked`、`ik_failure`、`workspace_limit`、`invalid_command`、`contact_limit`、`joint_speed_limit`、`nonfinite_state`、`model_penetration`。穿透需接触 distance < −3 mm 连续 5 ms；正常压入与深穿透分别验收。释放前撤离只在已进入 TRANSFER 且尚未交付、食物仍与勺接触时判失败。

## Reward specification v1

奖励使用本次动作推进后的物理结果，与 next observation 对齐，各分项在 `info.reward_terms` 输出，总 reward 为其和。

| 分项 | 定义 |
| --- | --- |
| pickup / delivery / success | 首次事件分别 +10 / +20 / +50；每回合至多一次 |
| failure | 首次失败 −50；多原因不叠加 |
| time | −0.01 × 实际推进秒数 |
| contact | −0.1 × 本步受保护接触组冲量（N·s），与 M1 接触监控统计一致 |
| progress | 同阶段前后目标距离差除以 0.1 m，再裁剪到 [−1, 1] |

ACQUIRE 的距离目标为食物；TRANSPORT、RECOVER、RETRACT 为等待位；APPROACH 为嘴入口。SELECT、WAIT_READY、TRANSFER 不发距离进展奖励。一个 Gym 步内发生阶段变化或终止时 progress 为零，防止目标切换产生虚假收益。超时不是任务失败，不发 −50。接触代价与腕部保护独立：外加腕力仍可能触发终止，但不伪造接触冲量。

## 完整快照（任务签名 v2）

`get_state()/set_state()` 与只读诊断 `snapshot()` 分开。快照包含 MuJoCo `mjSTATE_INTEGRATION`（含时间、qpos/qvel、warmstart、ctrl、外力等）、warning 状态、Mink 参考 q 和 TCP 目标、有效命令及时间戳、速度积分、执行故障、阶段与计时器、历史事件、奖励发放集合、接触统计、场景状态、环境步数／终止状态和 RNG。M3 尚无 DP 动作队列，因此不创建占位队列。

恢复核对 schema、事件判据版本、robot_id、MuJoCo 版本、整个编译模型与配置哈希以及回合时间上限，不兼容则拒绝。允许跨相同配置的环境实例恢复；不会调用 reset 来清空控制参考。快照恢复是显式诊断操作，不是教师或策略的物理状态写入通道。

验收目录 `snapshot.pkl` 是本机生成的完整快照，验证从磁盘恢复后的固定动作重放；只用于受信任的本地文件。报告同时保存输入文件哈希、state schema、事件 JSON、轨迹 CSV、测试 XML 与日志。步长对照容差集中于 `configs/acceptance_m3.json`。原始 M0/M1 报告保留，M1 本轮复验输出单独存放。

## v2 定向物理验收与 M4 边界

Panda、UR5e 的新餐具 M3 各 26/26 项正式通过，详见 [验收说明](acceptance.md)。观测顺序、维度、六维基座系 twist、20 ms step、八阶段、奖励、终止／截断和子步峰值／冲量契约保持。

21 类物理用例覆盖盘面沉降、稳定承载、离盘、回落、失去支撑及宽限内恢复、入口净空、开口不足及恢复、嘴内承接边界、到嘴未脱勺、未释放撤离、撤离后留存与丢失、正常／超限接触、浅接触／持续深穿透和勺柄接触。每类固定 seeds 0/1/2，记录预期事件及禁止出现的里程碑奖励。合成状态机测试保留为逻辑回归。

所有预置只在 reset 时完成：盘面、承载、离盘与柄部用例从未获取历史开始；回落、失去支撑与入口用例预置 acquired/TRANSPORT；承接边界及宽限恢复预置 acquired/TRANSFER；未释放与过早撤离预置 acquired/APPROACH。完整配置、reset_q、阶段和预置历史写入证据。之后仅经统一 TCP twist、IK、内置伺服和真实物理步进执行。接收测试预置食物到下颌支撑面，验证实际交付和撤离留存；离盘测试从已预置勺上食物抬升。这些不是从盘舀取或完整喂餐教师成绩。

入口物理测试采用嘴坐标 `[0.004, 0, -0.006] m`；当前代理口腔几何保持。定向用例固定头部，闭口／恢复用例头部初态单独记录；正常环境动态头部由 M1 回归覆盖。reset 重建 Mink Configuration 与 ConfigurationLimit，防止诊断下颌范围变化留下旧 IK 限位缓存。

正常／超限接触使用独立重编译诊断目标：10×10×20 mm，solref `[0.25,1]`、solimp `[0.05,0.995,0.001,0.5,2]`、priority 2。目标位置按 1 ms 基准的已沉降 TCP 冻结。正常接触达到 0.05 N 时在物理边界停止并保留伺服参考；超限用例继续推进直到正常 5 N 保护触发。它不修改正常模型或保护阈值。交付后丢失用例仅在真实 delivery 后施加嘴坐标 `[-0.1,0,0.3] N` 脉冲，最多 0.2 s，不瞬移食物。

每机器人 63 个基准运行，与 0.5 ms／100 iterations 和 1 ms／200 iterations／求解容差÷10 比较，共 126 组。冻结容差为事件时间 10 ms、TCP 2 mm，力为 0.05 N 或 20%，冲量为 0.005 N·s 或 20%。比较完整事件顺序、各事件／终态时间、实际 TCP 路径及事件位置、全部接触组峰值和冲量。额外记录 food—mouth／plate 载荷不改变受保护组接触代价；边界载荷只补充峰值，不重复积分。

正式证据在 `outputs/new_tableware/v2/m3/<robot>/`：报告及输入 SHA256、state_schema、事件、每物理子步的 `physics.jsonl.gz`（接触 ID、位置、力、距离、TCP、食物、F/T 和判据）、CSV、磁盘快照、全部数值对照及入口／交付／撤离的 visual/collision 截图。桌面 viewer 开启、同步、退出通过；缺项或 viewer 不可用仍为 incomplete 并返回非零。

任务签名 schema_version=2，event_rules_version=2，包含边界 qacc／sensordata。旧任务快照或旧事件规则被拒绝；当前磁盘恢复重放观测误差为 0。M4 框架保留，教师、示范采集与策略训练仍未放行，详见 [M4 接口](m4_interfaces.md)。
