# 当前单豆接口与执行约定

模型版本 `single_bean_native_v1`；正式编译仅 bean_000，一个自由关节、一个 ellipsoid 和同 body visual。`load_model(robot_id,timestep)` 四元返回不变；快照 schema 3 不变，模型签名拒绝旧 15 豆状态。

- 索引 bean_ids=[bean_000]；body/joint/visual/collision 数组为 `(1,)`；bean_qpos `(1,7)`、bean_dofs `(1,6)`。
- snapshot 的 bean_positions／线速度／角速度为 `(1,3)`，bean_quaternions 为 `(1,4)` wxyz；policy_obs.bean_relative_world 为 `(1,3)`。无旧 food_* 兼容字段。
- reset 保留 beans_in_bowl、beans_on_spoon、empty；碗中固定生成位置／姿态写入 scene 配置，seed 不随机化豆布局。出生速度为零，自然沉降后不清零速度；故障处理、唯一 step_physics、只读支撑诊断和严格 set_state 兼容性保持。
- 勺上预置只用于承载诊断，真实舀取从碗中开始。原接触保护、补偿、twist／IK／伺服、20 ms 命令和 10 ms 验收轨迹不变。
- M3/M4 保持 not_verified；task_mode=True 明确拒绝尚未迁移的单豆全流程。M1 数组变化不代表 M3 观测、奖励或训练接口已放行。

验收入口及范围见 [单豆 M1 方案](m1_beans_rebuild_plan.md)。
## 历史记录（下文仅适用于原版本）

# M1 接口与执行约定

## Native Beans M1-B/C/D（2026-10-02）

`FeedingTask/reset/StateProvider` 已迁移到 15 颗原生 Bean，快照签名 schema 为 3。M1-B 修订 2 正式双机器人 seeds 0–9 与接触／沉降数值对照已通过；M1-C 双机器人正式验收已通过，完整 M1 数值验收失败，未冻结；M3/M4 构造明确抛出 `NotImplementedError`。以下旧 v2 单块食物接口只作历史记录，不适用于 Beans。

- 默认 reset 为 `beans_in_bowl`；另外支持 `beans_on_spoon`（仅 bean_000 在勺头）和 `empty`。旧 `food_on_plate/food_on_spoon` 与单块食物 scenario 参数被拒绝。
- reset 先空载稳定机器人，再放置 Bean；沉降通过唯一 `step_physics()` 推进。正常预置超时／穿透／越界／warning 会返回 `terminated=True`、`failure_reason=bean_reset_settling_failed` 和 `reset_diagnostics.status=failed`，锁存 adapter 故障，禁止继续推进。可用 `reset(preset="empty")` 清理故障后进行空载控制诊断。
- 成功或失败的 reset 均保留自然得到的 Bean 速度，回合时钟重置为零。诊断包含出生间隙、初末完整 integration state、逐豆轨迹、接触证据、失败 ID、稳定窗口与计时。
- snapshot 字段为 `bean_ids[15]`、`bean_positions[15,3]`、`bean_quaternions[15,4]`（wxyz）、`bean_linear_velocities_world[15,3]`、`bean_angular_velocities_world[15,3]`；不再提供 `food_position`。
- StateProvider 的 `policy_obs.bean_relative_world[15,3]` 替代单食物相对位置。`oracle_info` 提供逐豆质量、接触、有限碗边界、碗／勺头支撑链和 reset 结果；这不是已冻结的 M3 策略观测。
- `contacts` 是 boundary 接触，`applied_contacts` 保留积分器实际使用的接触；记录两端 Bean ID、实际 condim/friction/solref/solimp。保护仍按语义 pair 聚合，多豆不得分摊绕过阈值。
- `get_state/set_state` 保存完整 `mjSTATE_INTEGRATION`、adapter、monitor 和执行记忆，拒绝旧 schema 或变更模型／配置／验收参数的状态。

独立命令：`conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1b --robot panda|ur5e`（分别传入机器人名称）。修订 2 报告为 `outputs/beans_native/v1/m1/<robot>/revision_2/m1b_report.json`，任一必需项失败返回非零。完整 `validate_m1` 已迁移到 M1-D；demo 使用新勺上预置，不代表真实扫取通过。

M1-C 入口：`python -m feedingrobot.scripts.validate_m1c --robot panda|ur5e`。`--cases` 选择 `hold tracking wrench faults reset head carry tilt acceleration reachability guards sweep_seed_0 sweep_seed_1 sweep_seed_2` 中的用例；未选项为 `not_verified`。`--output` 指定报告目录，默认为 `outputs/beans_native/v1/m1/<robot>/m1c/`。全部必需项通过才返回零，完整 M1 仍为 `incomplete`，M1-D 和 M3/M4 未验证。 当前报告中两台机器人各 14 项均通过；100 次 seed=7 reset 最大状态差为 0，固定壁辅助轨迹在 seeds 0/1/2 分别取出 bean_003、bean_012、bean_006。静止观察为 10 s，连续低速承托仍要求 0.5 s。

`bean_diagnostics(task, spoon_frame=True)` 只读扩展 `direct_support`、`support_edges`、勺头范围、`in_spoon_head`、`bean_positions_tcp`、`bean_linear_velocities_tcp` 与 `bean_angular_velocities_tcp`。相对速度扣除 TCP 平移和转动；近邻／区域包含不能单独证明承托。勺柄和勺颈接触不计承载，双重碗／勺支撑不计拾取。schema 3 保持不变。连续承托／脱离窗口由验收脚本逐物理步计算。

M1-C 报告保存输入哈希、实际接触和控制参数、逐物理步峰值、10 ms 逐豆轨迹与接触证据。每用例的 `*_states.pkl` 是可信本地完整初末回放快照（包含 integration、adapter、monitor、执行状态及各自实际控制参数），`*_trajectory.json` 是可读诊断；pickle 文件只用于本地产生的可信记录。倾斜与加速用例分别记录指定豆的连续脱离，压力增益只用于独立加速任务实例；回放初末快照时分别使用对应参数，恢复增益、控制限幅、头部驱动和保护配置后再调用 `set_state()`。

## 历史 v2 接口


M1-A 索引：`bean_ids` 为稳定的 bean_000–bean_014；`bean_bodies/bean_joints/bean_visual_geoms/bean_collision_geoms` 为形状 `[15]` 的整数数组；`bean_qpos` 为 `[15,7]`，`bean_dofs` 为 `[15,6]`，每行列出对应自由关节的全部地址。`bowl_geoms` 包含 17 个碰撞体，`bean_id_by_geom` 将 Bean collision geom ID 映射到稳定 Bean ID。仅 collision 属于 `food`，visual 不属于接触组；Beans 不进入 `tool_bodies`。`load_model(robot_id, timestep)` 的四元返回结构保持不变。

## 最小调用

```python
from feedingrobot.sim.task import FeedingTask

task = FeedingTask("panda")  # 或 "ur5e"
task.reset(seed=0, preset="food_on_spoon")
t = task.data.time
task.adapter.set_twist([0.01, 0, 0, 0, 0, 0], t, t + 0.1)
for _ in range(100):
    state = task.step_physics()
    if state["terminated"]:
        break
task.adapter.stop()
observation = task.provider.observe()
```

应用层通过这些接口交互；`model/data` 的公开访问用于 M1 诊断和 viewer，不应成为后续教师/策略写入物理状态的通道。

## 动作与时钟

`RobotAdapter.set_twist(twist, command_time, valid_until)` 接受基座坐标系下 TCP 原点的 `[vx, vy, vz, wx, wy, wz]`，单位 m/s、rad/s。线速度和角速度分别限制范数与变化率。旋转通过 SO(3) 指数映射更新；MuJoCo 和 Mink 四元数均按 **wxyz** 使用。

时间戳为 `data.time` 的仿真秒，不是墙钟。拒绝未来时间戳和倒序命令；已过期命令清空当前命令。首次 reset 后时间为零，物理 tick 默认 1 ms。每次 `step_physics()` 只调用一次 `mj_step`；适配器自己不推进物理。

Mink 使用独立 `Configuration`，硬约束冻结所有非机械臂 DOF，只将关节参考写入位置伺服 `ctrl`。机器人实际 `qpos` 仅在 reset 时初始化。头部和食物只通过场景物理运动。

正常运行默认线速度 0.05 m/s、角速度 0.5 rad/s，参考与实际偏差上限为 0.02 m、0.2 rad。关节目标速度限制为 0.8 rad/s；实际速度异常阈值为 2 rad/s。具体参数以机器人和场景配置为准。

## 停止与故障

- `stop()` 清空命令、目标推进与速度积分，在当前实际关节位置锁定新的伺服目标。它不是关闭重力或瞬间清零实际速度；停稳通过物理测试验证。
- 过期命令进入 `expired`，可以接受新的有效命令。
- 参考偏差受阻、工作空间超限、IK 失败、非法非有限命令锁存故障，恢复需要 `reset()`。
- 接触超限或实际状态非有限终止回合。后续 `step_physics()` 抛出异常，必须 reset；不继续保持位置顶住障碍后宣称安全。
- 非有限物理状态的步进结果仅保证 `terminated` 和 `failure_reason` 两个字段有效，调用者应立即停止该回合。

自碰撞和机械臂与桌面、餐盘、嘴部、地面的禁止碰撞由 Mink 约束。勺与任务物体的接触保留给物理求解。接触监控同时检查积分所用载荷和当前边界状态；冲量只对实际物理子步积分一次，避免遗漏短峰值或重复计数。

接触组累计力阈值为 5 N，补偿后腕部力阈值为 8 N。这些只是 P0 仿真的工程保护参数，不是人体安全标准。

## 状态出口

`StateProvider.observe()` 返回三个独立字典：

| 出口 | 内容 |
| --- | --- |
| `policy_obs` | robot_id、时间、实际 q/dq、TCP 位姿及世界系 twist、食物/嘴部相对世界系几何、F/T、当前接触、帧龄、执行状态 |
| `oracle_info` | 真实接触列表、回合终止原因、接触峰值、冲量和超限持续时间 |
| `scenario_state` | seed、场景驱动状态和未来事件容器，不进入策略 |

q/dq 的长度分别为 Panda 7、UR5e 6；动作始终为 6 维。默认诊断模式的阶段标签固定为 `m1_diagnostic`。显式启用 `task_mode=True` 后使用 M3 阶段与事件；Gym、奖励和完整快照契约见 [M3 接口规范](m3_interfaces.md)。

F/T 的 `raw_wrench_sensor` 是传感器坐标系中的父体对子体载荷；`wrench_world_at_tcp` 转为外界对工具的世界系力/力矩，力矩原点为 TCP。`compensated_wrench` 再减去刚性工具重力与惯性载荷。补偿使用模型惯性和当前运动状态，不使用接触求解力；这是理想仿真补偿，尚未模拟真实传感器误差。

## 复位与机器人切换

`FeedingTask.reset(seed, preset)` 支持 `food_on_plate`、`food_on_spoon` 和诊断用 `empty`。reset 清理外力、命令、参考、故障、事件和接触统计，在内置伺服下沉降后重置回合时钟。相同配置和种子的重放应可复现。

添加机器人需提供资产及完整 JSON 配置，特别是关节/执行器顺序、工具安装、TCP/F/T、机器人碰撞体归属、复位姿态和限制。加载阶段检查名称、标量转动关节、一对一位置伺服映射及输出力限制；只复制 XML 不构成已支持的机器人。

M1 可达性检查通过同一 IK／位置伺服／物理入口实际执行盘面上方、低位、抬升及嘴前等待路径，并检查禁止接触。它仍不是完整舀取、交付和撤离教师的成功证明。

## 新餐具装配与版本

所有入口使用新勺子和新盘子。`tool_variant` 已移除，不提供旧餐具兼容分支。公共 site 名称保持 `tcp`、`ft_site`、`plate_frame`、`mouth_entry`、`mouth_receiver`；新位姿以场景和机器人配置为准。工具局部 +x 为舀取方向，+z 为承载面法向。

`RobotIndex.tool_bodies` 包含传感器下游完整刚性子树；`spoon_geoms` 包含 145 个实际碰撞体，`scoop_geoms` 包含 130 个勺头凸网格，`handle_geoms` 包含 15 个柄/颈碰撞体，`plate_geoms` 包含 17 个盘子碰撞体。视觉网格无碰撞，不参与接触、承载或净空判断。F/T 补偿按子树中每个有质量 body 汇总重力、惯性和力矩臂。

默认勺子质量为 0.035 kg，连接坐标无额外质量。三种 reset 保持原接口；食物姿态与出生高度按实际支撑面计算。任务支持条件要求有效勺头接触及新几何区域，勺柄接触不构成承载。

快照继续使用 v2，编译模型和配置签名拒绝旧餐具状态；观测字段、单位与维度保持不变。回合和验收哈希包含新 XML、OBJ、接触片段及机器人网格，运行不依赖提取目录。旧回合、旧报告与旧放行门槛均已清理，不提供迁移。

M1-D 完整入口：`python -m feedingrobot.scripts.validate_m1 --robot all`，顺序验证 Panda、UR5e；单机器人和 `--cases` 局部验收不会发布冻结清单。报告为 `outputs/beans_native/v1/m1/<robot>/m1d/m1d_report.json`；仅最终 frozen 输入下双机器人全部必需项通过且结束哈希复核一致，才生成 `outputs/beans_native/v1/m1/freeze_manifest.json`。候选数值矩阵通过后须在最终输入下重新运行。控制 RTF 仅记录；沉降每组最多 5 s 仿真时间及 60 s 墙钟。控制对照覆盖 12 个运动用例，命令 20 ms、采样 10 ms，公共时间网格和终态比较 TCP、腕力峰值、语义接触峰值及 applied 冲量。跨数值回放只用于验收，不放宽公共 schema 3 快照检查。M3/M4 仍为 `not_verified`。

M1-D 第一轮候选完整入口已实际顺序执行双机器人，退出码 1；A/B/C、viewer、性能及来源清单通过，两个机器人输入哈希前后相同。半步长及高精度取豆峰值对照超限，未冻结；证据保存在 `outputs/beans_native/v1/m1/m1d_candidate_display/<robot>/m1d/`。阶段切换计时已修正为全回合共同 20 ms 命令网格，10 ms 采样不变；共同命令时钟完整复验已结束，命令返回 1；正式报告位于 `outputs/beans_native/v1/m1/<robot>/m1d/`。两机器人 A/B/C、viewer、性能及来源清单通过，初态回放核验通过，运行前后及双机器人输入 SHA256 一致；M1-D 数值对照失败，参数保留 candidate，未生成 freeze_manifest.json。原修订 2 接触参数（solref 2 ms）及轨迹目标保持不变；2.5 ms 隔离试验不能通过 Panda seed 1 的 5 s 沉降门槛，不采用。掉落仍按首次确认即结束，终态按各自结束状态比较，时间差只记录。
