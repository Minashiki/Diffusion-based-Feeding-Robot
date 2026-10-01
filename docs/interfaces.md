# M1 接口与执行约定

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
