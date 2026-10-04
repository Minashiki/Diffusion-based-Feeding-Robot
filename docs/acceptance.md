# 单豆喂餐原型验收

当前模型为 `single_bean_native_v1`，只编译 bean_000，固定布局自然沉降，Panda／UR5e 顺序执行。正式必需矩阵、原门槛和冻结流程见 [单豆 M1 方案](m1_beans_rebuild_plan.md)。单豆版双机器人完整候选 M1 已通过，最终冻结以匹配当前输入的报告和清单确认；M3/M4 not_verified，旧模型成绩不迁移。

当前报告使用 `outputs/single_bean/v1/m1/`，最终放行凭双机器人完整报告及同目录 `freeze_manifest.json`；历史失败输出保持。seeds 0–9 与 0–2 只验证固定布局重复性。双豆接触不属于正式验收，独立回归保留。
## 单豆 M1 完整候选结果

双机器人各 20/20 必需项通过，M1-A/B/C/D 均 passed，viewer 开启／同步／关闭通过。相关回归 137 项通过；两机器人各 100 次 reset 最大状态差为 0；三组共同初态共 60 个沉降用例通过。三组设置、seeds 0/1/2 共 18 次真实舀取均通过。M3/M4 保持 not_verified。

| 机器人 | 首次接触高度／竖向半径 | 首次向上力（mN） | 推进上升（mm） | 数值 TCP 最大差（mm） | 沉降最大穿透（mm） |
| --- | ---: | ---: | ---: | ---: | ---: |
| panda | -0.612120 | 0.460860 | 0.571599 | 1.092568 | 0.118141 |
| ur5e | -0.651882 | 0.479157 | 0.568816 | 0.997093 | 0.118141 |

完整候选报告位于 `outputs/single_bean/v1/m1/candidate/<robot>/m1d/`，输入前后及跨机器人一致。最终 frozen 输入报告位于 `outputs/single_bean/v1/m1/<robot>/m1d/`；统一 [冻结清单](../outputs/single_bean/v1/m1/freeze_manifest.json) 是最终发布依据，需全部必需项通过且输入、报告和证据 SHA256 一致。本文的数值表来自已经通过的完整候选；最终复验成绩读取输出报告，避免验收后改写被哈希的输入文档。

## 历史记录（下文仅适用于原版本）

# 新餐具 M1 / M3 正式验收：v2

本文仅记录历史 `new_tableware_v2` 盘子／单块食物成绩，不适用于当前 `beans_native_v1`。当前入口已迁移为原生 Beans 完整 M1-D，证据见 `outputs/beans_native/v1/m1/<robot>/m1d/`；M3/M4 保持未验证。

2026-10-01 在现有 `feedingrobot` conda 环境完成双机器人新餐具重建与正式放行。M1、M3 所有必需项目通过；M4 教师、示范采集和 DP/SAC 策略训练仍未放行。

| 机器人 | M1 | M3 | 物理基准运行 | 数值对照 | 最大事件时间差 | 最大 TCP 路径差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Panda | 15/15 | 26/26 | 63 | 126 | 2 ms | 0.0414 mm |
| UR5e | 15/15 | 26/26 | 63 | 126 | 1.5 ms | 0.0416 mm |

全量 pytest **246 项通过**，4 条 Gymnasium 无界 Box 提示；无失败或错误。Gymnasium／SB3 checker、VecEnv 终态语义、磁盘快照重放与不兼容状态拒绝通过。两款机器人磁盘恢复后的最大观测误差均为 0。桌面 viewer 开启、同步、退出通过，入口、交付、撤离的视觉／碰撞截图已检查。

## 判据与接口

- 勺支撑使用新勺头 130 个碰撞网格的实际横向界限、从食物质心沿 −TCP Z 的射线命中，以及实际接触沿 TCP +Z 的净承托载荷。仅有侧向接触、包围盒空角或勺柄载荷不能形成承载；旧承载盒不再决定 M3 pickup。
- 离盘高度使用全部食物角点在 `plate_frame` 局部法向上的最低值，联合真实盘面载荷判断离盘及回落，保留 2 mm 净离盘标准。
- 入口检查当前新勺头及食物完整包络；撤离检查全部餐具碰撞几何。浅入口 TCP 嘴坐标 `[0.004,0,-0.006] m` 已经真实物理验证。代理口腔几何保持，更深插入路径留待 M4。
- 保留 `FeedingRobot-v0`、六维归一化基座系 twist、20 ms step、八阶段、Panda 96／UR5e 94 维观测顺序、奖励权重、终止／截断和子步峰值／冲量契约。完整规范见 [M3 接口](m3_interfaces.md)。
- 任务签名 schema_version=2、event_rules_version=2，拒绝旧事件规则快照。reset 刷新 Mink Configuration 与 ConfigurationLimit，避免诊断下颌范围变化后使用旧 IK 限位缓存。

## 真实物理矩阵与冻结对照

21 类用例：`plate`、`carry`、`pickup_lift`、`plate_return`、`receiver`、`receiver_edge`、`receiver_outside`、`unsupported`、`unsupported_recovered`、`force`、`contact_safe`、`contact_force`、`penetration`、`shallow`、`entry`、`closed`、`recover`、`unreleased`、`early_withdrawal`、`post_delivery_loss`、`handle`。

覆盖盘面沉降、稳定承载、离盘与回落、失去支撑及宽限恢复、入口净空与开口不足、承接边界、到嘴未脱勺、未释放撤离、撤离后留存与食物丢失、正常接触与超限、浅接触与持续深穿透。每个运行记录预期结局、完整事件和里程碑奖励；合成测试保留为逻辑回归，物理成绩依据真实接触。

所有食物、阶段历史和诊断场景预置仅在 reset 设置并记录。后续动作经过统一 TCP twist、IK、伺服和物理步进。定向用例固定头部，正常动态头部／下颌另由 M1 验证。离盘用例从预置勺上食物抬升，承接用例预置食物到真实下颌支撑面，再执行物理交付与撤离；它们不替代 M4 从盘舀取和完整喂餐教师验收。

正常／超限接触使用独立诊断目标：10×10×20 mm，solref `[0.25,1]`、solimp `[0.05,0.995,0.001,0.5,2]`、priority 2。正常接触在实际载荷达到 0.05 N 的物理边界停止并保留参考，超限用例持续推进直到 5 N 保护触发。诊断目标按基准物理状态冻结，不修改正常场景。交付后丢失用例在真实 delivery 后施加最多 0.2 s 的嘴坐标 `[-0.1,0,0.3] N` 食物脉冲，不瞬移状态。

每类固定 seeds **0/1/2**，比较 **1 ms／100 iterations**、**0.5 ms／100 iterations**、**1 ms／200 iterations 且求解容差缩小十倍**。每机器人 63 个基准运行、126 组对照。比较事件顺序、里程碑／失败时间、终态及事件 TCP 位置、20 ms TCP 路径采样、全部接触组峰值与冲量。

[冻结容差](../configs/acceptance_m3.json) 保持：事件时间 10 ms、TCP 2 mm，力误差为 0.05 N 或基准值的 20%，冲量误差为 0.005 N·s 或 20%，取两者较大值。调试后未为通过而放宽。边界载荷补充峰值，冲量仅按积分器实际载荷计算，不重复积分。

## M1 回归与保护

新勺子、盘子共用完整碰撞及统一执行链，碗未启用。工具子树质量 0.035 kg，完整子树参与 F/T 补偿。装配、盘面／勺头沉降与承载、保持／跟踪、F/T、100 次 reset、实际 IK 伺服路径、异常命令、保护、步长／求解精度对照、动态头部和桌面 viewer 全部通过。

Panda／UR5e 的 2 s 位置漂移分别为 2.599／2.439 mm，六轴最大跟踪误差分别为 3.379／3.771 mm。正常运行保持 0.05 m/s、0.5 rad/s、关节目标 0.8 rad/s、实际异常 2 rad/s，以及 **5 N 接触／8 N 补偿腕力保护**；伺服增益、输出力限制和连续确认时间保持，gravcomp 均为 0。

M1 加速度掉落仍使用独立诊断配置：横向 1.5 m/s、参考加速度 100 m/s²、四倍 PD 增益和独立关节／参考限制。该配置不进入正常任务；v1 M1 证据完整保留。

## 证据与复现

- M1：[Panda 报告](../outputs/new_tableware/v2/m1/panda/report.json)、[UR5e 报告](../outputs/new_tableware/v2/m1/ur5e/report.json)。
- M3：[Panda 报告](../outputs/new_tableware/v2/m3/panda/report.json)、[UR5e 报告](../outputs/new_tableware/v2/m3/ur5e/report.json)。包含冻结配置、输入 SHA256、完整事件、预期结局、接触组峰值／冲量、全部数值对照和 viewer 证据；每用例保存 `metrics.json`、每物理子步 `physics.jsonl.gz` 和 `trajectory.csv`。
- [pytest 日志](../outputs/new_tableware/v2/pytest.log)、[JUnit](../outputs/new_tableware/v2/pytest.xml)、[机器可读摘要](acceptance_summary.json)。
- Panda：[入口视觉](../outputs/new_tableware/v2/m3/panda/viewer/entry/entry_visual.png)、[入口碰撞](../outputs/new_tableware/v2/m3/panda/viewer/entry/entry_collision.png)、[撤离后碰撞](../outputs/new_tableware/v2/m3/panda/viewer/receiver/retracted_collision.png)。
- UR5e：[入口视觉](../outputs/new_tableware/v2/m3/ur5e/viewer/entry/entry_visual.png)、[入口碰撞](../outputs/new_tableware/v2/m3/ur5e/viewer/entry/entry_collision.png)、[撤离后碰撞](../outputs/new_tableware/v2/m3/ur5e/viewer/receiver/retracted_collision.png)。交付及撤离视觉图位于同目录。
- 保留 [v1 Panda M1](../outputs/new_tableware/v1/m1/panda/report.json)、[v1 UR5e M1](../outputs/new_tableware/v1/m1/ur5e/report.json)、[v1 环境复核](../outputs/new_tableware/v1/environment.json) 和 `outputs/m0/`。新餐具上游版本与许可证仍按原来源清单待补充。

```bash
conda run -n feedingrobot python -m pytest -q
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot ur5e
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m3 --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m3 --robot ur5e
```

默认输出为 `outputs/new_tableware/v2/m1/<robot>/` 与 `v2/m3/<robot>/`。局部调试显式使用 `--output outputs/calibration/new_tableware/<name>`。所有必需项目通过才为 passed；局部运行、未选择项目或 viewer 不可用保持 incomplete，返回非零。

下一阶段按 [主方案](../SimModelPlann.md) 第 12 节重建 M4，先验证真实取餐，再验证承载运输、入口交付与撤离，冻结后执行独立教师门槛。本轮未生成正式示范集或训练策略。

M4 当前重建进展及取餐停步证据见 [M4 重建状态](m4_rebuild_status.md)。M1/M3 v2 放行结论保持；该记录不构成 M4 放行。

M1-D 第一轮候选完整入口已实际顺序执行双机器人，退出码 1；A/B/C、viewer、性能及来源清单通过，两个机器人输入哈希前后相同。半步长及高精度取豆峰值对照超限，未冻结；证据保存在 `outputs/beans_native/v1/m1/m1d_candidate_display/<robot>/m1d/`。阶段切换计时已修正为全回合共同 20 ms 命令网格，10 ms 采样不变；共同命令时钟完整复验已结束，命令返回 1；正式报告位于 `outputs/beans_native/v1/m1/<robot>/m1d/`。两机器人 A/B/C、viewer、性能及来源清单通过，初态回放核验通过，运行前后及双机器人输入 SHA256 一致；M1-D 数值对照失败，参数保留 candidate，未生成 freeze_manifest.json。原修订 2 接触参数（solref 2 ms）及轨迹目标保持不变；2.5 ms 隔离试验不能通过 Panda seed 1 的 5 s 沉降门槛，不采用。掉落仍按首次确认即结束，终态按各自结束状态比较，时间差只记录。
