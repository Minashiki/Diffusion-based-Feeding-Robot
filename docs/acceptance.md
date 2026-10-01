# 新餐具 M1 验收：v1

2026-10-01 在现有 feedingrobot conda 环境完成。Panda 和 UR5e 的全部 15 项 M1 检查通过；209 项 pytest 通过。宿主环境自检、CUDA、DP/SAC 最小算法自检与桌面 viewer 均通过，这些算法自检不是策略训练。

| 机器人 | M1 项目 | 2 s 位置漂移 (mm) | 六轴最大跟踪误差 (mm) | 已知力矩误差 (N·m) | reset |
| --- | ---: | ---: | ---: | ---: | --- |
| panda | 15/15 | 2.599 | 3.379 | 1.31e-05 | 100，最大差异 0 |
| ur5e | 15/15 | 2.439 | 3.771 | 9.55e-06 | 100，最大差异 0 |

## 基线与验证范围

- 两款机器人使用同一新勺子和新盘子；勺头 130 个凸网格、柄/颈 15 个碰撞体，盘子 17 个碰撞体，145 个源勺子—盘子 pair 全部有效。碗保留为未启用资源，运行不读取 extract。
- 源 XML/OBJ、内部变换和惯量保持；装配移除自由关节，限定默认参数/材质作用域。工具子树质量 0.035 kg，连接坐标无质量/碰撞几何，完整子树参与 F/T 补偿。
- 盘面和水平勺头承载至少 2 s，温和运输维持承载；倾斜和横向加速度诊断均产生真实滑出、持续失去支撑和几何离开承载区。柄部真实接触不构成承载。
- 实际伺服运动经过盘面上方、低位、抬升和嘴前等待位；检查整段机械臂禁止接触。该路径是执行/几何基线，不是完整取餐和交付教师。
- 保持、六轴跟踪、停止、F/T 符号/作用点/动态补偿、100 次复位、受阻释放、过期/失败/非有限命令与状态、超限终止通过。M1 guards 同时执行共用契约、物理阻挡和新餐具回归。
- 初始物理诊断固定头部；独立检查恢复后的动态头部/下颌。viewer 开启/同步/关闭通过，并检查 overview、spoon、spoon_collision 和 plate 截图中的视觉/碰撞对齐。
- 1 ms 基准为 100 次求解迭代；固定 seeds 0/1/2 对照 0.5 ms，以及 200 次迭代并收紧求解容差。比较承载结局、TCP 位姿、接触峰值和冲量，现有收敛容差通过。

正常运行保留 0.05 m/s、0.5 rad/s、关节目标 0.8 rad/s、实际异常 2 rad/s，以及 5 N 接触/8 N 补偿腕力保护；两款机器人 gravcomp 均为 0。执行与传感容差未放宽。

新勺头在旧 0.8 m/s 脉冲下仍保持承载。独立加速度压力诊断使用横向 1.5 m/s、参考加速度 100 m/s²、四倍原 PD 增益和单独的关节/参考限制，实际加速度为 Panda 31.92、UR5e 18.75 m/s²；所有值记录在报告中。正常伺服增益与执行器输出力限制保持，诊断不作为正常运行配置或人体安全标准。

## 证据与复现

- [Panda 正式报告](../outputs/new_tableware/v1/m1/panda/report.json)、[UR5e 正式报告](../outputs/new_tableware/v1/m1/ur5e/report.json)：每项指标、轨迹 CSV、阈值、装配配置和实际输入 SHA256。
- [环境复核](../outputs/new_tableware/v1/environment.json)、[pytest 日志](../outputs/new_tableware/v1/pytest.log)、[JUnit](../outputs/new_tableware/v1/pytest.xml)、[机器可读摘要](acceptance_summary.json)。原 outputs/m0 保留。
- [Panda 场景](../outputs/new_tableware/v1/m1/panda/overview.png)、[UR5e 场景](../outputs/new_tableware/v1/m1/ur5e/overview.png)、[勺头承载](../outputs/new_tableware/v1/m1/panda/spoon.png)、[碰撞对齐](../outputs/new_tableware/v1/m1/panda/spoon_collision.png)。
- 新餐具上游仓库、版本与许可证未随资源提供，来源清单如实记录为待补充。

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot ur5e
conda run -n feedingrobot python -m pytest -q
```

正式检查包含桌面 viewer；桌面不可访问时不宣称完整通过。局部检查使用 --cases 和独立 --output，未选择项目标记 not_verified，总报告 incomplete 并返回非零。

M3/M4 保留共用框架与有效回归；本轮只适配模型入口、支撑/净空几何、快照和输入哈希，以及失效的旧坐标测试夹具。M3 正式事件/物理门槛、M4 完整教师与采集门槛尚未放行，未生成示范数据集或训练 DP/SAC。旧餐具输出、报告、旧变体与专属调参/成绩测试已清理。
