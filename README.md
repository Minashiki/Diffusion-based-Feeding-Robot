# FeedingRobot DPRL — M0 / M1 / M3

新勺子和新盘子已接入机械臂，共用完整碰撞模型、TCP/F/T 和执行链；碗保留但不启用。M1 正式验收与新模型证据见 [验收说明](docs/acceptance.md)。M3 双机器人正式物理验收已通过；M4 框架保留，教师放行待后续实施；尚未生成示范集或训练 DP/SAC。

在 `feedingrobot` conda 环境中运行的状态驱动喂餐仿真基础。默认 Panda，支持通过配置切换六轴 UR5e；六维 TCP twist 经 Mink IK 转为 MuJoCo 内置关节位置伺服目标。

M0/M1 提供 P0 接触场景、执行保护、腕部 F/T、环境自检和物理验收。M3 增加 50 Hz Gymnasium 环境、八阶段任务事件、接触判据、奖励和完整快照。M2 已取消；M4 教师和采集管线现已实现但尚未物理放行；正式 DP/SAC 策略训练尚未实施。

## M4 教师与数据管线（未放行）

M4 已加入参数化教师、场景采样、20 Hz 动作／50 Hz 观测／物理子步日志、磁盘分片、恢复标注、独立数据划分与物理命令重放。新模型 M3 已正式通过，M4 待教师正式验收，启动示范集未生成；不能据接口或单次 pickup 声称 M4 完成。见 [M4 接口](docs/m4_interfaces.md)，新模型实施顺序与放行要求以 [主方案](SimModelPlann.md) 第 12 节为准。

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.demo_m4 --robot panda --headless
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m4 --robot panda --workers 4
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m4 --robot ur5e --workers 2
# 教师门槛通过后才允许执行：
conda run -n feedingrobot python -m feedingrobot.scripts.collect --robot panda
# 扩展训练回合预算，保留独立验证／测试及恢复配额：
conda run -n feedingrobot python -m feedingrobot.scripts.collect --robot panda --train-episodes 1000
conda run -n feedingrobot python -m feedingrobot.scripts.replay outputs/m4/panda/episodes/<seed> --robot panda
```

`demo_m4` 默认从盘中食物开始，失败返回非零；去掉 `--headless` 打开只读 viewer。教师放行后，`collect --viewer` 可在采集时显示环境；窗口按墙钟最多 30 FPS 刷新，不固定暂停物理推进，关闭窗口后继续采集。命令会保存独立演示目录。`validate_m4 --trials 3 --cases teacher replay --output outputs/calibration/new_tableware/m4_partial` 只做局部检查，不满足正式放行条件。复用报告或采集目录要求源码／配置／资产哈希一致，版本变化应使用新输出目录；M0 环境记录保留。

快照的任务签名升级至 v2，补齐立即恢复边界的加速度／传感器缓存；并记录 event_rules_version=2；旧任务快照和旧事件判据不兼容。

运行仅依赖 `assets/task/tableware/` 中的新勺子和盘子，不读取 `extract/`。保留源 XML/OBJ，装配时移除自由关节、限定默认参数作用域、解析网格路径，并筛选全部 145 个勺子—盘子 pair。工具为 0.035 kg 的刚性子树；传感器补偿完整子树负载，碰撞监控区分勺头与勺柄。所有入口使用同套餐具，旧工具变体接口已移除。来源、坐标与接入方式见 [餐具说明](assets/task/tableware/README.md)。

## M3 任务环境

新餐具 v2：Panda、UR5e 各 26/26 项通过，各 63 个基准物理运行及 126 组数值对照；M1 各 15/15，全量 pytest 246 项通过。v1 M1 证据保留。取餐采用真实勺头网格与承托载荷、盘坐标最低点及真实接触判据，正常保护不变。

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.demo_m3 --robot panda --headless
conda run -n feedingrobot python -m feedingrobot.scripts.demo_m3 --robot ur5e
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m3 --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m3 --robot ur5e
```

`demo_m3` 是零动作保持的 Gym 接口演示，默认从食物在勺上开始，不是全流程教师。正常任务 reset 默认食物在盘上。注册入口为 `gym.make("FeedingRobot-v0", robot_id="panda")`，先 `import feedingrobot` 完成注册；六维归一化动作仍走同一个 TCP 执行链。

环境支持 `get_state()/set_state()`，成功／失败与超时分别返回 `terminated/truncated`。字段、单位、阶段和奖励定义见 [M3 接口规范](docs/m3_interfaces.md)，新模型判据适配与验收要求见 [主方案](SimModelPlann.md)。

正式验收含桌面 viewer 开启／同步／退出检查，需可连接显示服务的宿主会话。无界面可用 `--cases event_logic environment snapshot plate carry receiver unsupported force penetration convergence`；未选 viewer 会标为 `not_verified`，总报告为 incomplete 并返回非零，不会伪报全通过。建议局部调试用 `--output outputs/calibration/m3_<name>`，避免覆盖正式结果。

## 运行

在本仓库根目录执行：

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.doctor
conda run -n feedingrobot python -m feedingrobot.scripts.demo --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.demo --robot ur5e
conda run -n feedingrobot python -m feedingrobot.scripts.demo --robot panda --headless
conda run -n feedingrobot python -m pytest -q
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot ur5e
```

`doctor` 默认检查 CUDA 和桌面 viewer，需在可访问显卡与显示服务的宿主会话运行。`--skip-viewer` 会明确标记该项未验证，整体报告不会伪报通过。

## 环境重建

当前环境已经安装本仓库的 editable 包；自检会检查实际导入路径，防止运行旧项目。

在另一台 Linux x86-64 机器的仓库根目录运行 `conda env create -f environment.yml`，需要支持 CUDA 13.0 wheel 的 NVIDIA 驱动。环境已存在时先导出快照，再执行：

```bash
conda run -n feedingrobot python -m pip install --dry-run -r requirements.lock.txt -e .
conda run -n feedingrobot python -m pip install -r requirements.lock.txt -e .
conda run -n feedingrobot python -m pip check
```

`requirements.txt` 保存直接依赖，`requirements.lock.txt` 保存本次解析后的全部 pip 版本，`environment.yml` 固定 Python 版本。原生 conda 包的精确清单、安装前快照、依赖解析报告和设备信息位于 `outputs/m0/`。当前按源码 checkout＋editable 安装交付，不是脱离资产目录的独立 wheel。

## 配置与报告

- `configs/robots/`：机器人关节映射、安装变换、复位姿态及执行限制。
- `configs/scene.json`：公共场景、物理步长、头部驱动与接触保护。
- `configs/acceptance.json`：正式检查的固定阈值和明确标记的掉落诊断配置。
- `configs/task.json`：M3 阶段、接触事件及奖励的固定工程阈值。
- `configs/acceptance_m3.json`：M3 固定 seeds 与步长／求解精度对照的事件时间、位置、力及冲量容差。
- `outputs/m0/doctor.json`：逐项环境与算法自检。
- `outputs/new_tableware/v2/m1/<robot>/`：重建后生成的新模型物理报告与轨迹。
- `outputs/new_tableware/v2/m3/<robot>/`：重建后生成的事件、快照、收敛和环境验收证据。
- `outputs/new_tableware/<version>/m4/<robot>/`：重建后生成的教师、重放及放行报告；正式采集显式传入匹配的 `--gate`。
- `outputs/calibration/new_tableware/`：新模型调试结果，与正式验收分开。
- `SimModelPlann.md`：从 M1 重建的主方案；`docs/interfaces.md`：执行接口与坐标/时间契约，餐具坐标与完整工具负载已按新模型适配。

部分检查可用 `validate_m1 --cases hold tracking` 单独运行；未选择的项目仍标记 `not_verified`，退出码为非零。建议使用 `--output outputs/calibration/<name>` 保存调试结果，避免覆盖完整报告。

UR5e 沿用上游位置增益乘 2、速度增益乘 √2 的现有配置，输出力限制保持上游值；两款机械臂均关闭 `gravcomp`。基准求解迭代数为 100，收敛对照为 1 ms／0.5 ms 与 200 次迭代。新勺头在旧 0.8 m/s 脉冲下仍保持承载，加速度掉落诊断单独使用横向 1.5 m/s、100 m/s² 参考加速度和四倍伺服增益；记录实际速度、加速度和掉落证据，正常运行增益、限速及力保护不变。

## 来源

旧项目复用提交：`2756d33cb1fc0536e425d29ec7bf0b4f6c7c4f50`。
Menagerie 资产提交：`c96a32d28fb5da84da38c1da4d749e7a13212855`。
文件哈希、修改说明和许可证见 `third_party_manifest.json`、`assets/third_party/` 与 `docs/licenses/`。

旧 M2 控制器、旧验收报告和旧项目目录均不是运行依赖。Git 提交与推送由用户管理。
