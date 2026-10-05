# 当前单豆 M4

固定碗、源勺和 `bean_000` 沿用已冻结 M1/M3。M4 教师接入 12 点舀取、精确回平与沉降、渐进释放和固定避让；观测 schema 3 为 Panda 108／UR5e 106 维，新增当前接收面位姿。接口和门禁见 [M4 文档](docs/m4_interfaces.md)。

当前完成状态只由匹配输入的 `outputs/single_bean/v1/m4/revision_3/<robot>/report.json` 及最终 `acceptance_audit.json` 确认。正式数据位于 `datasets/single_bean/v1/m4/panda/`；没有通过教师门禁不能采集。调试目录为 `outputs/single_bean/v1/m4_tuning/`。旧食物输出和诊断已清理，下文历史成绩不代表当前放行。

# FeedingRobot DPRL — 单豆喂餐原型

2026-10-03：默认场景切换为源勺、固定碗与一颗原生刚体 `bean_000`，模型版本 `single_bean_native_v1`。采用固定位置／姿态生成和自然沉降，Panda、UR5e 共用同一布局；本轮不验收随机布局。双机器人完整候选 M1 已通过；正式冻结以匹配当前输入的完整报告和清单为准。单豆 M1/M3 已冻结；M4 正在教师迁移与独立验收，正式放行以匹配输入的新报告及审计为准。

单豆 M3：`python -m feedingrobot.scripts.validate_m3 --robot all`，双机器人静态及动态完整流程、定向物理矩阵、数值对照、viewer 和完整 M1 回归通过后，才在 `outputs/single_bean/v1/m3/` 生成冻结清单。接口见 [M3 契约](docs/m3_interfaces.md)。零动作环境演示：`python -m feedingrobot.scripts.demo_m3 --robot panda --preset beans_in_bowl`。M3 证明驱动不作为 M4 教师放行证据。


```bash
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot all
```

完整入口顺序验收双机器人。报告位于 `outputs/single_bean/v1/m1/<robot>/m1d/`，仅最终 frozen 输入下全部必需项通过且哈希一致才发布独立 `freeze_manifest.json`。局部验收不发布清单。当前接口保留 `beans_in_bowl/beans_on_spoon/empty`，纯物理 Task 版本沿用 M3，Gym 外层快照 schema 为 3，逐豆位置／速度为 `(1,3)`、四元数为 `(1,4)`；模型签名拒绝旧 15 豆快照。

实施与判据见 [单豆 M1 范围](docs/m1_beans_rebuild_plan.md)、[验收说明](docs/acceptance.md) 和 [接口约定](docs/interfaces.md)。后续顺序为 **单豆 M1 → M3 完整舀取／携带／移到嘴前／释放／撤离及失败判定 → M4 教师与数据采集 → DP/RL**。多豆修复暂停；历史接触修正和失败报告保留，不作为单豆正式放行依据。

## 历史 15 豆及盘子 v2 记录

以下模型描述、旧命令、测试数量和阶段成绩仅对应历史版本，不能用于当前单豆放行；M3/M4 入口尚未迁移。

# FeedingRobot DPRL — M0 / M1 / M3

2026-10-03：从 M1 重建为勺子、固定碗与 15 颗 Beans，采用共享视觉 mesh＋同 body 原生 ellipsoid 碰撞体。实施与验收范围以 [Beans M1 重建方案](docs/m1_beans_rebuild_plan.md) 为准；旧 Beans SDF 专用文件已清理。以下 v2 成绩描述现有盘子与单块食物场景，原生 Beans 场景 **当前修订 4 的 M1-A/B 与单豆接入对照已通过；历史修订 2 的 M1-C 已通过；当前 15 豆动作及完整 M1-C/D 尚未通过，未冻结**，完整 M1 尚未通过。

当前默认模型为固定碗＋15 个原生刚体 Beans。M1-A 独立验收入口如下；报告仅证明装配、静态掩码匹配、编译和 100 步引擎冒烟检查，不证明沉降、承托、取餐或控制通过。

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1a --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1a --robot ur5e
```

历史 M1-A 报告已清理，阶段通过返回零，完整 `m1_status` 仍为 `incomplete`。`FeedingTask/reset/StateProvider` 已适配逐豆接口和 schema 3，提供 `beans_in_bowl/beans_on_spoon/empty`；失败 reset 会显式终止，空载 reset 可以恢复。新增 `python -m feedingrobot.scripts.validate_m1b --robot panda|ur5e`（分别传入机器人名称），报告为同目录下 `m1b_report.json`。修订 2 双机器人 seeds 0–9 及三组接触／沉降数值设置均已通过，当时报告位于修订 2（现已清理），参数仍为 candidate。M1-C 独立入口为 `python -m feedingrobot.scripts.validate_m1c --robot panda|ur5e --cases ... --output ...`；未选项目保留 `not_verified`，全部必需项通过才返回零。历史修订 2 的双机器人各 14 项 M1-C 必需验收均通过：每台 100 次 seed=7 reset 状态误差为零，seeds 0/1/2 共用固定壁辅助轨迹真实取豆，warning 为零，最大 Bean 穿透 0.2194 mm。96 项相关测试通过；修订 2 的 M1-B 三组数值设置及 viewer 回归也通过。完整 `validate_m1` 已迁移，M1-D 正在复验；M3/M4 尚未迁移。不保留单块食物兼容字段。**下文的盘子／单块食物接口、命令与验收成绩均为历史 v2 记录，不适用于当前 Beans 默认模型。**

当前修订 4 采用 65° 下半部接入与原生 CCD 精度修正，单豆三组真实取豆及原数值对照通过；129 项相关测试、双机器人 M1-A/B 和 60 项共同初态沉降／viewer 回归通过。完整 15 豆仍有下降／勺侧夹挤，参数为 candidate，M1-C/D 不放行。证据及修改范围见 历史接触修正记录（旧食物报告已清理）。

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
conda run -n feedingrobot python -m feedingrobot.scripts.replay outputs/single_bean/v1/m4/revision_3/panda/episodes/<seed> --robot panda
```

`demo_m4` 默认从盘中食物开始，失败返回非零；去掉 `--headless` 打开只读 viewer。教师放行后，`collect --viewer` 可在采集时显示环境；窗口按墙钟最多 30 FPS 刷新，不固定暂停物理推进，关闭窗口后继续采集。命令会保存独立演示目录。`validate_m4 --trials 3 --cases teacher replay --output outputs/single_bean/v1/m4_tuning/partial` 只做局部检查，不满足正式放行条件。复用报告或采集目录要求源码／配置／资产哈希一致，版本变化应使用新输出目录；M0 环境记录保留。

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

正式验收含桌面 viewer 开启／同步／退出检查，需可连接显示服务的宿主会话。无界面可用 `--cases event_logic environment snapshot bowl carry receiver unsupported force penetration convergence`；未选 viewer 会标为 `not_verified`，总报告为 incomplete 并返回非零，不会伪报全通过。建议局部调试用 `--output outputs/calibration/m3_<name>`，避免覆盖正式结果。

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
- `configs/acceptance_m4.json`：M4 独立数值容差；用户授权修订 3 接触确认 1.1 秒，其余值保持 M3 冻结容差。
- `outputs/m0/doctor.json`：逐项环境与算法自检。
- `outputs/single_bean/v1/m1/<robot>/`：重建后生成的新模型物理报告与轨迹。
- `outputs/single_bean/v1/m3/<robot>/`：重建后生成的事件、快照、收敛和环境验收证据。
- `outputs/single_bean/v1/m4/revision_3/<robot>/`：重建后生成的教师、重放及放行报告；正式采集显式传入匹配的 `--gate`。
- `outputs/single_bean/v1/m4_tuning/`：新模型调试结果，与正式验收分开。
- `SimModelPlann.md`：从 M1 重建的主方案；`docs/interfaces.md`：执行接口与坐标/时间契约，餐具坐标与完整工具负载已按新模型适配。

部分检查可用 `validate_m1 --cases hold tracking` 单独运行；未选择的项目仍标记 `not_verified`，退出码为非零。建议使用 `--output outputs/calibration/<name>` 保存调试结果，避免覆盖完整报告。

UR5e 沿用上游位置增益乘 2、速度增益乘 √2 的现有配置，输出力限制保持上游值；两款机械臂均关闭 `gravcomp`。基准求解迭代数为 100，收敛对照为 1 ms／0.5 ms 与 200 次迭代。新勺头在旧 0.8 m/s 脉冲下仍保持承载，加速度掉落诊断单独使用横向 1.5 m/s、100 m/s² 参考加速度和四倍伺服增益；记录实际速度、加速度和掉落证据，正常运行增益、限速及力保护不变。

## 来源

旧项目复用提交：`2756d33cb1fc0536e425d29ec7bf0b4f6c7c4f50`。
Menagerie 资产提交：`c96a32d28fb5da84da38c1da4d749e7a13212855`。
文件哈希、修改说明和许可证见 `third_party_manifest.json`、`assets/third_party/` 与 `docs/licenses/`。

旧 M2 控制器、旧验收报告和旧项目目录均不是运行依赖。Git 提交与推送由用户管理。

M1-D 完整入口：`python -m feedingrobot.scripts.validate_m1 --robot all`，顺序验证 Panda、UR5e；单机器人和 `--cases` 局部验收不会发布冻结清单。历史 M1-D 报告已清理；仅最终 frozen 输入下双机器人全部必需项通过且结束哈希复核一致，才生成 `outputs/beans_native/v1/m1/freeze_manifest.json`。候选数值矩阵通过后须在最终输入下重新运行。控制 RTF 仅记录；沉降每组最多 5 s 仿真时间及 60 s 墙钟。控制对照覆盖 12 个运动用例，命令 20 ms、采样 10 ms，公共时间网格和终态比较 TCP、腕力峰值、语义接触峰值及 applied 冲量。跨数值回放只用于验收，不放宽公共 schema 3 快照检查。M3/M4 仍为 `not_verified`。

M1-D 第一轮候选完整入口已实际顺序执行双机器人，退出码 1；A/B/C、viewer、性能及来源清单通过，两个机器人输入哈希前后相同。半步长及高精度取豆峰值对照超限，未冻结；历史候选证据已清理。阶段切换计时已修正为全回合共同 20 ms 命令网格，10 ms 采样不变；共同命令时钟完整复验已结束，命令返回 1；当时的报告已清理。两机器人 A/B/C、viewer、性能及来源清单通过，初态回放核验通过，运行前后及双机器人输入 SHA256 一致；M1-D 数值对照失败，参数保留 candidate，未生成 freeze_manifest.json。原修订 2 接触参数（solref 2 ms）及轨迹目标保持不变；2.5 ms 隔离试验不能通过 Panda seed 1 的 5 s 沉降门槛，不采用。掉落仍按首次确认即结束，终态按各自结束状态比较，时间差只记录。
