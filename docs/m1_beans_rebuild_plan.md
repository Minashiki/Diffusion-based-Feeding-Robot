# 单豆喂餐原型：M1 正式范围

日期：2026-10-03。模型版本：`single_bean_native_v1`。双机器人完整候选 M1 已通过；最终冻结以匹配当前输入的独立双机器人报告和冻结清单为准。M3/M4 为 `not_verified`。历史 15 豆和盘子模型成绩不能用于本版本放行。

## 场景与初始布局

默认模型只编译 `bean_000`：一个自由关节、原生 ellipsoid 和同 body 视觉 mesh。沿用源勺、水平固定碗、豆子质量／惯量／摩擦、修订 4 的接触参数、安装点与控制链。无隐藏的其他豆子，无人工承托，动作中不写豆子 qpos/qvel。

固定生成位置（碗坐标，m）为 `[0.01987645624922746,0.012001153227614164,0.0075]`，四元数（wxyz）为 `[-0.1723840605901834,-0.24869534818743078,0.8057862669099752,0.5090607542363261]`。XY／姿态取自历史 `profile65_single_matrix/baseline/states.pkl` 的 initial bean_011；仅生成高度提高，以满足原 0.5 mm 出生间隙。数值固化在配置中，运行不读取历史文件。生成时速度为零，随后自然沉降；沉降结束保留自然速度。

保留 `beans_in_bowl`、`beans_on_spoon`、`empty`。勺上预置只证明承载诊断，真实取豆必须从碗中 reset 开始。seeds 0–9 沉降、seeds 0–2 舀取是固定布局重复验证，不覆盖随机布局。两机器人使用同一布局和碗坐标路径。

## M1 必需项目与证据

- M1-A：单豆刚性归属、质量惯量、7 个 qpos／6 个速度 DOF、视觉与碰撞对齐、固定碗、145 个勺—碗 pair、掩码及引擎冒烟。Panda nq/nv=19/18，UR5e=18/17。
- M1-B：豆—碗、豆—勺头真实接触、三种 reset、seeds 0–9 三组共同初态沉降、viewer。双豆接触为 `not_applicable`，不属于正式必需矩阵；独立双豆回归夹具保留。
- M1-C：保持／六轴跟踪／F/T／异常命令／100 次 reset／动态头部／静态和温和携带／倾斜及加速度掉落／可达性／保护／seeds 0–2 真实舀取。
- M1-D：以上全部项目、12 个运动用例的共同初态数值对照、取豆后 viewer、性能记录、来源清单和输入一致性。三组设置为 1 ms／100 iterations／tolerance 1e-8，0.5 ms／100／1e-8，1 ms／200／1e-10；CCD tolerance 保持 1e-12。

真实取豆需初始碗支撑、豆子完整离开碗沿、实际勺头向上承托、勺头区域及连续 0.5 s 低速窗口。只靠邻近、侧碰、勺柄或双重碗／勺支撑不算取起。逐物理步记录首次前缘接触点、实际向上力、推进阶段位移和承托窗口；上升量重新实测，不要求复现历史 0.57 mm。

原门槛保持：沉降最多 5 s、完整稳定窗口 0.5 s、线速 1 mm/s、角速 0.1 rad/s、穿透 0.4 mm；5 N 语义接触／8 N 腕力保护；数值 TCP 位置 2 mm、旋转 0.035 rad，峰值力 0.05 N 或 20%，冲量 0.005 N·s 或 20%。控制命令 20 ms、轨迹 10 ms；RTF 记录，不新增完整控制 RTF 门槛。

独立加速度掉落诊断保留 1.5 m/s、100 m/s² 和四倍 PD 增益，首次满足原连续 0.1 s 脱离条件的事件时间仍记录。该诊断在 0.3 s 起脉冲，统一观察 0.2 s 至回合 0.500 s 再比较终态；不在不同掉落确认时刻直接比较高速运动的终态。倾斜诊断保持首次确认即结束。原 Panda 加速度对照的终态差为 2.472 mm、共同时间点最大差为 1.032 mm，失败证据保留在 `candidate_before_drop_horizon_fix/`；固定观察终点的双机器人六组局部物理对照通过原门槛。正常控制器与物理推进不变。

## 运行与放行

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 --robot all
```

Panda、UR5e 顺序运行。默认报告位于 `outputs/single_bean/v1/m1/<robot>/m1d/m1d_report.json`；候选完整运行使用 `--output outputs/single_bean/v1/m1/candidate`。独立 A/B/C 命令保留，默认输出也迁到单豆目录；`--cases` 未选项仍为 `not_verified`，不发布冻结清单。

两机器人完整候选通过后，定稿文档／验收摘要，准备 `frozen` 配置并同步来源哈希，在最终输入下完整重跑。仅双机器人 A/B/C/D 全部通过、前后与跨机器人输入 SHA256 一致时，生成 `outputs/single_bean/v1/m1/freeze_manifest.json`；记录参数、环境和所有报告／轨迹／快照／截图的证据哈希。失败恢复 candidate 并保留失败证据。冻结后修改源码、配置、测试或文档需要新输入复验。最终成绩在输出报告和清单中确认。

## 后续阶段

顺序为 **单豆 M1 → 单豆 M3 → M4 教师及采集 → DP/RL**。暂停多豆修复，保留历史 15 豆失败报告。

M3 必须从碗中自然初态，经统一 twist／IK／伺服／step_physics 完成舀取、携带、移到嘴前、释放、撤离，并验证实际承接和撤离后的留存。需覆盖未取到、途中掉落、到嘴未释放、提前撤离、交付后丢失、接触超限和超时等失败；阶段预置诊断不能替代完整流程成功。当前 task_mode 显式拒绝未迁移的 M3/M4。只有该完整流程验证通过，才能进入 M4 教师、示范采集和后续 DP/RL 训练。


## 历史记录（下文仅适用于原版本）

# M1 重建方案：勺子、碗与 15 颗原生刚体 Beans

日期：2026-10-03。状态：**历史修订 2 的 M1-C 已通过；当前修订 4 的 M1-A/B 与单豆修正通过，完整 15 豆动作仍失败，M1-C/D 不放行，参数未冻结**。

当前默认模型已替换为固定碗与 15 个原生刚体 Beans；共享视觉资源、装配、逐豆索引及资产哈希已实现。`FeedingTask/reset/StateProvider` 已迁移；默认碗中和勺上 reset 已通过修订 2 的 M1-B 门槛；失败仍显式终止并提供诊断。空载 reset 可用于控制回归，完整 `validate_m1` 已迁移，M1-D 复验进行中。M3/M4 单块食物任务未迁移，不能作为本阶段通过依据。

## M1-A 当前交付与验收

独立入口为 `python -m feedingrobot.scripts.validate_m1a --robot panda|ur5e`（分别传入机器人名称），无需实例化 `FeedingTask`。报告写入 `outputs/beans_native/v1/m1/<robot>/m1a_report.json`；仅 M1-A 全部用例通过时返回零，完整 M1 保持 incomplete，M1-B/C/D 标为 not_verified。

双机器人均通过：15 个 freejoint／ellipsoid／共享 mesh visual、同 body 刚性归属、105 个 Bean qpos／90 个速度 DOF、零关节耗散、显式质量惯量、编译后视觉对齐、水平固定碗与 17 个 collision、145 个源勺—碗 pair、静态掩码匹配及 100 步引擎冒烟。Panda 为 nq=117/nv=102，UR5e 为 nq=116/nv=101；使用 1 ms／100 iterations，全部 MuJoCo warning 为零。该结果不证明正常 reset、真实接触承托、沉降、IK 或保护通过。

OBJ 为项目离线生成的 16×32 椭球网格，共 482 顶点、960 个朝外三角面及解析平滑法线。编译出生位姿采用确定性的 8＋7 圆环与单位四元数，专用于 M1-A；不是第 6 节的随机 reset 实现。

本阶段共 15 项测试通过：`tests/test_beans_native.py` 和 `test_tableware.py` 中资源参数／输入哈希两项测试；旧任务测试未删除、未添加 skip，也未宣称全量 pytest 通过。实现复用模型加载器；独立引擎冒烟使用隔离 MjData，不新增运行时物理推进接口。后续仍沿用 FeedingTask.step_physics() 的唯一所有权。

## M1-B 当前交付与验收

独立入口为 `python -m feedingrobot.scripts.validate_m1b --robot panda|ur5e`（分别传入机器人名称）。修订 1 报告保留在 `outputs/beans_native/v1/m1/<robot>/m1b_report.json`；修订 2 使用 `--numerical-check --output outputs/beans_native/v1/m1/<robot>/revision_2`，报告写入该目录，失败返回非零。正常 reset 不隐藏失败，不清零沉降速度；失败状态禁止继续推进，空载 reset 可恢复。schema 3 拒绝旧单块食物快照。

参数修订 1 的正式 seeds 0–9 在两机器人上均未通过：5 s 内没有全体共同低速 0.5 s 窗口，最大穿透约 1.24–2.08 mm，超过 0.4 mm。Bean—碗、Bean—Bean、Bean—勺头真实接触及支撑均已产生；物理接触存在不能替代沉降验收通过。报告保存逐豆证据与失败时刻，viewer 保存 visual/collision/overlay 的碗近景、全景和工具近景。

参数修订 1 继续标为 candidate。Panda seed 0 对 9 组现有 solref/solimp/滑动摩擦候选进行了筛查：更硬候选可使穿透降到 0.4 mm 内，但仍不能形成共同低速窗口；没有候选冻结。详见 `outputs/calibration/beans_native/m1b_contact_trials.json`。该修订只筛查滑动摩擦。修订 2 按接触与沉降修复计划加入原生接触滚动／扭转摩擦，以耗散近乎纯滚动时滑动摩擦无法消耗的转动能量；不增加关节耗散、睡眠或人工承托，不改变验收门槛。本轮数值筛查见 `outputs/calibration/beans_native/m1b_revision2_margin_screen.json`：同时检查独立接触用例与 15 豆沉降，并匹配完整物理初态；初始候选在半步长下仍有峰值超限，修订 2 候选见第 5 节；中间失败报告保存在 `revision_2`，不覆盖修订 1。M1-B 修订 2 完成时，M1-C/D、100 次 reset 和完整控制循环性能尚未验证；本轮 M1-C 结果见下节。

修订 2 正式报告位于 `outputs/beans_native/v1/m1/panda/revision_2/m1b_report.json` 与 `outputs/beans_native/v1/m1/ur5e/revision_2/m1b_report.json`，均为 passed。单豆、双豆、勺头、三种 reset、M1-A 前置检查及 viewer 均通过；三组数值设置的 seeds 0–9 共 60 个沉降用例均达到共同低速／真实承托／包含窗口 0.5 s，沉降时间为 1.022–1.722 s，最大几何／接触穿透为 0.335204 mm，MuJoCo warning 为零。78 项相关测试通过；参数修订 2 仍为 candidate，完整 M1 仍为 incomplete。报告新增边界失败原因、最深穿透几何对／来源／时刻及仅按速度计算的低速窗口，保留原完整窗口判据。

## M1-C 修订 2 的历史交付与验收

Panda、UR5e 各 14 项必需验收全部通过，命令均返回 0；96 项相关回归通过。两台机器人各完成 100 次 seed=7 `beans_in_bowl` reset，注入旧命令、外力、故障和统计后均正确清理，完整状态最大差为 0（容差 `1e-9`），保留自然沉降速度。所有正式用例 warning 为零，最大 Bean 穿透为 0.2194 mm。单豆静态载荷相对误差分别为 `4.13e-7`、`8.02e-9`，温和携带误差为 0.308%／0.567%，均满足 20% 门槛。

| 机器人 | seed 0：bean_003 | seed 1：bean_012 | seed 2：bean_006 |
| --- | ---: | ---: | ---: |
| Panda 连续低速承托 | 1.214 s | 6.248 s | 9.759 s |
| UR5e 连续低速承托 | 9.378 s | 9.636 s | 1.107 s |

报告为 `outputs/beans_native/v1/m1/<robot>/m1c/m1c_report.json`；初末完整快照、实际参数、输入哈希、10 ms 逐豆轨迹、接触证据和性能计时同目录保存。历史候选报告保留在 Panda 的 `trajectory_yaw45/`、`trajectory_yaw35/` 和 `outputs/calibration/beans_native/m1c/`。当前输入版本的 M1-B 三组数值设置、seeds 0–9 和 viewer 再验收均通过，见 `m1c/m1b_regression_final_numerical/m1b_report.json`。M1-C 通过不代表完整 M1 放行；M1-D 的控制数值对照、参数冻结及 M3/M4 保持未验证。

独立入口为 `python -m feedingrobot.scripts.validate_m1c --robot panda|ur5e`，支持 `--cases` 和 `--output`；默认输出到 `outputs/beans_native/v1/m1/<robot>/m1c/`。未选用例保留 `not_verified`；全部必需用例通过才返回零。保持 schema 3，物理推进只调用现有 `step_physics()`，运行中不写 Bean qpos/qvel。三组控制数值对照和参数冻结明确留到 M1-D。

修订 2 采用壁辅助舀取：75° 下探、沿碗 +x 以 5 mm/s 将 Beans 推向碗壁，再借助壁约束回平和抬升。初始 `[-5,+12] mm` 候选与中间净空／漏取失败保留在 `outputs/calibration/beans_native/m1c/`。修订 2 锁定轨迹从 `[+5,+12] mm`、yaw −35° 开始，推豆阶段预留 3 mm 豆侧空间；提前回平至 60° 时转为 yaw −10°、横向 +6 mm，选定勺头前缘点相对碗底高度为 8.5 mm，后续 45°/30°/15°/0° 高度为 28/45/80/110 mm，预留 1 mm 壁侧空间。TCP 抬至碗底上方 180 mm 后，勺头短暂向上 5° 收豆并回平，再静止观察 10 s；连续承托和 M1-B 低速确认仍为 0.5 s，M1-B reset 的 5 s 沉降上限不变。目标由完整勺子碰撞顶点和碗壁内侧计算；真实碰撞、穿透、机械臂避碰、5 N 语义接触和 8 N 腕部保护保持原门槛。双机器人使用同一碗坐标轨迹，通过各自基座坐标发送 twist；新增回归核验目标一致至 `1e-9`。

逐豆诊断新增勺头投影区域、TCP 相对位置／速度、直接承托和支撑链证据。只在初始碗承托、实际离碗、失去碗承托并随勺抬升后连续受勺头承托且达到 M1-B 低速阈值 0.5 s 时确认拾取。预置单豆承载不计扫取。指定豆倾斜／加速掉落要求连续 0.1 s 同时失去实际勺头接触与承托并离开承托区域；加速压力设置仅作用于独立任务实例。

报告保存输入哈希、实际物理／控制参数、完整初末回放状态、10 ms 逐豆轨迹和接触／支撑链、逐物理步峰值与连续窗口、失败 ID／时刻／阶段及物理、IK、诊断、日志计时。100 次 seed=7 reset 注入旧命令、外力、故障、执行记忆和统计并检查清理，以 1e-9 容差检查状态复现；保留自然沉降速度。单豆静态承载 2 s、温和携带 1 s，静态及携带平均腕部载荷均按豆重验收，相对误差限 20%。

正式双机器人结果为 passed。完整 M1 仍为 `incomplete`；M1-D、M3 和 M4 保持未验证。

## M1-D：UR5e bean_011 顺序排查（2026-10-03）

逐物理小步记录、同状态重放和单豆隔离已完成，详见 [排查报告](m1d_ur5e_seed0_diagnosis.md)。基准 12.675 s 的 bean_011 同时受勺头、碗底和相邻两块碗壁约束。给半步长计算相同的 12.40 s 状态，腕部峰值仍达到 2.41 N；从 10.40 s 入豆前状态仅保留该豆碰撞时，三组峰值均约 0.0103 N。证据支持多豆接触参与路线分叉，之后局部夹挤维持大力。对 12.416 s 两组状态逐一固定几何再求解，力主要随状态而非求解设置改变；约 50 nm 位置／15 µrad 姿态差足以使豆—碗壁间隙跨过 50 µm margin 边界。

四项独立单变量筛查均淘汰：仅扩大 sweep 间隙导致漏取；仅将入豆横向位置改为 6 mm 在其他 seeds 触发保护或漏取，力／冲量对照也失败；仅将 solimp[0] 改为 0.95 导致沉降／穿透失败，改为 0.98 虽改善基准舀取，但半步长 seed 6 穿透达到 2.04 mm，舀取力对照亦失败。没有将这些候选写入正式参数或轨迹。86 项相关回归通过，基准局部重放位置和力误差为零，试验源输入运行前后哈希一致。尚无满足组合要求的修正，因此本轮未进入新的双机器人完整复验；M1-D 仍失败，参数保持 candidate，没有生成冻结清单，M3/M4 保持 not_verified。

后续 [动作试验](m1d_transition_followup.md) 已完成抬高 wall_align、拆分退／抬／转及横向侧移。抬高目标触发 contact_limit；拆分动作丢失承托或 blocked。前进 2.5 mm 后侧移 −y 3 mm 的候选从完整初态运行，三组均真实取到 bean_003，腕部峰值降至 0.069／0.066／0.079 N、最大穿透记录为 0；但半步长豆—勺峰值／豆—豆冲量及 200 iterations 豆—勺冲量仍超限。正式参数和轨迹未采用该候选，输入运行前后哈希一致；尚未扩大其他 seeds／Panda 或完整验收。下一项诊断转向 bean_003 的 sweep／wall_align／wall_45 接触历史，原冻结和 M3/M4 状态保持不变。


## M1-D：单豆前缘接触修正候选（2026-10-03）

新证据和修正见 [单豆接触报告](m1d_single_bean_contact_fix.md)。原 75°、yaw −35° 入豆首先接触 bean_011 的上半部，力向下；yaw 0°、从 `[-5,+12] mm` 空位下降再推进，首次接触转到下半部并向上受力，单豆真实进入勺内。保持原装配时，该 75° 候选加 `ccd_tolerance=1e-10 m` 在 UR5e 单豆和 15 豆 seed 0 均通过三组原力／冲量／TCP 对照；这些局部结果不代替完整 M1-D。

固定相同几何的接触查询及只含一颗 ellipsoid／一个源勺面 prism 的最小模型，确认 061/104/105 三个实际姿态会在粗 CCD 精度下出现反向法线。70° 的半步长用例仍在 `1e-10 m` 出现反向，因此正式新候选采用 `1e-12 m`，保留原生 CCD 和 35 次 CCD 迭代，三组力求解设置不变。新增法线回归覆盖这三个姿态。修订 3 的 70° 单豆三组原数值门槛均通过，采用 70° 作为新候选入口，75° 基准保留，65° 原装配受手腕／碗壁间隙限制。

Panda 原装配无法安全完成同一正向低位入豆；握持点沿原勺柄向尾端移动 10 mm 的试验通过 seed 0 真实取豆。双机器人候选 `spoon_position.x=0.06 m`，保持现有无质量连接的刚性点安装假设；这不是实体夹具握持验收。候选不改变源 mesh、豆子形状、质量、摩擦、solref、margin、速度或保护／验收阈值；仅将 solimp[0] 从 0.99 提高到 0.995，以控制新精度下的瞬时穿透。修订 3 的双机器人完整入口已结束，`outputs/beans_native/v1/m1/contact_fix_candidate_v3/` 两份报告均失败、输入哈希不变。carry 短暂分离、seed 1 真夹挤、seed 2 低速连续窗口不足，完整数值条件未通过；历史报告保留，参数继续为 candidate，M3/M4 为 not_verified。

当前修订 4 采用 65° / yaw 0°、`[-5,+12] mm` 与 `solimp=[0.99,0.999,0.0002,0.5,2]`。先仅恢复近接触阻抗到 0.99，完整共同 carry 初态六项通过；再缩短过渡 width 到 0.2 mm，UR5e 30 项共同初态沉降与双机器人六项自然 reset 携带全部通过。新自然 reset 的单豆三组都从下方接触 bean_011、向上运动并真实进入勺内，原力／冲量／TCP 对照通过。完整 15 豆仍会在不同布局压住其他豆子；提前抬升、85° 下降后回平、yaw −10° 等对照仍不能联合通过，不采用。全部逐小步证据、接触画面与失败范围见上述报告；仅单豆修复通过不能宣布 M1-C/D 当前版本通过或冻结。控制器、唯一物理推进入口和所有原门槛不改。

修订 4 配置采用后的 129 项相关测试通过。双机器人已顺序完成 M1-A/B、viewer 及三组 seeds 0–9 共 60 项沉降回归，均通过，最大穿透 0.330560 mm；报告在 `outputs/beans_native/v1/m1/contact_fix_candidate_v4/`，运行前后全输入哈希一致。完整 M1-C/D 未通过；不能沿用修订 2 的 M1-C 分数放行当前参数。

## 1. 目标与范围

将 M1 场景重建为 `机器人 + 原有勺子 + 固定碗 + 15 颗 Beans + 桌面 + 原有头部/下颌代理`。正常食物初态在碗内，勺子刚性安装到机器人末端。Panda 为首轮调试机器人，UR5e 使用同一执行接口完成回归。

每颗豆是独立刚体。视觉使用共享 mesh，碰撞使用 MuJoCo 原生 ellipsoid；两个 geom **直接挂在同一个 body 下**。MuJoCo 的 body 运动自然带动两者，无需位置同步脚本。geom 的这种刚性归属由 [MuJoCo MJCF 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom) 定义。

本轮明确采用以下假设：

- 共 15 颗同种、同尺寸豆子，稳定 ID 为 `bean_000`–`bean_014`。
- 首轮外形为光滑椭圆豆，不制作凹陷、纹理细节或变形动画。
- 碰撞统一使用 ellipsoid。capsule 暂不加入，不同时维护两种碰撞路线。
- 初始工程尺寸为 **14 × 9 × 8 mm**，均匀密度暂取 **1000 kg/m³**；这些是仿真假设，未经过真实豆子材料标定。
- 使用现有 MuJoCo 3.14.0、Mink、内置关节位置伺服及物理推进入口。
- 不建模凹陷卡住、嵌合、细部接触、熟豆变形、湿润黏附或压碎。
- M1 验证装配、控制、接触、沉降、简单扫取与承载；完整喂餐教师、粒子交付事件、示范采集和训练留到 M3/M4 及后续阶段。

成功标准：15 颗豆能独立运动、相互碰撞并受碗和勺子承托；视觉与碰撞对齐；真实机器人运动能完成简单扫取、抬升和掉落诊断；双机器人控制与保护回归通过；所有结论由新场景报告建立。

## 2. 现有基础与需要改动的部分

重建前的 [模型加载器](../src/feedingrobot/sim/model.py) 接入勺子和盘子，硬编码单个 `food` / `food_joint`；`asset_files()` 排除未启用的碗。[FeedingTask](../src/feedingrobot/sim/task.py) 的 reset、质量与摩擦设置、snapshot 和 StateProvider 也依赖单块食物。[M1 验收入口](../src/feedingrobot/scripts/validate_m1.py) 明确要求无碗，并通过单个食物位置判定承载和掉落。因此不能只在 XML 中增加 15 个 body。

直接复用的内容：

- 六维 TCP twist → Mink IK → 内置位置伺服 → `mj_step` 的控制链。
- 关节映射、Panda/UR5e 配置、限速、参考偏差限制、命令过期及故障停机逻辑。
- `FeedingTask.step_physics()` 对物理推进的唯一所有权，以及 applied 接触和 boundary 接触的现有采样顺序。
- 原勺子视觉 mesh、全部 **145 个碰撞 geom**，其中 **130 个勺头、15 个勺柄/勺颈**；工具质量 **0.035 kg**，TCP 与 F/T 安装。
- 原碗视觉 mesh、底盘及 16 个侧壁，共 **17 个原生碰撞 geom**。碗资源位于 [bowl.xml](../assets/task/tableware/bowl/bowl.xml)。
- 腕部力/力矩坐标变换与完整刚性工具子树的重力、惯性补偿。

必须适配的内容：碗装配、食物数组索引、reset、诊断状态、接触归属、M1 承托判据、IK 障碍物集合和输入哈希。控制算法及物理推进顺序不重写。

## 3. 单颗 Bean 的资源和刚体结构

### 3.1 视觉 mesh

新增 `assets/task/foods/beans/meshes/bean_visual.obj`。离线生成一份米制、以质心为原点、长轴沿局部 +x 的光滑椭球三角网格，外包尺寸为 14 × 9 × 8 mm；无需寻找第三方豆子资产，也不依赖旧 SDF 输出。

采用一份低面数 OBJ 和统一豆色即可。所有 15 个 visual geom 引用同一 mesh asset；不为每颗豆复制文件。视觉 mesh 与碰撞 ellipsoid 使用相同原点、轴向和尺寸，避免外形差异引起明显悬浮或穿入。

`visual` 使用 `contype=0`、`conaffinity=0`、`density=0`、`group=1`；`collision` 使用 `density=0`、`group=3`。质量与惯量由 body 的显式 inertial 提供。编译器使用 `inertiafromgeom="auto"`，核验实际编译结果，避免视觉网格改变质量；其行为见 [inertiafromgeom 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#compiler-inertiafromgeom)。

### 3.2 质量和惯量

令半轴 `a=0.007 m`、`b=0.0045 m`、`c=0.004 m`。原生 ellipsoid 的 `size` 是三个半轴，不是完整长度，见 [geom size 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-size)。

均匀实心椭球采用：

```text
m   = ρ × 4πabc / 3
Ixx = m × (b² + c²) / 5
Iyy = m × (a² + c²) / 5
Izz = m × (a² + b²) / 5
```

单豆质量约 **0.5278 g**，15 豆总质量约 **7.9168 g**。单豆对角惯量为 `[3.8264598520724e-9, 6.8612383554401e-9, 7.3098577863727e-9] kg·m²`。质心在 `[0,0,0]`，惯性主轴与 body 轴一致。

### 3.3 MJCF 模板

以下是装配模板；mesh 文件已在 M1-A 制作，编译出生位姿用于冒烟检查，正常 body 出生位姿后续在 reset 设置：

```xml
<!-- 合并到主场景的 asset 中，仅声明一次。 -->
<mesh name="bean_visual_mesh"
      file="assets/task/foods/beans/meshes/bean_visual.obj" />

<!-- 每颗豆直接挂在 worldbody 下；名称逐颗递增。 -->
<body name="bean_000" pos="0 0 0.1">
  <freejoint name="bean_000_joint" />
  <inertial pos="0 0 0" quat="1 0 0 0"
            mass="0.00052778756580309"
            diaginertia="3.8264598520724e-9 6.8612383554401e-9 7.3098577863727e-9" />
  <geom name="bean_000_visual" type="mesh" mesh="bean_visual_mesh"
        pos="0 0 0" quat="1 0 0 0" group="1"
        contype="0" conaffinity="0" density="0"
        rgba="0.45 0.18 0.07 1" />
  <geom name="bean_000_collision" type="ellipsoid"
        size="0.007 0.0045 0.004" pos="0 0 0" quat="1 0 0 0"
        group="3" contype="2" conaffinity="3" density="0"
        priority="2" condim="6" friction="0.25 0.0001 0.0001"
        solref="0.002 1" solimp="0.99 0.999 0.001 0.5 2"
        margin="0.00005" gap="0" rgba="0 0.55 1 0.25" />
</body>
```

装配器重复同一模板 15 次，不新增泛化粒子框架。路径在加载时解析为绝对路径，与现有餐具资源处理一致。使用 `freejoint`，避免继承餐具 joint 默认阻尼和 armature；编译后检查每豆六个 DOF 的 damping、armature、frictionloss 为零。

必须验证每豆两个 geom 的 `geom_bodyid` 相同、15 个 visual 的 `geom_dataid` 指向同一资源。每个 freejoint 占 7 个 qpos 和 6 个 qvel，自由豆共占 105 个 qpos 和 90 个速度 DOF；用名称查询地址，不假设它们位于数组末尾。

## 4. 碗、勺子与场景装配

碗替换主场景中的盘子；盘子资产留在仓库，正常场景不再加载。移除旧单块 `food`。保留桌面、地面、头部、下颌和机器人基础配置。

装配碗时复用现有 `_merge_asset()`：限定 default/材质作用域、解析网格路径、移除碗顶层 freejoint、保留内部变换与全部碰撞体。仅调整碗外层安装变换，使底盘水平、下表面贴桌。碗是固定刚体，不通过大质量或强伺服模拟固定。

建立 `bowl_frame`，原点在真实底盘上表面，+z 指向碗口。现有桌面顶面为 **−0.02 m**，底盘厚度为 **4.4 mm**，水平安装后底盘上表面应为 **−0.0156 m**。碗的平面位置首轮取旧食物区域 `[0.45,-0.18] m`，最终以机器人实际可达性检查确定，不能直接复制旧盘子 z 或姿态。

`contact_pairs.xml` 中按有效名称只选择 **145 个勺子—碗 pair**，这个数量已按源资产核对。不得加载旧勺子—盘子 pair，也不将全部 290 个 pair 原样 include。保留现有源餐具接触参数。

语义接触组使用 `arm / spoon / bowl / food / table / mouth / floor`：15 个豆 collision 都属于 `food`，另保存逐豆 ID；visual 不作为接触体。`group=1/3` 只承担显示分组。

豆子的掩码候选为 `contype=2, conaffinity=3`，使豆—豆及豆—现有餐具接触可由原生动态碰撞产生。逐类检查实际掩码的双向匹配，验证豆—豆、豆—碗、豆—勺头、豆—桌面/地面和豆—嘴代理接触；不通过增加大量显式 Bean pair 代替掩码检查。

在 `RobotAdapter` 的机器人环境障碍物集合中将 `plate` 改为 `bowl`。继续允许勺子任务接触；所有豆自由 DOF 仍由现有 `DofFreezingTask` 在 **IK 参考求解**中冻结，实际豆运动仍由物理引擎独立计算。

## 5. 原生接触参数与物理推进

修订 2 使用以下候选参数；M1-B 接触／沉降通过不代表材料标定或完整 M1 参数冻结：

| 参数 | 修订 2 候选取值 |
| --- | --- |
| 物理步长 | 1 ms |
| 积分器 / 求解器 | implicitfast / Newton |
| 求解迭代数 | 100 |
| 摩擦锥 / impratio | elliptic / 10 |
| Bean 接触维度 | condim=6 |
| Bean friction | `[0.25, 0.0001, 0.0001]` |
| Bean solref | `[0.002, 1]` |
| Bean solimp | `[0.99, 0.999, 0.001, 0.5, 2]` |
| Bean priority / margin / gap | `2 / 0.00005 / 0` |

源碗、勺子 collision 默认 priority 为 0；Bean 设为 2，使动态 Bean 接触采用上表参数，避免被碗的较高滑动摩擦覆盖。实际接触参数仍须从 `data.contact` 核对。显式勺子—碗 pair 采用 pair 自己的参数。组合规则见 [MuJoCo Contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters)。

保留 `step_physics()` 的推进、驱动写入、接触保护和传感器流程。运行中不写 Bean qpos/qvel，不锁定豆与勺子，不添加吸附、人工承托力或沉降末尾速度清零。

先检查单豆落下、碗底承托、豆—豆接触、勺头承托，再检查 15 豆。旧 SDF 标定参数不作为原生模型的已验证材料参数，不继承其极小步长、搜索起点、插件或引擎补丁。

修订 1 的 condim=3 只能耗散滑动，无法有效抑制近乎无滑移的持续滚动。修订 2 的 condim=6 提供接触法线扭转及切平面滚动阻力；friction 后两项单位为 m，均为 0.0001 m，仅在接触中产生阻力矩，离开接触后不施加关节阻尼。法向参数收紧以控制早期落底及豆间碰撞峰值；实测接触必须核验 condim、五个有效 friction 分量及 solref/solimp，不能改验收阈值。修订 2 另采用 0.05 mm 原生接触 margin，使引擎在微小间隙内开始接触求解，降低离散首次接触和短暂失去接触带来的尖峰；gap 为 0，视觉及 ellipsoid 尺寸不变。两侧 geom 的 margin 相加，因此 Bean—餐具为 0.05 mm、Bean—Bean 为 0.10 mm。这是接触皮肤近似，静态表面间隙也可达到同量级，不代表真实豆子材料标定；报告同时核验实际 includemargin。

## 6. 15 豆索引、reset 和状态

### 6.1 最小索引扩展

在现有 `RobotIndex` 增加稳定顺序的 `bean_bodies`、`bean_joints`、`bean_qpos`、`bean_dofs`、`bean_visual_geoms`、`bean_collision_geoms` 和 `bowl_geoms`。接触记录可通过 collision geom → Bean ID 的映射定位具体豆子。

不能将 `bean_000` 冒充旧 `food`，也不能用豆群质心代替逐豆承托、掉落或穿透判断。删除或适配本轮替换后产生的单块 food 索引和配置，不为历史数据添加兼容包装。

### 6.2 reset 预置

| 预置 | 用途与规则 |
| --- | --- |
| `beans_in_bowl` | 正常初态；15 豆在碗内无重叠出生并自然沉降 |
| `beans_on_spoon` | 单豆承载/掉落诊断；`bean_000` 预置勺头，其余 14 豆仍在碗内，不要求一勺装下 15 豆 |
| `empty` | 空载控制/F/T 诊断；15 豆在 reset 放到工作区外互不重叠的位置，运行中保持独立物理运动 |

正常初始布局采用两层 **8＋7**：各层圆环半径暂取 22 mm，中心高度距 `bowl_frame` 底面为 7.5 / 22 mm。seed 决定整体方位和逐豆随机单位四元数；初始速度为零。半径 7 mm 的包围球用于出生间隙筛选，要求豆间、底面与真实有限侧壁至少 0.5 mm 间隙；再核验实际碰撞初态无穿透。该布局是低成本生成规则，编译核验失败时修正布局。

勺上预置依据真实勺头碰撞面、Bean 姿态的支撑半径和 0.5 mm 间隙确定。只允许 reset 放置，不允许 episode 内将碗中豆子直接搬到勺上。

reset 保留现有清理外力、命令、参考、故障和接触统计的顺序。先让机器人空载稳定，再放置 Bean 并自然沉降；沉降也走同一物理入口。达到规定稳定窗口后清理回合统计并重置时钟，**不修改沉降得到的粒子速度**。失败显式返回诊断，不以继续等待到无限时长解决。

### 6.3 状态和快照

M1 snapshot 提供 `bean_ids[15]`、`bean_positions[15,3]`、`bean_quaternions[15,4]`、世界坐标线/角速度 `[15,3]`；逐豆质量、接触与诊断承托状态进入诊断信息。四元数顺序为 wxyz，单位沿用米、秒、千克。

同步适配 M1 StateProvider 的食物字段，清除对单个 `food_position` 的依赖。后续策略观测空间在 M3 单独定义；本轮不通过保留单粒子占位字段假装 M3/M4 已兼容。

复用完整 `mjSTATE_INTEGRATION` 保存/恢复和 adapter/monitor 状态机制；变更快照 schema/model signature，拒绝旧单块食物快照。新增碗 XML/mesh、豆 mesh、配置与判据源码到 `asset_files()` 和输入哈希。

## 7. M1 的逐豆物理证据

M1 使用小型、只读的逐豆诊断判据，避免调用仍依赖 `food_box`、`plate_frame` 的旧 M3 `evidence()`。M3 多粒子任务事件稍后重建，不在本轮同时改完整奖励、阶段机或教师。

**承托：**必须识别具体 Bean 与真实碗底/勺头的承载接触、向上的接触力及几何承托位置。勺柄/勺颈碰撞不算勺头承载。堆叠豆允许通过豆—豆承载链获得支撑，不能要求每豆直接触底或直接触勺。

**离碗和携带：**扫取豆必须离开碗支撑，随真实勺子抬升，并在规定观察窗口内留在勺头承托范围；不能仅凭勺子附近的距离认定拾取。

**掉落：**指定承载豆连续失去勺头承托且移出承托区域，才算掉落。记录豆 ID、发生时间和轨迹；其他豆留在碗里属于正常现象。

**边界：**结合接触距离与椭球几何范围判断碗底、侧壁及碗口包含关系。壁面投影仅用于候选筛选，有限边缘以 mj_geomDistance 确认穿透，同时保留中心内侧、碗底、碗口和外径检查，避免将侧壁无限平面延伸后误判；不把单个接触距离当作完整几何穿透证明，也不重建通用 SDF 几何诊断系统。

**F/T：**补偿仅覆盖固定勺子子树的自重与惯性。Beans 是自由 body，不加入 `tool_bodies`；勺子承载豆的重量应保留为可测外部载荷。保留现有接触保护按语义 pair 求和的机制，多个豆同时接触不能绕过 5 N 接触阈值和 8 N 腕力阈值。

## 8. 实施顺序与验收

下列新增数值门槛是本方案提出的工程标准，需要在首次原生运行前写入验收配置；不是已有实测结论。控制/F/T/保护门槛继续使用 [现有 acceptance.json](../configs/acceptance.json)。

| 阶段 | 实施内容 | 验证与完成条件 |
| --- | --- | --- |
| M1-A 资源与编译 | 制作共享 Bean mesh；装配碗与 15 body；筛选源 pair；建立索引与哈希 | Panda/UR5e 均能编译、forward、短步进；15 个 freejoint、15 个 ellipsoid、15 个 visual 且共享一份 mesh；同豆两 geom 同 body；质量惯量与坐标正确；17 个碗 collision 和 145 个勺—碗 pair 全部存在，无插件及悬空引用 |
| M1-B 原生接触与沉降 | 单豆/双豆接触检查；15 豆正常 reset；viewer 对齐检查 | 固定 seeds 0–9，每组在最多 5 s 仿真内达到全体共同低速窗口：线速度 <1 mm/s、角速度 <0.1 rad/s，连续 0.5 s；全体留在碗内；所有 Bean 接触/几何穿透不超过 0.4 mm；无非有限状态和 MuJoCo warning |
| M1-C 控制与取餐诊断 | 复用空载保持、六轴跟踪、reset、外力/F/T、可达性、故障保护；单豆勺上承载；固定低速扫取、抬升、倾斜/加速掉落 | 原控制门槛通过；100 次同 seed reset 可复现且清理执行状态；单豆静态承载 2 s 和温和携带通过；真实接触获得载荷；固定 seeds 0/1/2 的简单扫取各至少 1 豆离碗并在抬升后稳定承托 0.5 s；指定掉落豆出现真实脱离证据；初态勺上测试单独标记，不计为扫取成功 |
| M1-D 双机器人冻结 | 完整新 M1 报告、步长/迭代对照、viewer、性能记录；适配文档与 manifest | 两机器人全部必需用例通过；新配置/资产/源码哈希冻结；未跑项目标为 not_verified；旧报告不补齐新场景成绩；M3/M4 保持未验证 |

补充验收要求：

1. **视觉对齐。**分别显示 visual、collision 和叠加视图，核对每颗豆、碗内表面、勺头和 TCP。用编译后的 mesh 顶点及变换验证外形，不能仅比较可能受 mesh 编译重定心影响的 geom 原点。
2. **碰撞有效。**代表性物理用例确认豆—豆、豆—碗、豆—勺头接触真实产生；视觉 geom 从不进入接触。初态不应产生豆—桌面穿透或机器人意外碰撞。
3. **可达性。**沿统一控制链实际执行碗口上方、扫取起点、抬升净空位与嘴前等待位。碗比旧盘子更深，路径和净空必须重新计算，不能只改 `plate_frame` 的名字。
4. **数值对照。**使用 1 ms / 100 iterations、0.5 ms / 100 iterations、1 ms / 200 iterations 并收紧 solver tolerance。匹配布局/初态与时间戳。本轮对照仅覆盖 M1-B 单豆、双豆、勺头承托与 seeds 0–9 沉降，恢复完整 integration 初态及驱动时钟，以 10 ms 公共网格记录逐豆位置分歧和沉降时间差；后续控制、承载/掉落及力/冲量对照仍须完成。控制轨迹、力及冲量沿用旧控制容差；单豆承载/掉落结局一致；15 豆比较包含、穿透、稳定性和任务证据，记录逐豆分歧，不要求多接触粒子最终排列逐位相同。
5. **性能。**记录加载、物理步、IK、forward 和日志耗时，分别报告纯物理与完整控制循环的 RTF。目标为接近实时；首轮 headless 沉降验收墙钟预算设为每组 60 s（最多 5 s 仿真），超时算失败并定位瓶颈。关闭日志或 viewer 的加速必须保持相同物理结果，不因性能问题降低粒子数。
6. **回归。**更新与此次装配直接相关的控制/餐具测试，再新增共享 mesh、自由 DOF、逐豆 reset、碰撞、承托与快照测试。旧 M3/M4 单块食物测试不能证明新场景兼容；后续重建相应接口再执行全量任务验收。

加速度掉落可复用现有独立压力诊断配置，报告实际速度、加速度和使用的增益；这些设置不进入正常运行。M1 的低速扫取是接触场景验证，不等于 M4 完整喂餐教师成功率放行。

## 9. 文件修改范围与输出

| 文件/目录 | 必要改动 |
| --- | --- |
| `assets/task/foods/beans/meshes/bean_visual.obj` | 新增一份共享光滑视觉 mesh |
| `assets/task/scene.xml` | 移除单块 food；保留公共场景，由加载器装配 15 个 Bean body |
| `configs/scene.json` | bowl 安装、Bean 数量/半轴/密度/接触参数/reset 参数替换旧 plate/单食物配置 |
| `src/feedingrobot/sim/model.py` | 碗与 Bean 装配、逐豆索引、接触分组和运行资产哈希 |
| `src/feedingrobot/sim/task.py` | 逐豆 reset、质量/摩擦、snapshot/StateProvider、快照版本 |
| `src/feedingrobot/control/adapter.py` | 仅更新环境障碍物分组；核验新增自由 DOF，不重写 IK 和伺服 |
| `src/feedingrobot/sim/contacts.py` | 按需增加 Bean ID 映射；复用力读取与保护聚合 |
| `src/feedingrobot/scripts/validate_m1.py`、`configs/acceptance.json` | 原生 Beans 必需用例及逐豆判据、数值/性能门槛、新版本报告 |
| `src/feedingrobot/scripts/demo.py`、`doctor.py` | 适配新的 reset 与装配自检，保留入口用途 |
| `tests/test_contracts.py`、`test_tableware.py` 与新的 Bean 原生测试 | 替换“无碗/单 food”断言，验证实际控制与接触行为 |
| `docs/interfaces.md`、`assets/task/tableware/README.md`、来源清单 | 在实施完成后更新当前坐标、字段、资产来源与哈希 |

源机器人和源勺子/碗 XML、OBJ 不做无关改造。若 Bean 诊断判据需要独立文件，只建立一份供 M1 入口复用的小模块；不恢复原 SDF 的阶段 A/B/C 工具链。

新输出统一为 `outputs/beans_native/v1/m1/<robot>/`，调试使用 `outputs/calibration/beans_native/`。报告至少保存：配置/源码/资产哈希、实际求解参数、逐豆索引、初末完整状态、逐豆轨迹、必要接触证据、失败 ID/时刻、各用例指标、性能计时和 viewer 截图。不再为每一步重复保存整份模型或密集引擎内部搜索日志。

以下为完整后续控制／验收迁移的目标入口，不构成本轮 M1-B 已通过的结论：

```bash
conda run -n feedingrobot python -m feedingrobot.scripts.doctor
conda run -n feedingrobot python -m feedingrobot.scripts.demo --robot panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 \
  --robot panda --output outputs/beans_native/v1/m1/panda
conda run -n feedingrobot python -m feedingrobot.scripts.validate_m1 \
  --robot ur5e --output outputs/beans_native/v1/m1/ur5e
```

## 10. 旧 Beans SDF 清理与版本边界

按此次要求，直接删除以下 SDF 专用内容，不归档、不保留回滚副本：

- `assets/task/foods/beans/bean_sdf.cc`。
- `tools/beans_*` 原型、标定、沉降、诊断、候选运行/验收、C++ 探针、引擎 trace header 和碰撞 patch，以及 `build_beans_plugin.py`、`build_beans_engine.py`。
- 六份旧 `tests/test_beans_*` 和仅服务于 SDF 构建的 `tests/conftest.py`，以及对应 Python 字节码缓存。
- `docs/beans_sdf_prototype_plan.md`、`beans_stage_a.md`、`beans_stage_b.md`、`beans_stage_c.md`。
- `build/beans_sdf/`、`build/beans_engine/`、`outputs/beans_prototype/` 全部生成内容，包括自编译引擎、原始轨迹、图像、冻结标定及报告。

保留现有餐具/机器人资源、控制和物理推进代码、M0 环境记录、与 SDF 无关的餐具历史证据及用户已有 `.gitignore` 修改。已有 v2 M1/M3 报告描述盘子与单块食物，不构成新 Beans 场景验收。

本文对“碗＋15 Beans”的场景范围优先于旧 [SimModelPlann.md](../SimModelPlann.md) 中“碗不启用、只使用盘子”的规定。后续顺序为：**原生 Beans M1-A → M1-B → M1-C → M1-D → 多粒子 M3 → 新教师 M4**。M1 完整通过后再适配粒子拾取、交付/掉落、奖励和教师；不因控制链可复用而沿用旧任务放行结论。

## M1-D 完整入口与冻结

`python -m feedingrobot.scripts.validate_m1 --robot all` 顺序运行双机器人完整验收。
保留 `--robot panda|ur5e`、`--cases`、`--output`；新增用例 `assembly contacts convergence viewer performance manifest`。
每机器人报告写入 `outputs/beans_native/v1/m1/<robot>/m1d/`，独立 A/B/C 只声明自身阶段。
控制基准执行 14 项，两组对照执行除 reset/guards 外全部 12 项。三组设置为 1 ms/100/1e-8、0.5 ms/100/1e-8、1 ms/200/1e-10。
每个对照从基准完整 reset 快照开始，核验非数值结构及配置相同，恢复控制器、传感器、monitor 和驱动时钟；公共 set_state 保持严格。
命令刷新 20 ms、轨迹采样 10 ms；公共网格和终态 TCP 位置/旋转门槛 2 mm/0.035 rad；力 max(0.05 N,20%)、冲量 max(0.005 N·s,20%)。峰值含 applied/boundary，冲量只计 applied。
各设置独立满足原判据；允许取豆 ID/排列变化，记录逐豆分歧、阶段与掉落时间差。
viewer 保存碗中 reset 与实际连续低速承托达到 0.5 s 的取豆时刻完整快照、visual/collision/overlay；两机器人验证打开、同步和退出。
最终输入包括双机器人资产、配置、源码、测试、文档、依赖锁及来源 manifest，排除所有 outputs 与 freeze_manifest 本身。
完整控制 RTF 仅记录。沉降每组同时满足最多 5 s 仿真时间、60 s 墙钟。
先执行 candidate 矩阵；通过后更新修订 2 状态为 frozen 并修正来源哈希，再对最终输入执行双机器人完整复验。失败恢复 candidate，保留失败证据且不发布清单。
只有双机器人 A/B/C/D 全部通过且最终哈希相同才发布统一 freeze_manifest.json，记录输入 SHA256、实际参数、环境及报告/证据哈希。
局部运行未选项 not_verified、返回非零；缺失 viewer、旧模型、哈希漂移、任一机器人失败均不能冻结；M3/M4 保持 not_verified。

M1-D 第一轮候选完整入口已实际顺序执行双机器人，退出码 1；A/B/C、viewer、性能及来源清单通过，两个机器人输入哈希前后相同。半步长及高精度取豆峰值对照超限，未冻结；证据保存在 `outputs/beans_native/v1/m1/m1d_candidate_display/<robot>/m1d/`。阶段切换计时已修正为全回合共同 20 ms 命令网格，10 ms 采样不变；共同命令时钟完整复验已结束，命令返回 1；正式报告位于 `outputs/beans_native/v1/m1/<robot>/m1d/`。两机器人 A/B/C、viewer、性能及来源清单通过，初态回放核验通过，运行前后及双机器人输入 SHA256 一致；M1-D 数值对照失败，参数保留 candidate，未生成 freeze_manifest.json。原修订 2 接触参数（solref 2 ms）及轨迹目标保持不变；2.5 ms 隔离试验不能通过 Panda seed 1 的 5 s 沉降门槛，不采用。掉落仍按首次确认即结束，终态按各自结束状态比较，时间差只记录。

本轮共同命令时钟 M1-D 对照失败项：Panda 半步长为 tilt、acceleration、reachability、三个取豆 seed，高精度为三个取豆 seed；UR5e 半步长为取豆 seeds 0/1，高精度为三个取豆 seed。各设置原运动判据均通过，数值峰值／冲量或 TCP 对照超限仍判失败。126 项相关测试通过，demo 双机器人及 doctor IK 冒烟通过。每机器人本轮 100 次 reset 最大状态差为 0；所有控制数值回放完整初态匹配。取豆稳定承托 viewer、visual/collision/overlay 与完整快照均已保存。候选失败，因此不进入 frozen 最终复验。
