# 当前装配范围：单豆喂餐原型

2026-10-03：统一加载器默认装配原勺、水平固定碗与 bean_000；源 XML/OBJ、145 个勺—碗 pair 和 0.035 kg 工具子树保持。豆使用同 body visual mesh／原生 ellipsoid。正式单豆验收与冻结见 [M1 范围](../../../docs/m1_beans_rebuild_plan.md)，M3/M4 尚未放行；以下成绩属于历史模型。
## 历史记录（下文仅适用于原版本）

# 碗、盘子和勺子独立模型

本目录从仓库 `extract/` 完整复制。当前统一加载器接入原勺子、水平固定碗与 15 个原生刚体 Beans，Panda 和 UR5e 均完成 M1-A 独立编译验收；运行不依赖提取目录。基础 XML/OBJ 保留原始内容，装配只处理自由关节、安装变换、默认参数/材质作用域、网格路径与新增 site。Beans M1-B 已实现逐豆 reset 与真实接触诊断，修订 2 双机器人沉降及 M1-C 控制／真实取豆验收已通过；M1-D 完整数值复验失败，参数未冻结，见 [M1 重建方案](../../../docs/m1_beans_rebuild_plan.md)。[旧验收说明](../../../docs/acceptance.md) 描述历史盘子／单块食物模型。

当前项目加载碗，盘子仅作为未启用的源资源保留；此次范围以 [Beans M1 重建方案](../../../docs/m1_beans_rebuild_plan.md) 为准。装配时仅保留 `contact_pairs.xml` 中引用勺子/碗有效 geom 的 145 个 pair，不能直接 include 含盘子引用的完整片段。下文的三个物体组合说明仅用于资源通用使用。

从 `scene_100.xml` 提取；已确认 `scene_300.xml` 中这三个物体的完整 body 定义相同。

- `bowl/bowl.xml`：碗，1 个视觉网格，17 个基本形状碰撞体，质量 0.28 kg。
- `plate/plate.xml`：盘子，1 个视觉网格，17 个基本形状碰撞体，质量 0.24 kg。
- `spoon/spoon.xml`：勺子，1 个视觉网格，14 个勺柄碰撞网格、1 个勺颈碰撞体和 130 个勺头碰撞网格，质量 0.035 kg。
- `contact_pairs.xml`：原场景中勺子与碗、盘子的 290 个专用接触 pair；这是组合场景用的 include 片段，不能单独加载。

每个分类目录自包含，可单独复制使用，不依赖原目录。OBJ 文件逐字节复制，材质由 MJCF 的颜色定义，无外部纹理依赖。保留原始尺寸、质量、惯量、自由关节、内部坐标变换、碰撞掩码和物理参数。长度单位为米，质量单位为千克，角度单位为弧度，四元数顺序为 w x y z。

## 加载

```python
import mujoco
model = mujoco.MjModel.from_xml_path("assets/task/tableware/bowl/bowl.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
```

也可替换为 `assets/task/tableware/plate/plate.xml` 或 `assets/task/tableware/spoon/spoon.xml`。
独立模型不含桌面、地面或机器人；自由物体在重力下会下落。

## 接入机械臂场景

将对应 XML 的 asset 子元素和 worldbody 下的物体 body 合并到项目 MJCF，并将 mesh 的 file 路径改为项目实际资源路径。独立模型中的 option/default 来自原场景；合并时核对项目默认参数，避免直接覆盖项目设置。保留碰撞体的 contype/conaffinity 掩码，并检查机器人碰撞掩码是否匹配（勺柄 conaffinity=16，勺头及碗盘 conaffinity=7）。

同时引入三个物体时，在主 MJCF 根节点中添加 `<include file="assets/task/tableware/contact_pairs.xml"/>`，路径按主文件位置调整。仅引入部分物体时，只保留引用已存在 geom 的 pair。

如需把勺子刚性安装到末端执行器，把其 body 放到末端 body 下，并移除 `dynamic_spoon2_freejoint`；设置安装位置和姿态。碗盘若固定在桌面上，同样移除对应 freejoint。

视觉 geom 为 group=1，碰撞 geom 为 group=3；显示时可隐藏 group=3。

## 初始位姿

提取后的顶层 body 初始位姿统一为 `pos="0 0 0" quat="1 0 0 0"`，内部变换未改动。原场景顶层位姿如下，可按需要恢复：

| 模型 | pos（米） | quat（w x y z） |
| --- | --- | --- |
| bowl | `-1.802955 0.249493 1.10665205875` | `1 0 0 0` |
| plate | `-1.81185954664 0.08861149868 1.10083469296` | `0.996082544327 0 0 -0.0884283035994` |
| spoon | `-1.84433362125 0.07029811915 1.11926794785` | `0.650474131107 -0.0370082072914 0.0317600481212 -0.757961153984` |

## 当前 M1 装配坐标

源勺子的局部 +x 指向勺头，+z 为承载面法向；TCP 在源勺子坐标 `[0.04, 0, -0.0062854]` m，依据碰撞网格内表面射线确定。勺头有效几何为 `spoon_scoop_part` 下的 130 个凸网格；勺柄/勺颈的 15 个碰撞体不算食物承载或嘴部净空。

`tool_mount` 仅为无质量、无碰撞的刚性连接和 F/T 坐标。勺子源 body 在连接坐标中的候选位置为 `[0.06, 0, -0.004]` m；较原安装点沿柄尾移动 10 mm，为低位正向入豆预留手腕间隙。此处采用仿真中的刚性点安装，未验证实体夹具握持；Panda／UR5e 法兰变换与复位关节值分别保存在机器人配置。传感器下游唯一原始质量为 0.035 kg，完整子树惯量参与补偿。

碗固定在桌面，源内部位移和旋转不改动，外层姿态抵消内部旋转使底盘水平。`bowl_frame` 位于底盘圆柱上表面，世界坐标 `[0.45, -0.18, -0.0156]` m；桌面顶面 −0.02 m，底盘厚度 4.4 mm。外层安装位置约为 `[0.448315341173, -0.180072230707, -0.013997757508]` m，四元数约为 `[0.999987763364, -0.000318679324, 0.004893989854, 0.000648405647]`。

15 个 Beans 共用项目生成的 14×9×8 mm 椭球视觉网格，每个 body 内另设原生 ellipsoid collision 和显式质量惯量。M1-A 的两层确定性布局仅供编译冒烟；正常 reset 已迁移并有真实接触证据，修订 2 已通过 M1-B 沉降和 M1-C 真实取豆门槛。M1-A 不通过旧单块食物判据或初态布局宣称承载成功。

## 来源与许可证

本目录餐具由用户确认来自 [EBiM Benchmark](https://github.com/EBiM-Benchmark/benchmark) 的 `task3_mujoco/`，从 `scene_100.xml` 提取，并核对 `scene_300.xml` 的物体定义。上游 Task 3 源于 Ahmed Shokry 的 `Mujoco_Genisis_Model`；EBiM 的 Task 3 README 与 NOTICE 明确记录，该工作经作者同意以 Apache-2.0 贡献。

随仓库保留上游完整 [LICENSE](../../../docs/licenses/ebim-benchmark/LICENSE)、[NOTICE](../../../docs/licenses/ebim-benchmark/NOTICE) 和 [Task 3 来源说明](../../../docs/licenses/ebim-benchmark/task3_mujoco_README.md)。NOTICE 中其他任务的声明属于上游完整记录，不表示本项目使用了这些任务。

许可文件核查版本为 EBiM commit `161db49e9eafd34be31e5bdada72d8e2f796c16a`；这不是已确认的资产提取版本，实际提取 commit 尚未核实。不能用机器人的 Menagerie 许可证替代餐具许可证。

本项目的修改包括：将物体与接触 pair 提取为独立 MJCF，调整网格相对路径，统一顶层初始位姿；运行装配时移除自由关节、调整安装变换、限定默认参数与材质作用域、添加 site 并筛选有效接触 pair。OBJ 文件保持原始内容。基础资源及运行装配配置 SHA256 见根目录 `third_party_manifest.json` 和正式 M1 报告。

M1-D 第一轮候选完整入口已实际顺序执行双机器人，退出码 1；A/B/C、viewer、性能及来源清单通过，两个机器人输入哈希前后相同。半步长及高精度取豆峰值对照超限，未冻结；证据保存在 `outputs/beans_native/v1/m1/m1d_candidate_display/<robot>/m1d/`。阶段切换计时已修正为全回合共同 20 ms 命令网格，10 ms 采样不变；共同命令时钟完整复验已结束，命令返回 1；正式报告位于 `outputs/beans_native/v1/m1/<robot>/m1d/`。两机器人 A/B/C、viewer、性能及来源清单通过，初态回放核验通过，运行前后及双机器人输入 SHA256 一致；M1-D 数值对照失败，参数保留 candidate，未生成 freeze_manifest.json。原修订 2 接触参数（solref 2 ms）及轨迹目标保持不变；2.5 ms 隔离试验不能通过 Panda seed 1 的 5 s 沉降门槛，不采用。掉落仍按首次确认即结束，终态按各自结束状态比较，时间差只记录。
