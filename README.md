# FeedingRobot DPRL

基于 **MuJoCo** 的机器人喂餐仿真研究项目，覆盖舀取、携带、送到嘴前、释放与撤离的完整任务流程，并提供教师示范、数据采集和扩散策略训练工具。

当前场景由固定碗、勺子和一颗豆子组成，以 **Franka Panda** 为主要开发与训练平台，同时保留 **UR5e** 的仿真支持。

## 核心功能

- **物理仿真与控制**：MuJoCo 场景、Mink 逆运动学、接触检测与力保护。
- **任务环境**：Gymnasium 接口、任务阶段、奖励与状态快照。
- **示范与数据**：完整喂餐教师、正常与恢复示范采集、轨迹重放。
- **扩散策略**：默认使用 MinaDiTV5，保留 MinaDiTV4 对照，支持训练和闭环评估。

## 快速开始

推荐使用 Linux x86-64 与 Conda。项目使用 Python 3.12，完整环境包含 CUDA 13.0 版 PyTorch；GPU 训练需要兼容的 NVIDIA 驱动。

克隆仓库后，在仓库根目录执行：

```bash
conda env create -f environment.yml
conda activate feedingrobot
```

运行 Panda 的完整教师演示，查看从舀取到交付、撤离的过程：

```bash
python -m feedingrobot.scripts.demo_m4 --robot panda
```

可视化演示需要桌面显示环境。在服务器或无界面环境中运行：

```bash
python -m feedingrobot.scripts.demo_m4 --robot panda --headless
```

演示会在 `outputs/calibration/m4/` 下保存轨迹与结果。策略训练所需的数据和冻结证据见训练指南。

## 项目结构

- `src/feedingrobot/`：仿真、控制、任务环境、教师与策略实现。
- `configs/`：机器人、场景、任务和模型配置。
- `assets/`：机器人与餐具模型。
- `docs/`：接口、训练和验收文档。
- `datasets/`、`outputs/`：示范数据与运行结果。

## 进一步阅读

- [MinaDiT 模型介绍](MinaDiT_MODELS.md)
- [策略训练与评估](docs/m5_training.md)
- [任务环境接口](docs/m3_interfaces.md)
- [教师与数据管线](docs/m4_interfaces.md)

第三方资产与依赖的来源、修改记录及许可证见 [来源清单](third_party_manifest.json) 和 [许可证目录](docs/licenses/)。
