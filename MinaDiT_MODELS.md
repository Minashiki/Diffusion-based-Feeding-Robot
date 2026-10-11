# MinaDiT 模型介绍

当前支持 **MinaDiTV4** 与默认使用的 **MinaDiTV5**。
V4 之前的模型入口、旧权重、专用工具及验证输出已清理；可复用数据集和训练所需的冻结证据保留。
模型结构与工程测试通过不代表正式训练或闭环验收通过。

## 模型与采样

| 项目 | MinaDiTV4 | MinaDiTV5（默认） |
|---|---|---|
| DiT Block／hidden／heads／MLP | 6／512／8／2048 | 相同 |
| 动作序列自注意力 | 因果 | 相同 |
| 动作输入 | 六维空间加噪，再补零到 512 维 | 相同 |
| 条件编码 | 两帧状态、10 帧 causal CNN 历史、阶段／交互 token | 相同 |
| Horizon | 16；20 Hz 预测 0.8 秒，每次执行前四步 | 相同 |
| 连续观测、派生速度及帧龄 | 原始物理尺度 | train-only DP limits |
| 干净动作 | train-only limits，范围映射至 `[-1,1]` | DP 官方 FP32 逐维 limits |
| 类别、one-hot、布尔值与 mask | 保留原表示 | 相同 |
| 主配置预测目标 | 归一化干净动作 `sample/x0` | 相同 |
| 训练加噪 | cosine／100 | 相同 |
| DDIM 推理 | leading／10 | trailing／100 |
| 训练与推理精度 | FP32 IEEE，关闭 BF16／TF32 | 相同 |
| 配置 | `configs/minadit_v4.json` | `configs/minadit_v5.json` |

V4 另有 `configs/minadit_v4_epsilon.json`，仅改变预测目标供独立对照。
共同结构约 4,101 万参数，每个 Block 包含自注意力、条件交叉注意力、MLP 与 adaLN-Zero 残差调制。
实现依赖为 `MinaDiTV5 → MinaDiTV4 → ActionDiT`；公共底座及迁移后的因果 attention 保持原状态键与初始化顺序。

两版均采用 `pre_action` 观测语义，从命令日志恢复动作前的执行状态，只修改内存副本。
采样结束后反归一化为基座 TCP twist（m/s、rad/s），再交给现有控制限幅和保护。
`clip_sample=True` 裁剪归一化干净动作，不保证线／角速度向量范数满足控制上限。
V5 的统计、FP32 运算顺序、近常值规则及参数契约见 [归一化说明](docs/minadit_v5_normalization.md)。

## 数据与运行绑定

当前训练使用冻结 M4 Panda 的 100 条正常与 100 条恢复 train 回合，
加上 start_pose_v2 的 200 条正常 train 回合，共 400 条、338,555 个合法动作起点。
训练按阶段、回合、合法窗口依次均衡采样；padding、无效动作与 held-out 数据不参与拟合或损失。
validation／test 继续使用原 M4 独立划分。历史纠偏数据保留供后续复用，当前配置不加入这些数据。

保留完整 `datasets/`、M4 修订 4 的父冻结证据及 start_pose_v2 的采集报告／审计。
训练会另存归一化记录，不改冻结数据文件、原归一化器或旧审计内容。
checkpoint 严格绑定源码、配置、数据、归一化、模型、EMA、优化器与随机状态。
本次清理改变了源码集合；旧 checkpoint 不能通过改写 `source_hashes` 或模型名称绕过绑定检查。
V5 首次训练从头初始化，同一运行续训复用原 checkpoint 的拟合参数。

## 当前入口

省略 `--model`／`--config` 的训练和工程验证默认选择 V5；V4 必须显式选择。
各模型使用独立新目录，续训使用原目录的 `last.pt`。闭环评估加载对应 checkpoint 的 EMA 权重。
通过 validation 选模后才使用 test，达到完整成功率 ≥80%、阶段成功率 ≥90% 才冻结 `DP_v1`。

命令与资源预算见 [M5 训练指南](docs/m5_training.md)。
V4 的动作尺度与预测目标对照见 [训练改进说明](docs/m5_diffusion_improvement.md)。
