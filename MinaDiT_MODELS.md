# MinaDiT 模型介绍

本文介绍本项目的三版动作扩散 Transformer：**MinaDiTv1**、**MinaDiTV2** 与后续训练默认使用的 **MinaDiTV3**。名称大小写按此约定使用。

MinaDiTV3 沿用 V2 网络结构，取消条件输入及动作标签归一化，并将推理改为 FP32；训练仍沿用原 BF16 自动混合精度策略。旧模型及其历史运行环境保留。正式训练和完整闭环验收由用户后续执行；历史 V2 工程验证结果见文末。

## 1. 三版模型对比

| 项目 | MinaDiTv1 | MinaDiTV2 | MinaDiTV3：默认模型 |
| --- | --- | --- | --- |
| Transformer Block 数 | 12 | 6 | 6 |
| Hidden size | 512 | 512 | 512 |
| Self-Attention heads | 8 | 8 | 8 |
| 动作序列 Self-Attention | 双向，合法动作之间相互可见 | 因果，第 i 个动作只关注第 0 至 i 个动作 | 与 V2 相同 |
| 条件 Cross-Attention | 8 heads，关注当前与历史观测条件 | 保留相同条件接口与机制 | 与 V2 相同 |
| MLP hidden size | 2048 | 2048 | 2048 |
| 残差调制 | adaLN-Zero | adaLN-Zero | adaLN-Zero |
| 动作输入 Embedding | 可学习 Linear：6 → 512 | Zero Padding：6 → 512 | Zero Padding：6 → 512 |
| 连续观测、历史与派生速度 | 现有 Z-score 归一化 | train-only Min-Max，训练范围映射到 0–1 | 原始物理尺度，无归一化 |
| 干净动作标签 | 现有 Z-score 归一化 | train-only Min-Max，训练范围映射到 0–1 | 原始 TCP twist，m/s、rad/s |
| 附加帧龄特征 | 保留当前秒数表示 | 单独按有效 train 特征统计并归一化 | 原始秒数 |
| 类别、one-hot、布尔值与 mask | 现有表示 | 保留原值与类型 | 保留原值与类型 |
| 预测 Horizon | H=16 | H=16 | H=16 |
| 输出 | 6 维噪声预测 epsilon | 6 维噪声预测 epsilon | 6 维噪声预测 epsilon |
| 推理精度 | 原 BF16／FP32 自动策略 | 原 BF16／FP32 自动策略 | FP32，关闭 BF16 autocast 和 TF32 |
| 参数量 | 80,406,374 | 41,007,462 | 41,007,462 |
| 实现状态 | 已有实现及历史权重 | 已实现；正式训练与闭环验收待执行 | 正式训练与闭环验收待执行 |

V1/V2 参数量已按实际模型核验，V3 保留 V2 结构。减少约 49% 的参数不等于推理耗时减少 49%，实际速度与显存需实测。输入 Linear 仅有 3,584 个参数，主要减负来自层数减少；V3 的 FP32 推理耗时和显存也需单独实测。

## 2. 共同的数据与执行接口

三版模型均使用 Panda 原有观测 schema 和 6 维基座坐标系 TCP twist，三个平移分量单位为 m/s，三个旋转分量单位为 rad/s。

- 输入条件包括两帧低维状态、最近 0.2 秒的十帧历史，以及阶段和交互信息。当前 Panda 接口为 `states[B,2,122]`、`history[B,10,28]`，编码后产生一个全局条件和十三个条件 token。
- 观测条件仅由当前及过去可用观测构造。MinaDiTv1 的双向动作注意力不表示读取未来真实观测；它联合处理正在去噪的预测动作块。
- 每次预测十六个动作，20 Hz 下覆盖 0.8 秒；通常每 0.2 秒重新规划，只执行前四步，阶段变化取消旧动作队列。
- 保留 epsilon 预测目标、100 步训练加噪和十步确定性 DDIM。现有评估协议中的 `leading` / `trailing` timestep spacing 差异需显式保存在运行记录中，各版对比采用同一协议。
- 合法动作窗口仍在阶段变化、无效标签或时间不连续处截断。时间维度的 padding 与动作通道补零是两件不同的事；有效 mask 继续用于注意力和损失。
- V1/V2 采样结束后按对应模型的归一化器恢复物理动作单位；V3 采样结果直接使用物理单位。随后进入既有速度、加速度、IK、力和工作空间保护执行链。

## 3. MinaDiTV2 / MinaDiTV3 的动作处理过程

```text
原始六维动作标签
  → V2 使用训练集统计做 Min-Max；V3 保留 m/s、rad/s
  → 得到干净动作 x0 [B,16,6]
  → 在六维动作空间加入扩散噪声，得到 xt [B,16,6]
  → 每个动作保留前六个通道，右侧补 506 个零
  → 加入现有位置编码，得到 [B,16,512]
  → 六个 DiT Block：因果 Self-Attention + 条件 Cross-Attention + MLP
  → 保留最终调制、LayerNorm 和 Linear(512,6)
  → 输出 epsilon [B,16,6]
```

Zero Padding 只替换动作输入 Linear，不替换条件编码器、注意力中的 Q/K/V 投影或动作输出头。先加噪再补零，补零通道不单独加入扩散噪声。

因果 mask 沿动作时间轴定义，与合法动作 padding mask 同时生效；条件 Cross-Attention 保留全部有效的当前和历史条件。每轮仍并行去噪整个动作块，不改成逐动作自回归生成。

## 4. V2 的 0–1 归一化约定

范围包括连续观测、历史特征、十二维派生目标速度、帧龄和干净六维动作标签。历史中的观测字段复用对应状态字段的统计，派生速度在原始物理表示上计算后再归一化。

对每个连续通道使用：

```text
offset = train_min
scale  = train_max - train_min
x_norm = (x - offset) / scale
x      = x_norm * scale + offset
```

常量训练通道使用 `scale=1`，其训练值映射为零，避免除零。阶段类别索引、one-hot、交互布尔值、执行状态和有效 mask 保留原表示；无效行归零，padding 和 mask 不参与连续统计。

V2 只统计实际参与训练的有效条件与监督标签，其配置仅包括旧 M5 V1 使用的 M4 正常／恢复 train 窗口；不包含 v4 纠偏池。validation/test 不参与统计。V2 归一化器明确记录 `kind=minmax`、版本、统计范围和来源绑定，单独保存到新训练目录及 checkpoint；不把 min/range 伪装成 mean/std，也不覆盖冻结数据的原 `normalization.json`。

**0–1 约束针对干净训练数据。** 扩散加噪后的动作、Gaussian 噪声、epsilon 和网络中间特征不受此约束，不做 0–1 裁剪，不在噪声输出头加 Sigmoid。

方案默认对 validation 和线上观测沿用同一仿射变换，允许超出训练范围，并记录越界比例。若要求线上观测始终严格落在 0–1，需另行选择截断策略；截断会丢失超过示教范围的力或位姿误差幅度，且不适用于扩散噪声。

### V3：原始输入与 FP32 推理

V3 取消上述数据预处理归一化：连续状态、历史、派生目标速度和帧龄均保留原始尺度，六维动作标签直接使用 m/s、rad/s。类别、one-hot、布尔值、有效 mask 和无效行处理规则保留；网络内部 LayerNorm 与 adaLN-Zero 保留。V3 不拟合 Min-Max 或 Z-score 统计，但仍保存 `kind=none`、`schema_version=3` 的身份变换记录及训练来源绑定，便于 checkpoint 审计和恢复。

V3 推理固定使用 FP32，关闭 BF16 autocast，并在推理作用域内关闭矩阵乘法和卷积的 TF32，结束后恢复调用方原设置。训练自动混合精度策略与 V2 相同。改变输入和动作尺度后必须从头训练 V3，不能把已有 V2 checkpoint 改名当作 V3。

## 5. 模型选择与旧模型保留方案

训练、工程验证和评估入口已支持模型选择。

- 训练及工程验证提供 `--model MinaDiTv1` / `--model MinaDiTV2` / `--model MinaDiTV3`。新训练未指定模型时默认 V3；旧配置没有模型名称时按 MinaDiTv1 解释。
- 保留 `configs/dp_dit.json` 的 V1 语义，V2 使用 `configs/minadit_v2.json`，V3 使用 `configs/minadit_v3.json`。新配置及测试路径显式加入父数据审计允许列表。显式模型选择与配置内容冲突时拒绝启动。
- 推理和评估从 checkpoint 读取模型名称、完整配置及对应归一化器；可显式指定模型名称进行一致性检查。不能交叉使用各版权重、归一化器或 optimizer。
- 保留现有 `ActionDiT` 兼容入口及 V1 参数键；V2/V3 通过统一模型构造函数加载。历史诊断工具继续明确使用 V1。
- 原有权重、输出目录、采集报告和数据保留。新的训练目录按模型名称区分，例如 `outputs/single_bean/v1/m5/dit/MinaDiTV3/run_001/`，不移动或覆盖历史目录。
- 新 checkpoint 记录精确模型名称、归一化模式、配置和训练来源；恢复仅允许相同模型、配置、归一化器和数据绑定。

当前旧 checkpoint 审计绑定全仓 `source_hashes`。新增代码或配置后，保留相同参数结构仍不足以保证旧 checkpoint 可加载。已保存并核验与旧 checkpoint 逐文件匹配的运行环境 `outputs/single_bean/v1/m5/dit/MinaDiTv1/runtime_001/`；历史 V1 评估和续训由该环境承接，模型选择入口需明确路由并记录实际运行版本。原 schema 1 入口自动路由；旧纠偏模型通过 `feedingrobot.policies.legacy` 启动原专用入口，具体命令见训练指南。源码哈希检查继续保留。

现有 v4 训练入口要求精确继承旧 100k EMA、旧归一化器及其零更新审计，不能直接套用到 V2/V3。V2/V3 应采用各自新的训练与模型审计记录，继续复用已核验的采集数据证据。新旧对比各自加载完整的模型、配置和归一化器，再使用相同物理初态、随机种子与评估协议。

## 6. 已采轨迹的复用

网络变化不改变原始轨迹格式，因此无需因本次重构重新采集。

| 数据 | 可复用方式 |
| --- | --- |
| M4 正常与恢复数据，共 260 回合 | 保留正常和恢复各 100/15/15 的 train/validation/test 划分；训练及统计只使用合法 train 数据 |
| v4 合并纠偏数据，共 44 条 | 保留教师 ownership、alignment/transition/aligned 采样池及完整 pickup_hold 标签边界 |
| v3 的 12 条数据 | 已包含在 v4 的 44 条中，使用 v4 时不重复加入 |
| 原归一化文件 | V1 沿用；V2 从已有原始训练数据另拟合新统计；V3 使用原始数据和身份变换记录，不覆盖原文件 |

复用数据不保证 V2/V3 泛化或闭环成功。因果注意力、较浅主干、Zero Padding 和原始输入尺度的实际收益需要验证，不能由网络尺寸直接推断。

## 7. 实施完成的判据

1. 模型选择正确：未命名旧配置选择 V1，显式名称冲突被拒绝；旧权重在匹配环境中通过原审计。
2. V1 行为保留：参数键、归一化及固定输入输出与旧基准一致，历史 checkpoint 和输出文件不变。
3. V2/V3 结构正确：六个 Block、6 → 512 通道补零、六维噪声输出；在非零注意力门控下，扰动未来动作不改变较早位置的噪声预测。
4. 数据尺度正确：V2 仅 train 拟合，状态与历史字段一致，动作正反变换可往返；V3 条件和动作保持原始尺度；无效行及短动作窗口结果有限，噪声保持未裁剪。
5. 训练与恢复可用：有限步训练、EMA、采样和同版本 checkpoint 恢复通过；跨模型或跨归一化器续训被拒绝。
6. V3 推理精度正确：外层 BF16 autocast 不影响 FP32 推理，推理期间 TF32 关闭，结束后恢复调用方设置。
7. 评估可比较：各版分别加载各自权重与归一化器，在相同 validation 场景和采样协议下报告拾豆、交付、恢复、动作跳变、耗时与显存；独立 test 仍留给最终验收。

## 8. 当前实现参考

- [DiT 实现](src/feedingrobot/policies/dit.py)
- [V3 默认配置](configs/minadit_v3.json)
- [V2 配置](configs/minadit_v2.json)
- [V1 配置](configs/dp_dit.json)
- [特征、动作窗口及归一化](src/feedingrobot/policies/data.py)
- [checkpoint 与父数据审计](src/feedingrobot/policies/audit.py)
- [M5 训练指南](docs/m5_training.md)
- [v4 纠偏训练](tools/m5/correction_v4/training/README.md)
- [v4 评估](tools/m5/correction_v4/evaluation/README.md)

## 9. 历史工程验证与正式训练

此前混合数据配置的更名源码通过 81 项回归测试。在 RTX 5080 上以完整 512 隐藏维度、batch=64、2 个 DataLoader workers 和 BF16 完成 2 步训练，再从 checkpoint 续训至第 4 步。GPU 因果独立性、EMA 的 DDIM 采样、单批 validation 前向和 250 ms 物理执行接入均通过；旧 145k V1 EMA 严格加载与前向通过。该短验证当时使用 M4/v4 混合数据，共 143,791 个合法窗口。用户随后指定首次训练只用正常／恢复数据，当前默认配置已移除纠偏来源，改为 113,136 个 M4 train 窗口；旧混合验证产物不能续训为此次首次训练。

这些历史 V2 检查只证明当时的训练链路可运行，不证明收敛或闭环成功，也不能替代 V3 验证。正式训练、完整 validation/test 和 DP_v1 冻结尚未执行。详情见 [V2 工程验证报告](outputs/single_bean/v1/m5/dit/MinaDiTV2/pipeline_smoke_001/refactor_report.json)。

V3 已通过 70 项模型与数据管线回归测试。在 RTX 5080 上使用完整 512 隐藏维度、batch=64 和 2 个 DataLoader workers 完成 2 步训练并续训至第 4 步；条件与动作原值一致、EMA 的 FP32 DDIM 采样、单批 validation 和 250 ms 物理执行接入均通过。外层 BF16 下仍观测到 FP32，并关闭推理中的 TF32。双样本采样预热后耗时约 25–26 ms，仅为本次硬件上的短测量，不代表闭环成功或实时验收。检查点与报告另存于 [V3 验证目录](outputs/single_bean/v1/m5/dit/MinaDiTV3/pipeline_smoke_001/verification_report.json)，该检查点只有 4 步更新；V3 正式训练与完整闭环验收尚未执行。

在项目根目录启动默认 V3 正式训练：

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp --device cuda --headless --output outputs/single_bean/v1/m5/dit/MinaDiTV3/run_001
```

默认 100,000 updates。续训、评估与旧模型使用命令见 [M5 训练指南](docs/m5_training.md)。
