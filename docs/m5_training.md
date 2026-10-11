# M5：Panda MinaDiTV4／V5 训练与验收

当前默认模型为 **MinaDiTV5**，保留 MinaDiTV4 作为动作归一化对照。
V4 之前的模型入口、历史权重和烟测输出已清理；数据集与训练所需的冻结证据保留。
工程验证与数据预检不代表正式训练或闭环验收完成，达到 test 门槛才冻结 `DP_v1`。

## 环境、模型与数据

以下命令在 `/home/minashiki/FeedingRobot_DPRL` 执行，使用已有环境
`/home/minashiki/anaconda3/envs/feedingrobot`。训练与推理均采用 FP32 IEEE，关闭 BF16 autocast 和 TF32。
默认使用 CUDA；CUDA 不可用时明确退出，`--device cpu` 仅供显式诊断。
整个进程树默认限制为 6 个逻辑 CPU，最多允许 8 个，PyTorch threads=4、DataLoader workers=2。

V4／V5 共用 6 个 DiT Block、hidden=512、8 heads、MLP=2048、因果动作自注意力与动作通道补零。
H=16，在 20 Hz 下预测 0.8 秒，每 0.2 秒重新规划，只执行前 4 步。
主配置预测归一化干净动作 `sample/x0`，使用 cosine/100 加噪和 `clip_sample=True`：

| 模型 | 归一化 | DDIM 推理 | 配置 |
|---|---|---|---|
| MinaDiTV4 | 有效 train 动作逐维 limits；连续条件保持物理尺度 | leading / 10 | `configs/minadit_v4.json` |
| MinaDiTV5（默认） | 动作、连续观测、派生速度及帧龄逐维 DP limits | trailing / 100 | `configs/minadit_v5.json` |

V4 的 epsilon 对照使用 `configs/minadit_v4_epsilon.json`，与主候选分开训练。
两版均保留类别、one-hot、布尔值及 mask，使用 `pre_action` 观测语义，采样后反归一化为基座 TCP twist。
速度、加速度、IK、力及工作空间保护继续生效。V5 数值与拟合规则见
[归一化契约](minadit_v5_normalization.md)。

当前配置使用冻结 M4 Panda train 的 100 条正常与 100 条恢复回合，再加入
`datasets/single_bean/start_pose_v2/panda/train/` 的 200 条正常回合，共 400 条、338,555 个合法动作起点。
validation／test 使用原 M4 独立划分，不参与归一化拟合或训练。
阶段、回合和合法窗口依次均衡采样，padding 不进入损失。

启动时核验 `outputs/single_bean/v1/m4/revision_4/` 的冻结清单、审计、源文件快照及完整 M4 数据哈希，
并读取 `outputs/single_bean/start_pose_v2/collection_001/report.json` 与 `audit.json` 校验扩充数据。
这些文件是训练依赖；保留全部数据集，不改冻结数据、划分或原归一化文件，不重跑旧验收或重放。

## 工程检查与正式训练

先检查模型、数据管线与归一化：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest \
  tests/test_m5.py tests/test_minadit_v4.py tests/test_minadit_v5.py \
  tests/test_dp_normalization.py tests/test_minadit_v5_normalization.py \
  tests/test_diffusion_diagnostics.py tests/test_diffusion_data_audit.py -q
```

可选工程验证只在固定 8 个窗口上更新至多 200 次，并检查采样与短时物理执行链：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.validate_m5 \
  --model MinaDiTV5 --device cuda --headless \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV5/engineering_001
```

`diagnostic.pt` 仅供诊断，续训及最终冻结均拒绝该文件。
新运行目录必须不存在；启动正式 V5 训练：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp \
  --config configs/minadit_v5.json --model MinaDiTV5 --device cuda --headless \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001
```

省略 `--model` 与 `--config` 同样选择 V5。V4 训练显式传入对应配置与 `--model MinaDiTV4`，使用独立目录。
默认 AdamW lr=1e-4、weight decay=0.01、batch=64、EMA=0.9999、梯度 norm 上限 1，共 100,000 updates。
每 1,000 次记录离线 validation loss 与采样后的物理动作指标；每 5,000 次保存 checkpoint；
每 10,000 次暂停训练，在正常／恢复各 15 个 validation 场景执行完整闭环。
`best.pt` 按两类完整成功率的平均值选择，推理使用 EMA 权重。

主要输出为 `metrics.jsonl`、`hardware.json`、`data_summary.json`、`parent_audit.json`、
`train_extension_audit.json`、`normalization.json`、`last.pt`、`step_<N>.pt`、`best.pt` 与 `validation_<N>/`。
不传 `--headless` 时闭环评估可显示 viewer；不可用或开销超预算时关闭显示并记录原因。

## 续训与源码绑定

继续原目录，并加载该目录的 `last.pt`：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp \
  --device cuda --headless --updates 100000 \
  --resume outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001/last.pt \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001
```

`--updates` 是累计目标。checkpoint 保存模型、EMA、优化器、随机数、已消费的采样状态、
配置、归一化、源码哈希和数据绑定。续训直接复用绑定的归一化参数，不重新拟合。
V4／V5 的归一化器、训练目标与优化器不能混用；V5 从头初始化，不把 V4 权重改名续训。

源码绑定仍然严格：改变绑定源码、配置、机器人或数据后应开始新运行。
不能修改 checkpoint 中的 `source_hashes`、配置或审计字段来绕过检查。
本次清理改变了源码集合，清理前的 checkpoint 不能在当前源码下直接续训或评估。

## 闭环评估与冻结

先用 validation 检查候选模型，并使用尚不存在的输出目录：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.eval_dp \
  --checkpoint outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001/best.pt \
  --split validation --device cuda --headless \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001/manual_validation_001
```

闭环从对应 M4 完整初态运行实际模型，记录事件、失败原因、接触／腕力、动作跳变及推理耗时。
同步推理会暂停仿真，不能据此声称通过 M7 实时性门槛。选模完成后才使用 test：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.validate_m5 \
  --freeze --checkpoint outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001/best.pt \
  --device cuda --headless \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001/final_acceptance_001
```

正常／恢复各 15 个场景，完整成功各至少 12/15；pickup、delivery、retract 与 recovery 成功率至少 90%。
所有恢复场景均须进入 RECOVER，且完成真实 `RECOVER→WAIT_READY`。
全部通过才发布绑定 checkpoint、EMA、归一化、源码、M4 父证据及 test 报告的 `freeze_manifest.json`。
工程 loss 下降、烟测或成功视频不能替代该门禁。
