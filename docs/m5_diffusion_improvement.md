# M5 动作归一化与训练改进方案

2026-10-09。本文记录保留的 V4 动作尺度与预测目标对照，当前默认训练模型为 V5。
V4 之前的模型、权重和对应离线诊断输出已清理；本文件不代表正式训练或闭环验收完成。
V5 在相同架构与数据上增加连续条件的逐维拟合归一化，推理使用 trailing/100，
详见 [V5 归一化契约](minadit_v5_normalization.md) 和 [当前训练指南](m5_training.md)。

## V4 输入与模型契约

- 动作采用 DP 的逐维 limits 归一化：`normalized = (physical - offset) / scale`。
  正常通道 `offset=(max+min)/2`、`scale=(max-min)/2`，范围映射到 `[-1,1]`。
  范围小于 `1e-4` 的通道沿用官方做法：`offset=min`、`scale=1`，不放大小变化。
- 统计只拟合原训练集及 200 回合 start-pose 扩展的有效动作，排除 padding 和 held-out。
  400 个训练回合共 338,555 个合法动作起点；16,937 个 validation 合法动作也没有范围越界。
  全量 NumPy 往返最大误差为 `2.78e-17`；Torch FP32 的 dtype/device 与往返另有测试。
  拟合的六个半范围约为 `[0.04891, 0.04853, 0.04000, 0.40365, 0.37893, 0.49618]`。
- V4 只归一化动作，保留观测连续量的原尺度，以隔离动作尺度的作用。
  V5 对动作、连续观测、派生速度及帧龄拟合逐维 DP limits，类别及 mask 保留原表示。
- 输入时间语义改为 `observation_timing=pre_action`。命令日志恢复动作前执行状态，
  在内存副本上只替换 `execution_status`，不改冻结文件或物理字段。
  430 个 train/validation 回合的 1,229,115 条普通观测交叉核验零不一致。
  丢弃动作边界帧会造成部分观测滞后 10 ms，以及 244 条动作的阶段滞后，因此保留精确边界。
- 保持现有 6 层、512 hidden、8 heads、16 步 horizon 的 DiT，约 4101 万参数。
  默认模型为 `MinaDiTV5`；运行 V4 对照需显式指定对应配置和模型。
- 主候选直接预测归一化干净动作 `sample/x0`；对照预测 `epsilon`。
  两个候选统一使用 cosine/100、DDIM leading/10、`clip_sample=True`、训练与推理 FP32 IEEE。
  训练 target 与 scheduler 的 prediction_type 保持一致，不能直接复用旧 epsilon 检查点续训。
- 采样中的 clip 作用于归一化的预测干净动作；之后反归一化。
  逐维 clip 不保证三维速度范数满足控制上限，近常数通道也不由该 clip 强制固定；原控制限幅仍生效。

归一化与预测目标的依据：[DP 官方 normalizer](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/common/normalizer.py)、
[DP 论文](https://arxiv.org/html/2303.04137v5)、
[DP3 官方配置](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/master/3D-Diffusion-Policy/diffusion_policy_3d/config/dp3.yaml)。
leading/10 在本项目中已有离线支持；它改变初始噪声与训练分布的匹配，仍需闭环验证。

## 训练与验证顺序

1. 在独立目录从头训练两个候选，先各 20,000 updates；同一数据、seed、预算和架构，仅预测目标不同。
   两配置均为 batch 64、AdamW lr `1e-4`、weight decay `0.01`、梯度上限 1、EMA `0.9999`。
   这些超参数沿用原方案用于对照，尚未证明是最优设置。
2. 每 1,000 updates 同时记录去噪 loss 与完整采样后的物理动作指标。
   固定 16 个 validation 窗口：一半取回合首个合法窗口、一半按阶段抽样。
   单列回合起点、序列首动作、前四步、合法 horizon 的线／角向量 RMSE、零动作基线、超限率。
   指标使用独立 RNG，不改变训练随机状态；epsilon 与 x0 loss 数值不可直接横向比较。
3. 每 5,000 updates 留检查点，每 10,000 updates 做既有 validation 闭环评估。
   成功率相同的检查点再比较动作指标，不能只按低 loss 或首次生成的 `best.pt` 选模型。
4. 20k 后先用离线工具对两个候选检查首动作、多种子、条件打乱、10/20/50/100 步敏感性。
   初步工程门槛建议首动作线向量 RMSE ≤0.005 m/s、角向量 RMSE ≤0.05 rad/s，
   方向合理、超限率接近专家；这是待验证的项目门槛，不是论文通用标准。
   若起点仍差，先比较首窗口采样比例与 EMA 滞后，避免盲目延长训练。
5. 候选通过起点检查且闭环拾取改善后，再扩大至 100k 和多训练种子。
   沿用既有全任务成功率 ≥80%、阶段成功率 ≥90% 的最终门槛，最后才使用 test。
   本轮没有运行 test、冻结或发布模型。
数据中的教师 SELECT 还有一个约 1 ms 的无效标签控制前缀，恢复状态不改写这个物理前缀；
若新候选起点仍异常，应再做相同初态、相同物理前缀的条件对照。

## 可执行入口

在 `/home/minashiki/FeedingRobot_DPRL` 的 GPU 环境执行，两次运行使用不同新目录：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp \
  --config configs/minadit_v4_epsilon.json --model MinaDiTV4 \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV4/epsilon_pilot_001 \
  --device cuda --updates 20000 --headless

PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp \
  --config configs/minadit_v4.json --model MinaDiTV4 \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV4/sample_pilot_001 \
  --device cuda --updates 20000 --headless
```

训练会自行拟合并绑定统计，保留原始数据与冻结审计；V4／V5 及不同预测目标使用独立新目录。
源码哈希继续严格校验，不修改 checkpoint 的绑定字段以适配清理后的源码。
这些命令是训练入口，不代表已有训练权重或闭环验收结果。

验证命令：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest \
  tests/test_dp_normalization.py tests/test_minadit_v4.py \
  tests/test_diffusion_diagnostics.py tests/test_diffusion_data_audit.py \
  tests/test_minadit_v5.py tests/test_minadit_v5_normalization.py tests/test_m5.py -q
```
