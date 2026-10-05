# M5：Panda 状态条件 DiT 训练与验收

本版本交付训练工具，不代表 DP 已完成正式训练或喂餐验收。只使用冻结的单豆 Panda M4 数据，训练后达到闭环门槛才冻结 `DP_v1`。UR5e 实现与历史证据保留。

## 1. 环境与资源

在 `/home/minashiki/FeedingRobot_DPRL` 执行以下命令。统一使用已有环境，不需要重新安装 PyTorch、LeRobot 或扩散库。

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

模型检查、训练、推理默认 `--device cuda`，支持 BF16 时自动使用 BF16，否则使用 CUDA FP32。CUDA 不可访问会明确退出，不自动改用 CPU；受限执行会话与宿主 GPU 访问分别记录。`--device cpu` 仅供显式诊断。

`configs/dp_dit.json` 默认给整个进程树 6 个逻辑 CPU，最多允许 8 个；PyTorch threads=4，DataLoader workers=2，interop threads=1，其他数值库线程数=1。已有原生库线程和后续子进程均继承 affinity 限制。评估串行运行，viewer 至多显示一个环境。所有运行记录实际硬件与 affinity。

## 2. 网络和数据

固定 12 个 DiT Block、hidden size 512、自注意力和交叉注意力均为 8 heads、MLP hidden dim 2048。每个 Block 是自注意力、交叉注意力、MLP 三条 adaLN-Zero 残差分支。该设计基于 [DiT 论文](https://arxiv.org/abs/2212.09748)，采用项目自己的动作序列实现，没有加载图像 DiT 权重。

H=16：20 Hz 下预测 0.8 秒，每 0.2 秒重新规划，只执行前 4 步。当前恢复示范只有 6–8 个合法动作，暂不扩大 horizon。模型输出 epsilon，使用 100 步 cosine 加噪和 10 步确定性 DDIM；输出反归一化后为基座坐标系 TCP twist（m/s、rad/s），经现有速度、加速度、IK、力和工作空间保护执行。

两帧状态间隔 50 ms，最近 0.2 秒提供 10 帧历史。过去位姿估计嘴部／接收面当前速度；未来状态、教师内部子段、种子和驱动计划不输入网络。条件包含 512 维全局编码及 13 个 token：2 个状态、10 个 causal CNN 历史、1 个阶段／交互。每次规划只编码一次条件。

监督标签使用下发 `actions`，不是原始提议或测得速度。正常回合使用合法动作，恢复回合只使用完成 `RECOVER→WAIT_READY` 的合法恢复段。窗口从当前动作开始，在阶段变化、无效动作或时间不连续处截断，padding 不进入损失或动作自注意力。样本先均衡选择有标签的阶段，再选择回合与窗口，避免大量 ACQUIRE 标签淹没恢复、等待、交付与撤离。

连续观测使用 M4 已冻结的 train 统计；one-hot 与布尔值保持原值。动作与派生速度的均值、尺度只从合法 train 标签拟合，常量尺度为 1。M5 归一化器单独保存，不改 M4 文件。历史按整数 tick 查询，只使用当时已存在的观测，并包含帧龄／有效 mask。

启动读取器会核验冻结清单、审计、数据文件集合、哈希、split、配额与已有重放摘要。所有旧验收与重放均不重复执行。M5 新代码、新配置、测试和本指南作为新增输入；README 与主计划的更新单独允许，既有物理／教师／采集代码不能变化。

## 3. 小规模检查

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m pytest tests/test_m5.py -q
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m feedingrobot.scripts.validate_m5 --device cuda --headless --output outputs/single_bean/v1/m5/dit/engineering_001
```

第二条命令只在固定 8 个窗口、固定噪声／时间步上更新最多 200 次，要求末段平均 loss 较初段下降至少 50%，并验证 DDIM 样本与 100 ms 真实物理执行链。`engineering_report.json` 保存损失、参数量、合法阶段数量、硬件和父证据；`physical_prefix.json` 的结果只证明短时接入，不能当作喂餐成功率。

`diagnostic.pt` 仅供诊断，训练恢复和最终冻结都会拒绝它。工程通过后状态为 `ready_for_training`，正式训练为 `not_run`，闭环验收为 `not_verified`，DP 为 `not_frozen`。每次新运行使用不存在的目录，避免覆盖历史结果。

## 4. 私下正式训练和续训

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m feedingrobot.scripts.train_dp --device cuda --headless --output outputs/single_bean/v1/m5/dit/run_001
```

默认 AdamW lr=1e-4、weight decay=0.01、batch=64、EMA=0.9999、梯度 norm 上限 1.0，共 100000 updates。每 1000 次做固定噪声的离线 validation，每 5000 次保存 checkpoint，每 10000 次对 validation 的正常／恢复各 15 个冻结场景执行完整闭环。闭环评估可能耗时较长，这属于你的正式训练过程。

不指定 `--headless` 时，闭环评估显示当前模型。预热后比较三组等工作量窗口，显示中位耗时超过 headless 的 1.5 倍就自动关闭，预算报告进入评估记录。没有可用显示服务时继续无显示评估，报告如实记录。

先用以下短运行验证训练入口、数据 worker 和 checkpoint，再决定正式预算：

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m feedingrobot.scripts.train_dp --device cuda --headless --updates 100 --output outputs/single_bean/v1/m5/dit/training_probe_001
```

短运行同样只是训练工具检查，不满足闭环放行。继续原目录、恢复 `last.pt`：

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m feedingrobot.scripts.train_dp --device cuda --headless --updates 100000 --resume outputs/single_bean/v1/m5/dit/run_001/last.pt --output outputs/single_bean/v1/m5/dit/run_001
```

`--updates` 表示累计目标，不是本次追加次数。正常结束保存 `last.pt`；中断后恢复最近一次已保存 checkpoint，最多丢失一个保存间隔的进度。checkpoint 保存网络、EMA、优化器、随机数、已消费的阶段采样状态、配置、归一化、源码与数据哈希。改变配置、代码、机器人或数据时应开始新目录，不能混用旧 checkpoint。

主要输出：`metrics.jsonl`、`hardware.json`、`data_summary.json`、`parent_audit.json`、`normalization.json`、`last.pt`、`step_<N>.pt`、`best.pt` 和各轮 validation 报告。`best.pt` 根据正常／恢复完整成功率的平均值选择，推理使用 EMA 权重；loss 最低不等于喂餐最好。

## 5. 闭环评估和正式冻结

手动检查候选模型，先使用 validation：

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m feedingrobot.scripts.eval_dp --checkpoint outputs/single_bean/v1/m5/dit/run_001/best.pt --split validation --device cuda --output outputs/single_bean/v1/m5/dit/run_001/manual_validation_001
```

评估从对应 M4 完整初态开始，运行实际 DP，不执行教师或保存的命令。1 ms 物理、20 ms 历史、50 ms 动作、200 ms 重规划共用整数时钟；阶段变化在物理步立即取消旧队列，减速保持，下一动作网格重新规划。同步推理会暂停仿真，因此推理耗时单独记录，不能用该结果宣称实时性已通过 M7。

每回合保存事件、失败原因、接触／腕力峰值、冲量、动作跳变、推理耗时及预测／下发／实际参考／实测速度。先看 pickup，再看 delivery 和 retract；恢复场景还必须出现真实 `RECOVER→WAIT_READY`。大量时间限、掉豆、提前撤离或同阶段抖动应从阶段指标、mask、条件响应和示范覆盖诊断。当前约 260 个记录并不保证这个模型规模的泛化。

只有完成 validation 选模后才使用 test，避免用 test 调参数。最终冻结命令直接进行一次 test 验收：

```bash
conda run -p /home/minashiki/anaconda3/envs/feedingrobot --no-capture-output python -m feedingrobot.scripts.validate_m5 --freeze --checkpoint outputs/single_bean/v1/m5/dit/run_001/best.pt --device cuda --headless --output outputs/single_bean/v1/m5/dit/run_001/final_acceptance_001
```

正常／恢复分别为 15 个预定场景，完整成功各至少 12/15；pickup、delivery、retract 与恢复完成率至少 90%。阶段分母是实际进入该阶段的回合，报告同时给出进入数量；所有 15 个恢复场景均须进入 RECOVER，不能通过绕过恢复提高成绩。失败返回非零，保存报告但不发布冻结清单。

全部通过才发布 `freeze_manifest.json` 的 `frozen`／`DP_v1`，绑定 checkpoint、EMA、条件编码器、归一化、训练配置、源码、M4 父证据与 test 报告。冻结清单与权重一起保存，之后才进入 M6。工程报告、诊断过拟合或成功视频均不能替代此门禁。
