# v4 评估入口交付

已实现独立的 CUDA 评估入口，实际 v4 权重评估、完整 validation、正式 test 和发布均未执行。原训练目录继续为 `trained_experimental_not_released`，M5 保持 incomplete、DP_v1 未冻结。运行命令见 [README.md](/home/minashiki/FeedingRobot_DPRL/tools/m5/correction_v4/evaluation/README.md)。

## 实现

- `audit` 核验原祖先／完整数据／零更新审计、v4 检查点有限值、更新计数、来源、运行 provenance、训练源码快照和连续日志。105k 只选取对应前缀，保留并核验后续 110k 日志，不截断或改写原文件。
- `offline` 配对 100k 父 EMA 和指定候选；原 validation 与明确标记的 v4 train 诊断分别统计，包含方向格、等待年龄、X 临界起点及完整 pickup_hold 首末窗口，使用固定噪声和 leading 10 步采样。
- `small` 固定每个模型 16 场；`validation` 固定每个模型 66 场。复用同一初态与因果历史，并核对实际保存的初态 SHA256。没有教师下发动作或下降门控。
- 每 1ms 记录纠偏连续资格及失稳重置；分别记录真实豆子拾豆资格、线速／角速和 reset_mask。原 200ms 重规划、50ms 动作、60s 时限、阶段取消和物理保护保留。安全制动独立归属，所有模型动作监督 mask 为 false。
- CUDA 策略回合串行，随后并行 CPU 精确回放，比较物理、观测、事件、结局及逐边界拾豆记录。回放失败不能产生成功完成的小评估或正式发布。
- `test` 要求匹配且完整通过的 validation 证据与同一候选。原正常／恢复各 15 场门槛不变；全部 test 和回放通过且显式给出 `--freeze` 才发布 schema 4 的 v4 冻结清单。原实验权重、schema 和 diagnostic 标签不改写。

## 资源与验证

默认 6 个逻辑 CPU，可选 7 或 8；模型和采样强制 CUDA。评估 Torch CPU threads 等于预算，interop／BLAS 为 1，DataLoader workers 为 0。回放阶段主进程 Torch CPU threads=1，五或七个工作进程各单线程且不初始化 CUDA。

独立无模型探针确认：导入前保存核 0–7，冻结 v4 包导入后收窄至 0–5，可恢复 0–7；Torch CPU threads=8、interop=1，所有原生线程亲和性均受八核集合约束，CUDA 未初始化。

最终宿主六核回归 **104 项通过，3 项未修改的长时采集校准测试未重跑，45.92s**。覆盖新增评估测试、原 v4 训练入口、M5 因果输入／归一化／采样／恢复／原诊断权重拒绝发布／DataLoader spawn-mmap，以及 v4 采样、持稳、拾豆记录和进程预算检查。四类接管的 300ms 固定动作后缀、安全制动后缀及无效模型提议均通过精确回放；没有实际 v4 策略能力测量。发布门禁测试使用模拟权重和报告。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
timeout 180s taskset -c 0-5 /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest \
  tools/m5/correction_v4/evaluation/test_evaluation.py \
  tools/m5/correction_v4/training/test_training.py \
  tests/test_m5.py tools/m5/correction_v4/test_v4.py \
  -k 'not real_controlled_handover_and_exact_replay and not all_32_real_direction_velocity_handovers_and_replays' \
  -q --durations=5
```

早期沙箱扩展回归在旧 DataLoader 检查处阻塞，确认进程后中断，得到 57 项通过和 3 项未选结果；该批次没有计作完整通过。最终宿主重跑包含该 DataLoader 检查，以上述 104 项结果作为交付证据。未宣称重新运行全部历史 175 项或旧 M4 物理验收。

最终只读保留审计通过：344 个冻结源码、941 个 v4 数据文件及文件集合、原报告、六个采集源码、两个训练源码和 50k／100k 祖先检查点均匹配既有绑定。新实现只位于本独立子目录。完整工具验证记录：[report.json](/home/minashiki/FeedingRobot_DPRL/outputs/single_bean/v1/m5/dit/correction_v4_evaluation_entry_check_001/report.json)。
