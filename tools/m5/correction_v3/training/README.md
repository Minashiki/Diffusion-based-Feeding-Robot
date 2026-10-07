# v3 专用训练和可视化闭环入口

这两个入口位于独立子目录；没有修改 v1/v2/v3 采集工具、原证据、归一化或原权重。`train.py` 只使用原训练集及 **v3 接收的 12 条**，不拼入 v2 数据。新增 Python 工具独立绑定并保存快照，旧采集报告的源码绑定继续有效。

训练从原实验 **100000 EMA** 初始化模型和 EMA，重新创建优化器与随机状态。保持原架构、扩散损失、优化器参数、EMA 和梯度保护；每批 75% 原数据，25% v3 alignment / transition / aligned 三池等概率采样。外部制动、DP 前缀等无效标签仍由已审计的合法窗口读取器排除。这批数据监督的是制动后低速下降中的姿态纠偏，没有高速自主制动标签。

训练强制 CUDA，无 CPU 回退；有支持时使用 BF16 autocast。默认限制整个进程及导入时的本地线程池到最多 8 个逻辑核，运行时缩至 6 个，Torch CPU 线程为 6，interop 与 BLAS 线程为 1。`--cpu-threads 8` 可选，允许范围 6–8；DataLoader workers 固定为 0，批次同步生成，避免额外进程争抢 CPU。原检查点 config 不被修改，实际硬件和预算单独记录。

在仓库根目录的 feedingrobot 环境运行：

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
ENTRY=tools/m5/correction_v3/training
OUT=outputs/single_bean/v1/m5/dit
COMMON=(
  --base-checkpoint "$OUT/run_001/step_50000.pt"
  --parent-checkpoint "$OUT/correction_run_001/step_100000.pt"
  --v1-report "$OUT/correction_dataset_v1_001/report.json"
  --data-report "$OUT/correction_v3_dataset_001/report.json"
  --audit-report "$OUT/correction_v3_warmstart_audit_001/report.json"
  --cpu-threads 6
)

# 只检查祖先、数据、配对校准、零更新审计及合法采样池；不建模型、不训练。
# 每个新输出目录必须尚不存在。
"$PYTHON" "$ENTRY/train.py" preflight "${COMMON[@]}" \
  --output "$OUT/correction_v3_preflight_001"

# 私下小规模续训示例：累计目标 100200 = 在 100k 后新增 200 次更新。
# 这是命令示例，不代表已训练或足够改善闭环；按需要改目标。
"$PYTHON" "$ENTRY/train.py" train "${COMMON[@]}" --updates 100200 \
  --output "$OUT/correction_v3_run_001"

# 同一 v3 分支继续到累计 100400；继承本分支的模型、EMA、优化器及 RNG。
"$PYTHON" "$ENTRY/train.py" train "${COMMON[@]}" --updates 100400 \
  --output "$OUT/correction_v3_run_001" \
  --resume "$OUT/correction_v3_run_001/last.pt"
```

`--updates` 始终是累计目标，必须大于当前 step。首次输出目录不覆盖；恢复要求同一目录、相同祖先、数据、工具、Torch/diffusers 版本和 CPU 预算。保存 `step_T.pt`、`last.pt`、`metrics.jsonl`、`provenance.json` 和 `status.json`。原频率的 validation 去噪 loss 与检查点保留，最后不足保存周期的小段也会保存。不自动运行闭环、不生成 best、不使用 test、不冻结。

恢复会核验从 100001 到检查点 step 的连续有限 metrics。若进程异常中断后日志超前于最近检查点，入口会拒绝恢复；先备份日志，再将工作副本保留到所选检查点对应 step。不要直接覆盖既有证据报告或编辑权重。一次正常结束的小规模训练可直接恢复。

## 看 MuJoCo 小规模闭环

先训练，再执行：

```bash
"$PYTHON" "$ENTRY/evaluate.py" small "${COMMON[@]}" \
  --checkpoint "$OUT/correction_v3_run_001/last.pt" \
  --output "$OUT/correction_v3_small_001"
```

**默认打开 MuJoCo 窗口，按仿真时间实时显示每个回合。** ObserverViewer 在独立进程只读显示，不修改物理。需要宿主桌面的有效 DISPLAY；窗口不可用或中途关闭会明确报错，不会悄悄改成无界面。只有主动指定 `--headless` 才禁用窗口。实时显示会增加等待时间。

依次比较 100k 父 EMA 和训练后的 v3 EMA，同一个物理 tick 使用相同初始扩散噪声，leading / 10 步 / 每 200ms 重规划、每 50ms 下发动作。默认 `--noise-seed 0`；可另用新输出目录指定其他噪声种子。

- 纠偏探针：独立校准种子 **780001、780003、780005、780007**，覆盖 P1/P2 停稳/动量四类，不属于训练的 12 条。从经过校验的 `handover_state.pkl` 恢复真实低速接管状态和过去的观测历史。后续动作全部来自模型，直到真实拾豆、运输、交付、回撤或失败；不执行教师纠偏、教师 entry 或教师 sweep。原 reset 起算的 **60s** 限制保持。
- 原 validation 探针：排序最前的 3 个正常场景与 1 个恢复场景，从原初始状态运行完整模型闭环，保留原物理保护和阶段取消规则。这里没有低速快照或额外外部预制动，检验模型能否自己完成完整流程。
- 纠偏通过沿用角误差 `<0.01rad`、水平误差 `<0.7mm`、实际线速度 `<2mm/s`、角速度 `<0.05rad/s`，连续稳定 200ms。纠偏期间每个物理步检查高度漂移 `<0.7mm`；漂移或净空超限触发真实零命令安全制动并判定该探针失败，不把制动当作纠偏成功。完成纠偏后允许正常下探，后续继续原物理保护。

每场 `result.json` 分别包含 `model_alignment`、`external_braking`、真实 `pickup` / `transport_completed` / `delivery` / `success` / `recovery_completed`、事件、接触/腕力峰值和可视化报告。`trace.json` 记录模型预测、下发动作、动作归属和纠偏残差；完整命令/物理/观测记录支持精确回放，入口自动核验回放。所有新动作标签均无效，诊断输出不会进入训练。

`external_braking.prefix_commands` 是快照前已发生的外部制动；`interventions` 和 `commands` 是本次模型纠偏期间新发生的安全制动。两者都不计作模型自主制动。校准快照只证明局部低速接管条件下的表现，完整初始状态探针另行检查真实闭环。

总报告列出两模型的纠偏通过数、完整成功数、安全介入次数，以及逐场拾豆、运输、交付、恢复和完整成功的退化清单。`small` 的 `diagnostic_completed` 只表示诊断完成；少量成功不能视为 M5 验收。查看是否纠偏改善、确实拾豆且下游未退化，再扩大验证。

## 完整 validation

```bash
"$PYTHON" "$ENTRY/evaluate.py" validation "${COMMON[@]}" \
  --checkpoint "$OUT/correction_v3_run_001/last.pt" \
  --output "$OUT/correction_v3_validation_001"
```

比较两模型的全部 8 个独立校准快照及原 validation 正常/恢复各 15 场，仍默认逐场可视化。报告 `validation.v3` 使用原验收门槛：正常/恢复各完整成功至少 12/15，各阶段成功率至少 90%，15 个恢复场景全部进入 RECOVER。`validation_acceptance_passed` 还要求 8 个探针均自主完成纠偏及完整流程、无新安全制动、逐场下游指标无退化；失败时退出码为 1。不能用局部纠偏通过替代完整流程结果。

工具只支持诊断和 validation，不提供正式 test 或冻结入口。完整 validation 达标后再进入正式 test 与冻结，M5 在这之前保持 incomplete。

## 回归

```bash
"$PYTHON" -m pytest "$ENTRY/test_training.py" \
  tools/m5/correction_v3/test_v3.py tools/m5/correction_v2/test_v2.py \
  tools/m5/test_correction_data.py tools/m5/post_training/test_calibrate_descent.py \
  tests/test_m5.py -q
```

新测试的更新仅针对随机初始化的小模型，验证 EMA 初始化和连续/中断恢复的一致性；不更新实际 100k 权重。真实物理测试用固定零模型动作运行短诊断后缀并精确回放，不作为策略能力证据。

本轮交付的预检、宿主 CUDA/线程预算、MuJoCo 图形烟测及 118 项回归结果见 [RESULT.md](RESULT.md)。
