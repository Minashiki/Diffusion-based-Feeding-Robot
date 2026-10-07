# M5 下探纠偏 v3：真实动量制动与独立安全停止

v3 独立保存工具和证据，沿用正式 50k 与实验 100k EMA。v2 的十条接收数据、失败现场、校准和工具不变，不能直接拼入 v3 配额。

## 本轮改变

- P1 的约 50mm/s 前缀先取消 DP 队列，再以 `external_brake` 命令连续制动。保留完整物理和适配器状态，不调用 stop/reset 清零速度，不修改关节、位姿或观测。制动段不作为教师标签。
- 动量组在实际向下速度 `>2 且 <=5mm/s`、实际线速度 `<=5mm/s`、参考线速度 `<=4.5mm/s` 时重新检查类别、走廊及净空，捕获纠偏高度。参考速度可以已为零，但实际向下速度必须非零；停稳回合不能计入动量组。
- 制动下移量与纠偏高度漂移分开记录。纠偏仍要求高度漂移 `<0.7mm`，并在每个物理步检查。位置、姿态、连续 200ms 稳定判据及原物理保护保持。
- 净空余量包含 450ms 物理制动包络，依据已保存 50mm/s 前缀约 400ms 的实际制动加一个 50ms 决策间隔；该估算必须由新配对物理校准验证。
- DP 前缀的安全停止独立于类别资格。当前全勺净空不足制动余量加 2mm 时，连续制动并拒收该回合。角残差 2–4°、水平偏移 1–2mm 的采样门槛保持。
- 原合法窗口、真实 entry/sweep、75% 原数据与 25% 三池采样、原归一化和精确回放要求保持。

## 验证和运行

所有输出目录必须不存在，不覆盖旧证据。使用现有 feedingrobot 环境：

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
TOOL=tools/m5/correction_v3/run.py
BASE=outputs/single_bean/v1/m5/dit/run_001/step_50000.pt
PARENT=outputs/single_bean/v1/m5/dit/correction_run_001/step_100000.pt
V1=outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json
V2DATA=outputs/single_bean/v1/m5/dit/correction_v2_dataset_001/report.json
V2CAL=outputs/single_bean/v1/m5/dit/correction_v2_calibration_001/report.json
DIAG=outputs/single_bean/v1/m5/dit/correction_v3_diagnosis_001
CAL=outputs/single_bean/v1/m5/dit/correction_v3_calibration_001
DATA=outputs/single_bean/v1/m5/dit/correction_v3_dataset_001
AUDIT=outputs/single_bean/v1/m5/dit/correction_v3_warmstart_audit_001

# CPU：三个保存的高速接管状态需完整成功；两个原未接管前缀需安全停止。
# 保存状态的回放验证纠偏后缀，历史种子/命令仅用于诊断，不计入新配额。
"$PYTHON" "$TOOL" diagnose --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" --previous-report "$V2DATA" --output "$DIAG"

# CPU：DIAG passed 后，新种子 780001–780008，各类别两组配对校准。
# P1 在 ABOVE 安全区域引入偏差后，以实际 45–51mm/s 到达制动触发点。
# 必须在原 60s 内完整成功，并通过基线与纠偏的物理/观测/事件回放。
"$PYTHON" "$TOOL" calibrate --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" --previous-report "$V2CAL" \
  --diagnosis-report "$DIAG/report.json" --output "$CAL"

# CAL passed 后，宿主 CUDA BF16：新种子 780101–780124，原 2/2/2/2/4 配额。
"$PYTHON" "$TOOL" collect --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" --calibration-report "$CAL/report.json" --output "$DATA"

# 仅在新数据 ready_experimental、12 条配额和回放/合法窗口通过后执行。
"$PYTHON" "$TOOL" audit --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" --data-report "$DATA/report.json" --output "$AUDIT"
```

现有目录应先读报告，不能重跑固定种子。工具变化使旧报告绑定失效；修订后的校准需要新的版本、输出目录和未用种子。失败诊断/校准保留现场并停止后续组，不改 status、门槛、尝试上限或 60s 时限放行。

采集拒绝 CUDA/BF16 不可用，不能转 CPU 生成真实 DP 前缀。审计为 CPU FP32 64 样本有限 loss 和逐张量 EMA 承接检查，更新为零；本工具不训练、不运行完整 validation/test、不冻结。

在宿主 feedingrobot 环境运行回归（包含多进程 mmap 检查）：

```bash
"$PYTHON" -m pytest tools/m5/correction_v3/test_v3.py \
  tools/m5/correction_v2/test_v2.py tools/m5/test_correction_data.py \
  tools/m5/post_training/test_calibrate_descent.py tests/test_m5.py \
  -q
```

实际执行结果见同目录 `RESULT.md`。
