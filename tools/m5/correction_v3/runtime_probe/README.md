# 130k 执行窗口与下降门控诊断

独立入口不修改旧工具、训练集、归一化或权重。只加载100k父EMA和130k EMA，不创建优化器、不训练、不导出训练标签、不运行正式test或冻结。

固定四个独立低速快照780001/780003/780005/780007，依次运行两个模型的三个模式，共24回合：`baseline200` 每200ms规划、执行前四步；`replan50` 每50ms规划、只执行首步；`holdgate200` 保持200ms规划，纠偏稳定满200ms前仅屏蔽世界坐标向下速度。三个模式不组合。预测仍为16步、leading/10步、eta=0；各物理tick采用同一噪声规则。门控不增加高度反馈，不能保证实际高度立即停止，原每物理步0.7mm漂移及净空保护仍然生效。实际修改过命令的纠偏通过仅算辅助通过。

默认强制CUDA，宿主支持时使用BF16，CPU限制6核（可设8）。默认打开MuJoCo窗口，关闭或无法打开会报错；只有明确指定 `--headless` 才关闭显示。显示等待或推理超周期不会跳过物理步。每场从快照继续至真实终止或原reset起算60s，包括拾豆及后续流程。

诊断CPU预算与旧训练绑定分开：`--cpu-threads 8`使用8核执行，但旧训练审计仍使用检查点来源的6核绑定，不重写祖先信息。GPU与软件运行环境仍须匹配基线。

在仓库根目录、宿主feedingrobot环境执行：

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
OUT=outputs/single_bean/v1/m5/dit
"$PYTHON" tools/m5/correction_v3/runtime_probe/run.py small \
  --base-checkpoint "$OUT/run_001/step_50000.pt" \
  --parent-checkpoint "$OUT/correction_run_001/step_100000.pt" \
  --v1-report "$OUT/correction_dataset_v1_001/report.json" \
  --data-report "$OUT/correction_v3_dataset_001/report.json" \
  --audit-report "$OUT/correction_v3_warmstart_audit_001/report.json" \
  --checkpoint "$OUT/correction_v3_run_001/last.pt" \
  --baseline-report "$OUT/correction_v3_small_001/report.json" \
  --output "$OUT/correction_v3_runtime_probe_001" \
  --cpu-threads 6 --noise-seed 0
```

输出目录必须不存在；当前检查点必须为130000且SHA匹配基线。噪声种子必须匹配对应基线报告。三个模式都从同一初始状态开始。首先对两模型的8场baseline核对旧报告全部有效数组、实际命令、初始状态和事件；不一致立即保存error报告并停止。旧binding原样核验，新工具/输入/checkpoint另行绑定并在运行前后核验。所有24场自动精确回放。

不传`--output`时使用示例中的`correction_v3_runtime_probe_001`。本轮已保留`_001`、`_002`和`_003`，复跑应指定新的未使用目录；逐场结果与下一步判断见[RESULT.md](RESULT.md)。

输出包括每场 `result.json`、`trace.json`、命令、物理/动作/观测数组、归属和源码快照。原预测保存在`proposals.npy`，实际命令在`actions.npy`；所有`action_mask`均为false。`model_hold_gate`与真正`external_brake`分开记录，不能把门控协助宣称为自主纠偏或自主制动。

`alignment_diagnostics`记录首次瞬时合格、最长连续稳定、首次实际下发下降指令和原条件通过；`model_alignment.success`只计无门控修改的通过。`hold_gate.assisted_alignment_pass`单列辅助通过。推理计时覆盖采样至CPU拷贝完成，单列每场首次调用、后续p50/p95/max和超周期比例；不含viewer等待、落盘或回放。动作变化区分普通动作与阶段/制动/门控边界。`diagnostic_completed`仅表示24场证据与回放完成，不等于策略验收通过；M5保持incomplete。

```bash
"$PYTHON" -m pytest tools/m5/correction_v3/runtime_probe/test_probe.py \
  tools/m5/correction_v3/training/test_training.py \
  tools/m5/correction_v3/test_v3.py tools/m5/correction_v2/test_v2.py \
  tools/m5/test_correction_data.py tools/m5/post_training/test_calibrate_descent.py \
  tests/test_m5.py -q
```

新增测试只使用随机小模型或固定动作，不更新实际模型。结果先判断无辅助短窗口是否改善；仅门控改善说明需要进一步处理持稳/转场决策；无改善则继续分析覆盖和监督。局部改善但未拾豆时不扩大到完整validation，本入口不自动补数据或训练。
