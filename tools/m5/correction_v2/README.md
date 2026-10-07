# M5 下探纠偏 v2：校准、宿主采集、零更新审计

本工具独立于 v1，固定正式 50k 父证据和 `correction_run_001/step_100000.pt` 的 EMA。
不创建优化器、不更新权重、不运行完整 validation/test、不冻结。真实 DP 前缀必须使用宿主 CUDA BF16，无法使用时直接拒绝，不能换成 CPU 采集。

## 运行顺序

在仓库根目录使用现有 feedingrobot 环境。所有输出目录必须不存在；不要覆盖报告或旧工具。

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
TOOL=tools/m5/correction_v2/run.py
BASE=outputs/single_bean/v1/m5/dit/run_001/step_50000.pt
PARENT=outputs/single_bean/v1/m5/dit/correction_run_001/step_100000.pt
V1=outputs/single_bean/v1/m5/dit/correction_dataset_v1_001/report.json
CAL=outputs/single_bean/v1/m5/dit/correction_v2_calibration_001
DATA=outputs/single_bean/v1/m5/dit/correction_v2_dataset_001
AUDIT=outputs/single_bean/v1/m5/dit/correction_v2_warmstart_audit_001

# CPU 教师校准；已有该目录时先阅读报告，不要重复执行。
"$PYTHON" "$TOOL" calibrate --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" \
  --previous-report outputs/single_bean/v1/m5/dit/descent_coupled_calibration_002/report.json \
  --output "$CAL"

# 仅在 CAL/report.json 的 status=passed 时，在宿主 GPU 终端执行。
"$PYTHON" "$TOOL" collect --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" --calibration-report "$CAL/report.json" --output "$DATA"

# 仅在数据 status=ready_experimental、12 条配额及所有回放达标后执行。
"$PYTHON" "$TOOL" audit --base-checkpoint "$BASE" --checkpoint "$PARENT" \
  --v1-report "$V1" --data-report "$DATA/report.json" --output "$AUDIT"
```

三个命令各自执行前置条件校验，不能通过改报告 status 放行。失败校准保留现场并停止后续组；修改控制器会改变 v2 工具哈希，使旧校准失效。重新校准需要新版本及与历史证据不重叠的种子方案，不能直接重复这些固定种子。

## 教师版本和门槛

`descent_alignment_v2_001` 在接管高度保持 z，使用原教师的位置/姿态增益和速度限制纠正水平位置与姿态。每 50ms 检查角误差 <0.01rad、水平误差 <0.7mm、角速度 <0.05rad/s、线速度 <0.002m/s，连续 200ms 后恢复原路径；对齐期间高度误差 >=0.7mm 即拒绝。

P1 停稳组在 TCP 到碗坐标 z=70mm 后制动，沿用先前实际停稳约 62mm 的验证区域；真实 DP 的 P1 动量组在约 62mm 直接接管。P2 在原 entry 目标上方 3–8mm 的窄区间触发，不自动扩大范围。
DP 偏离类别要求角残差 2–4°、水平偏移 1–2mm、角速度 <0.05rad/s。停稳组释放时线速度 <0.002m/s；动量组释放时向下速度 >0.002m/s。

几何筛查遍历全部 145 个勺体碰撞几何与全部碗几何，检查当前及保持高度对齐的旋转/平移路径，并扣除插值误差和按实测/参考速度估算的制动余量。剩余净空必须 >2mm。它不能证明 PD 执行器制动距离，必须结合每类两次配对物理成功；原接触、腕力及其他保护始终执行。

校准种子 760001–760008，每组两个，按 P1 停稳、P1 动量、P2 停稳、P2 动量顺序执行。校准使用教师前缀、已验证的低速联合脉冲、停稳；动量组额外以 −4.5mm/s 的实际向下命令引入速度。校准的实际接管高度包含此前制动位移，须读取 `trace.json` 的释放位置，不把触发高度等同于接管高度。扰动与制动从不作为教师标签，不改关节、位姿或观测。完整成功和回放是必要条件，首次失败立即停止。

实际 DP 采集按观察到的 ABOVE/pre_entry 走廊记录路径进度，不按时间或最近示范点跳段。接管取消动作队列，不重置适配器速度；对齐后 P1 恢复 pre_entry，P2 恢复 entry，只有原教师的真实到达判据可以推进 sweep。

## 数据和采样

| 类别 | 接收 | 最多种子 | 种子区间 |
| --- | ---: | ---: | --- |
| P1 停稳 | 2 | 4 | 770001–770004 |
| P1 动量 | 2 | 4 | 770005–770008 |
| P2 停稳 | 2 | 4 | 770009–770012 |
| P2 动量 | 2 | 4 | 770013–770016 |
| 已对齐对照 | 4 | 8 | 770017–770024 |

偏离组必须有实际 DP 前缀，不会用脉冲替代缺失配额。每个候选先运行同种子/场景的无扰动教师基线，验证初始物理状态和几何配对并回放。已对齐对照使用无扰动原教师。
每条接收回合必须在从 reset 起算的 60s 内真实 pickup、delivery、完整成功；全部原物理保护和逐步回放保持不变。未覆盖、失败或缺少 entry/sweep 合法窗口的尝试保留在 `attempts/`，不进入 `data/train/`。不足 12 条时报告 `quota_incomplete`，不能审计。

`action_owner.npy` 区分 DP、外部制动、脉冲、v2 对齐教师、原路径教师；`action_stages.npy` 记录真实教师目标阶段。只有教师拥有的完整动作标签有效，阶段中断/末尾不完整动作无效。
v2 的动作观测来自命令下发前，DP 非教师前缀仍保留在因果历史中；对照基线仅为旧教师物理一致性保留原记录时机。`plans.json` 保存真实 DP 规划输入和完整预测，`trace.json` 保存当前位姿/速度/残差与教师阶段。它们不作为模型额外输入。

保留原合法动作窗口截断规则。额外采样从 alignment、transition、aligned 三池等概率选择，再按回合/窗口均衡；每批 75% 原 M4、25% v2。transition 必须含真实 entry 与 sweep，包含纠偏完成后 pre_entry→entry→sweep 衔接。原 v1 数据仅作祖证。

## 审计与验收

报告绑定正式 50k、100k、v1、校准、v2 数据报告及工具 SHA256；验证精确文件覆盖、split/seed 隔离、原归一化、实际回放结果。
回放要求物理最大误差 <=1e-10、观测最大误差 <=1e-7，事件和结果完全一致。
零更新审计模型/EMA 从 100k EMA 逐张量承接，不加载旧优化器/RNG；CPU FP32 计算 64 样本混合批次的有限 loss，逐回合读取三类重点窗口，然后重新审计输入文件。
`audit_passed` 必须同时有 `step=100000`、`model_exact=true`、`ema_exact=true`、`optimizer_created=false`、`optimizer_updates=0`。有限 loss 只是计算与加载验证，不能证明策略已拾豆。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest \
  tools/m5/correction_v2/test_v2.py tools/m5/test_correction_data.py \
  tools/m5/post_training/test_calibrate_descent.py tests/test_m5.py \
  -q -k 'not spawn_workers_read_identical_mmap_windows'
```

实际执行状态和后续宿主交接见同目录 `RESULT.md`。
