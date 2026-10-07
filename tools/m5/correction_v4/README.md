# M5 v4 数据修复与受控方向示教

本目录的 run.py 只修复监督和采集示教，不提供训练或旧检查点恢复入口。旧 v3 12 条保持逐字节不变；新版本的 32 条来源为新 reset、真实路径教师前缀和真实命令扰动，不能称为 DP 前缀或自主制动。扰动、制动、前缀均不作为标签。输入、模型、归一化、原 60s 时限及保护阈值不变，M5 保持 incomplete。

数据审计完成后，私下 CUDA 训练使用独立子目录的 [训练说明](/home/minashiki/FeedingRobot_DPRL/tools/m5/correction_v4/training/README.md)。新增入口绑定训练工具并接入 v4 完整采样范围，不改变采集工具哈希；训练须由用户启动。

CPU 亲和性从导入开始限制到至多 6 个逻辑核，interop/BLAS 为 1。配对物理回合用 5 个独立单线程进程，主进程留 1 个 CPU 预算；每个回合只有一个物理所有者。不启动 DataLoader 子进程。教师采集和命令回放使用 CPU FP64 物理，不执行模型；最终零更新审计在全部物理进程退出后用 6 线程 CPU FP32 100k EMA。多进程步骤在宿主执行。

扩展数据占每批 25%，三池等概率。池内均匀选回合、教师阶段、窗口；纠偏池的纠偏过程/最终等待各半。后段覆盖到完整 pickup_hold，包含 ACQUIRE 与 TRANSPORT 两部分，并按相位与拾豆确认终点截尾。等待起点只用于离线索引与诊断，不进入策略特征。旧示教仍是 50ms 教师资格检查；新示教每 1ms 检查，连续满 200ms 后在下个 50ms 边界转场，保持原反馈律。

四类别按 p1_stopped/p1_moving/p2_stopped/p2_moving 排序，每类四象限按 (-,-)/(-,+)/(+,-)/(+,+) 排序，格内 toward/away。首两版已使用 800001–800014，保留证据；最终配对校准使用未用的 800015–800046。采集每格最多两个独立尝试，800101–800164。任何历史使用过的种子禁止重用。配对校准失败即停；配额不足保存 quota_incomplete，不进入零更新审计。

P1 高速下降与真实制动会产生横向跟踪漂移。外部下降/制动命令以原位置增益的两倍保持指定 1.5mm 偏差，下降线速仍限幅至原机器人门槛；Z 制动始终是真实参考减速，不调用 stop/reset 清除速度。上述 XY 反馈与全部准备动作仍屏蔽监督，正式纠偏教师保持原反馈律。

每条示教都要真实穿越对应 X 临界区间，并记录该区间是否包含 50ms 教师起点；不把短于 50ms 的穿越误判为教师失败。整批按四类别与正负 X 分组，每组至少两条独立示教具有临界教师起点，否则校准失败或数据标为 coverage_incomplete。每条仍须覆盖四个持稳年龄区间。

在仓库根目录的 feedingrobot 环境执行；每个输出目录必须尚不存在，工具哈希改变后须使用新重索引报告。下面使用最终报告名称，实际结果见 RESULT.md。

```bash
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
ENTRY=tools/m5/correction_v4/run.py
OUT=outputs/single_bean/v1/m5/dit
COMMON=(
  --base-checkpoint "$OUT/run_001/step_50000.pt"
  --checkpoint "$OUT/correction_run_001/step_100000.pt"
  --v1-report "$OUT/correction_dataset_v1_001/report.json"
  --data-report "$OUT/correction_v3_dataset_001/report.json"
)
"$PYTHON" "$ENTRY" reindex "${COMMON[@]}" --output "$OUT/correction_v4_reindex_005"
"$PYTHON" "$ENTRY" calibrate "${COMMON[@]}" \
  --reindex-report "$OUT/correction_v4_reindex_005/report.json" --output "$OUT/correction_v4_calibration_003"
"$PYTHON" "$ENTRY" collect "${COMMON[@]}" \
  --calibration-report "$OUT/correction_v4_calibration_003/report.json" --output "$OUT/correction_v4_dataset_001"
"$PYTHON" "$ENTRY" audit "${COMMON[@]}" \
  --collection-report "$OUT/correction_v4_dataset_001/report.json" --output "$OUT/correction_v4_warmstart_audit_001"
"$PYTHON" "$ENTRY" pickup-audit "${COMMON[@]}" \
  --episode "$OUT/correction_v3_runtime_probe_003/holdgate200/v3/780001_calibration" \
  --output "$OUT/correction_v4_pickup_audit_002"
```

新报告保存源码快照、旧报告/数据绑定、回合配对与回放、真实接管方向/速度和临界覆盖。每个回合的 alignment_boundaries.npy 记录逐毫秒位置误差、角误差、速度与稳定计时；pickup_boundaries.npy 记录事件逻辑实际收到的支撑、离碗、真实豆子速度与资格计时。pickup_diagnostics.json 的 reset_mask 位依次表示支撑、离碗、线速、角速失败；同时失败的条件全部保留。诊断包装不修改原事件逻辑，原 update 恰好调用一次。

旧 780001 仅执行已保存命令回放，先通过物理 ≤1e−10、观测 ≤1e−7 与事件/结局一致检查，再解释资格中断。X 临界穿越与持稳年龄分别验收，不声称两者必然共同出现。窗口重复损失位置不代表独立示教数量；数据通过不代表策略或 M5 验收通过。

```bash
"$PYTHON" -m pytest tools/m5/correction_v4/test_v4.py \
  tools/m5/correction_v3/runtime_probe/test_probe.py tools/m5/correction_v3/training/test_training.py \
  tools/m5/correction_v3/test_v3.py tools/m5/correction_v2/test_v2.py \
  tools/m5/test_correction_data.py tools/m5/post_training/test_calibrate_descent.py tests/test_m5.py -q
```
