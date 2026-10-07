# v4 私下 CUDA 评估

本入口独立于已绑定的采集和训练目录。保留原 v4 数据、工具、日志、权重及诊断标签；不创建优化器，不续训，不重复运行旧 M4 物理验收。实际策略评估由用户在宿主 feedingrobot 环境启动。

模型、条件编码和 DDIM 采样使用 CUDA；支持时使用 BF16 autocast，无 CPU 推理回退。默认 6 个逻辑 CPU，允许 `--cpu-threads 6|7|8`。导入前限制亲和性至最多 8 核，恢复选择的宿主核集合后执行评估，避免冻结 v4 包的六核导入限制阻止八核评估。Torch CPU threads 等于所选预算，interop/BLAS 为 1，DataLoader workers 为 0。

为节省时间，不启动 viewer。CUDA 策略回合串行运行，每个 MuJoCo 回合只有一个物理所有者。策略回合全部结束后，主进程 Torch CPU threads 降为 1，使用预算减 1 个单线程 CPU 进程并行精确回放：六核时 5 个，八核时 7 个。工作进程不加载策略、不创建 CUDA 上下文；整个进程树的 CPU 亲和性始终包含至多所选数量的逻辑核。同步推理暂停仿真，耗时进入记录，不能用于宣称 M7 实时性通过。

## 私下运行

在宿主 Bash 中设置以下公共参数。每个输出目录必须尚不存在，失败证据也应保留；重跑请换新的目录编号。

```bash
cd /home/minashiki/FeedingRobot_DPRL
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
ENTRY=tools/m5/correction_v4/evaluation/run.py
OUT=outputs/single_bean/v1/m5/dit
CPU_THREADS=6                         # 可改为 7 或 8
CPU_SET="0-$((CPU_THREADS-1))"
COMMON=(
  --base-checkpoint "$OUT/run_001/step_50000.pt"
  --parent-checkpoint "$OUT/correction_run_001/step_100000.pt"
  --v1-report "$OUT/correction_dataset_v1_001/report.json"
  --old-data-report "$OUT/correction_v3_dataset_001/report.json"
  --data-report "$OUT/correction_v4_dataset_001/report.json"
  --audit-report "$OUT/correction_v4_warmstart_audit_001/report.json"
  --cpu-threads "$CPU_THREADS"
)
CANDIDATES=(
  "$OUT/correction_v4_run_001/step_105000.pt"
  "$OUT/correction_v4_run_001/step_110000.pt"
)

# 可选：只核验祖先、完整数据绑定、检查点有限值、来源、源码快照及日志前缀。
# 不构建策略模型，不执行回合；后续每个模式也会执行这些核验。
taskset -c "$CPU_SET" "$PYTHON" "$ENTRY" audit "${COMMON[@]}" \
  --checkpoint "${CANDIDATES[@]}" --output "$OUT/correction_v4_eval_audit_001"

# 固定窗口、噪声 0/1/2，对照 100k 父 EMA 与两个候选 EMA。
taskset -c "$CPU_SET" "$PYTHON" "$ENTRY" offline "${COMMON[@]}" \
  --checkpoint "${CANDIDATES[@]}" --output "$OUT/correction_v4_eval_offline_001"

# 先完成小规模无辅助闭环及精确回放，再决定扩大。
taskset -c "$CPU_SET" "$PYTHON" "$ENTRY" small "${COMMON[@]}" \
  --checkpoint "${CANDIDATES[@]}" --output "$OUT/correction_v4_eval_small_001"
```

`audit` 不必与 `offline` 重复执行。105k 会检查截至该检查点的连续日志前缀，同时验证后续到 110k 的日志连续、有限；不会截断原日志。每个模式均绑定并复核完整训练日志和权重文件，评估期间应保持该分支文件不变。

`offline` 使用原 validation 的每回合／阶段代表，以及 v4 合并 train 的每回合／教师阶段代表、四个等待年龄桶、可用的 X 临界起点和 pickup_hold 首末窗口。报告分组包含来源、阶段、类别、32 个方向格及等待年龄；前四步与合法完整 horizon 的误差、速度和限幅比例分别统计。训练窗口明确标为 `v4_train_diagnostic`，用于解释监督拟合。噪声、预测、标签和 mask 保存在 `samples.npz`；标签 mask 只参与统计，不输入 DDIM。原 validation 的固定 seed=123 去噪损失也统一重算。

`small` 每个模型固定 16 场：8 个 v4 独立校准快照、4 个旧失败快照、原 validation 的 3 正常＋1 恢复；两个候选加父模型共 48 场。新快照为 800015/800022/800025/800028/800034/800035/800040/800045，每类两个、toward/away 各一且 X 正负兼有，整组覆盖四象限。旧快照为 780001/780003/780005/780007。名单固定，不按模型结果替换失败场。

所有接管回合保留接管前的合法因果历史，恢复相同完整状态；各模型记录的初态 SHA256 必须相同。相同物理 tick 使用相同初始扩散噪声。保持 1ms 物理、20ms 历史、50ms 动作、200ms 重规划，阶段改变立即取消旧队列；时限始终为原 reset 起算 60s。没有下降门控或教师下发动作。教师对象只提供纠偏目标供诊断，不进入模型条件或执行回路。

每场分别保存连续 TCP 纠偏资格 200ms 与真实豆子拾豆资格 500ms。纠偏每 1ms 检查横差、角差、速度和高度漂移，失稳立即重置。原高度／净空保护与真实安全制动继续执行，制动单列归属。拾豆直接记录原事件逻辑收到的支撑、离碗、豆子线速／角速、资格计时和 reset_mask，不改变原事件逻辑。回放除物理 ≤1e-10、观测 ≤1e-7、事件和结局一致外，还比较逐边界拾豆诊断 ≤1e-10。

`small`/`offline` 的 `diagnostic_completed` 表示工具执行完成。精确回放失败会使 `small` 返回 `diagnostic_failed` 和非零退出码。先读各模型自主纠偏、拾豆、运输、交付、完整成功及介入记录；若仍无自主改善，应按资格重置原因定位后决定续训。

## 完整 validation 与正式 test

```bash
# small 有改善后，先对表现较好的一个候选做完整 validation，以减少回合数。
# 这里的 110k 只是命令示例；CHOSEN 应由实际评估结果确定。
CHOSEN="$OUT/correction_v4_run_001/step_110000.pt"
taskset -c "$CPU_SET" "$PYTHON" "$ENTRY" validation "${COMMON[@]}" \
  --checkpoint "$CHOSEN" --output "$OUT/correction_v4_eval_validation_001"

# 也可在一次 validation 中传入两个候选：--checkpoint "${CANDIDATES[@]}"
# 在 validation 选定同一 CHOSEN 后，才运行预定 test。
# --freeze 明确请求在全部门禁通过时发布新目录内的 DP_v1 清单。
taskset -c "$CPU_SET" "$PYTHON" "$ENTRY" test "${COMMON[@]}" \
  --checkpoint "$CHOSEN" \
  --validation-report "$OUT/correction_v4_eval_validation_001/report.json" \
  --freeze --output "$OUT/correction_v4_eval_test_001"
```

`validation` 每个模型运行 66 场：全部 32 个 v4 校准快照、4 个历史快照、原正常／恢复各 15 场。一个候选加父模型共 132 场，两个候选共 198 场。原 validation 的完整成功各 ≥12/15、四项阶段率各 ≥90%，15 个恢复场景均进入 RECOVER；新 32 场须全部自主纠偏和完整成功且无安全介入，逐场相对父模型不得新增拾豆／下游／纠偏退化或安全介入；全部回放通过。历史结果单列诊断并参与退化比较。新快照的严格要求属于 v4 实验选模门禁。

报告的 `arms[].validation_passed` 表示对应候选是否通过；总体 `validation_passed` 表示至少一个候选通过，不自动选择最后或最低 loss 的模型。`test` 只接受报告中真正通过的同一候选，验证报告全部文件、输入、工具、绑定和采样协议未变；CPU 预算可在 6–8 内调整。`test` 只运行原预定正常／恢复各 15 场，不执行父模型或校准回合。

正式 test 沿用原 M5 门槛：正常／恢复完整成功各 ≥12/15；pickup、delivery、retract、recovery 阶段率各 ≥90%，分母为实际进入阶段的回合，并报告进入数量；全部 15 个恢复场景必须进入 RECOVER。全部精确回放通过且显式给出 `--freeze` 才创建 `freeze_manifest.json`、报告 M5 complete／DP_v1 frozen；失败保存证据、不创建清单。省略 `--freeze` 只保存 test 结果，M5 仍 incomplete。

发布清单使用明确的 schema 4／`audited_v4_fork` 承接协议，绑定原实验检查点与 EMA（含条件编码器）、归一化、训练配置、M4 父证据、100k 与 v4 祖先／数据绑定、leading 采样器、评估源码快照、validation 和 test 证据。原检查点的 `diagnostic=True` 和 schema 不改写。原 schema 1 评估／加载入口仍不兼容 v4；后续使用方须核验并支持本 schema 4 清单。

## 工具验证

```bash
taskset -c 0-5 "$PYTHON" -m pytest \
  tools/m5/correction_v4/evaluation/test_evaluation.py \
  tools/m5/correction_v4/training/test_training.py -q
```

测试只使用随机小网络、模拟评估门禁和真实快照上的固定动作短后缀，不执行实际 v4 策略评估或生产权重训练。交付验证结果见同目录 `RESULT.md`。
