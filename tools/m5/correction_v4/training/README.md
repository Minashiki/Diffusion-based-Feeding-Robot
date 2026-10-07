# v4 私下 CUDA 训练

使用已审计的 v4 合并 44 条数据和完整 pickup_hold 索引。初始模型及 EMA 从 100000 EMA 加载，优化器/RNG 重新初始化；保持原模型、输入、归一化、batch_size=64、学习率 1e−4、EMA=0.9999 和梯度保护。每批 75% 原数据、25% v4 三池，阶段及窗口均衡沿用 v4 读取器。该入口在独立子目录，原 v4 工具哈希与采集、零更新报告继续有效。

训练强制 CUDA，无 CPU 回退。支持时使用 BF16 autocast；当前宿主 RTX 5080、Torch 2.11.0+cu130、CUDA 13.0 均已只读核验，BF16 可用。模型、EMA、训练批次和优化器状态在 CUDA；CPU 用于读取、采样及辅助计算。CPU 亲和性限定六核，Torch CPU 线程为 6，interop/BLAS 为 1，DataLoader workers 为 0，不额外启动采样子进程。此入口只接受 --cpu-threads 6。

在宿主终端执行，不在缺少 GPU 访问的沙箱中运行：

```bash
cd /home/minashiki/FeedingRobot_DPRL
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PYTHON=/home/minashiki/anaconda3/envs/feedingrobot/bin/python
ENTRY=tools/m5/correction_v4/training/train.py
OUT=outputs/single_bean/v1/m5/dit
COMMON=(
  --base-checkpoint "$OUT/run_001/step_50000.pt"
  --parent-checkpoint "$OUT/correction_run_001/step_100000.pt"
  --v1-report "$OUT/correction_dataset_v1_001/report.json"
  --old-data-report "$OUT/correction_v3_dataset_001/report.json"
  --data-report "$OUT/correction_v4_dataset_001/report.json"
  --audit-report "$OUT/correction_v4_warmstart_audit_001/report.json"
  --cpu-threads 6
)

# 可选：重新预检，核验祖先、数据、校准、零更新审计和采样池；不建模型或优化器。
# 已完成的预检目录为 correction_v4_training_preflight_001；这里另用新目录。
taskset -c 0-5 "$PYTHON" "$ENTRY" preflight "${COMMON[@]}" \
  --output "$OUT/correction_v4_training_preflight_002"

# 首轮对照示例：累计 110000，即从 100k 新增 10000 次更新。
# 新训练目录必须尚不存在；不会自动启动闭环或更新验收状态。
taskset -c 0-5 "$PYTHON" "$ENTRY" train "${COMMON[@]}" --updates 110000 \
  --output "$OUT/correction_v4_run_001"

# 若先做 200 步运行检查，把首次目标改成 100200，完成后用本命令继续到 110000。
# 同一 v4 分支恢复模型、EMA、优化器和 RNG。
taskset -c 0-5 "$PYTHON" "$ENTRY" train "${COMMON[@]}" --updates 110000 \
  --output "$OUT/correction_v4_run_001" --resume "$OUT/correction_v4_run_001/last.pt"
```

--updates 是累计目标，必须大于当前 step。训练前核验数据与审计绑定；中途/结束保存前再核验工具、报告、数据和旧祖先。原 validation 去噪 loss 每 1000 步记录，检查点每 5000 步及正常结束时保存。输出包括 step_T.pt、last.pt、metrics.jsonl、provenance.json、status.json；恢复必须保持同目录、数据、工具、运行库及预算一致。异常中断后若日志比所选检查点超前，恢复会拒绝，先备份日志再保留到该检查点 step。

110000 是首轮对照预算建议，不代表已证明足够或最优。用 105000/110000 EMA 与 100k 对照纠偏持稳、实际拾豆及下游完整成功，再决定是否继续。该入口没有自动闭环、test 或冻结流程，训练不会令 M5 自动通过；v3 专用评估入口也不能直接接收 v4 分支绑定。

本次仅完成预检和 4 项入口测试，未创建实际训练模型/优化器或启动训练。预检报告：
[correction_v4_training_preflight_001](/home/minashiki/FeedingRobot_DPRL/outputs/single_bean/v1/m5/dit/correction_v4_training_preflight_001/report.json)。
