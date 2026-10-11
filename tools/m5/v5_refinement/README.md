# MinaDiTV5：从 90k 分叉精修

入口 `run.py` 从原始 `step_90000.pt` 创建独立训练运行，不修改旧 checkpoint、配置或源码绑定。
继承 raw 模型、EMA、AdamW 动量、随机数与阶段采样状态，复用原归一化参数。
加载优化器后覆盖学习率，不重新拟合数据统计。

| 累计 updates | 学习率 | EMA decay |
|---|---:|---:|
| 90001–100000 | `2e-5` | `0.9998` |
| 100001–110000 | `1e-5` | `0.9998` |

每 1000 步记录离线 loss、首动作及前四步等物理动作指标；每 5000 步保存 checkpoint 并进行
固定 15 normal + 15 recovery 闭环验证。默认不超过累计 110000 步。
100000 步验证后，只有完整成功数大于 0，或取豆至少 5/30 且正常场景取豆大于 0，才进入第二段。
门槛不通过时保存该步 checkpoint 和 `status.json`，停止；恢复入口会拒绝继续越过失败门槛。

在项目根目录执行。先做只读检查，无需 CUDA，不创建输出目录：

```bash
cd /home/minashiki/FeedingRobot_DPRL
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/v5_refinement/run.py \
  --init-from outputs/MinaDiTV5/run_002/step_90000.pt \
  --output outputs/MinaDiTV5/refine_90k_001 --check-only
```

确认 GPU 环境后开始精修，输出目录必须尚不存在：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/v5_refinement/run.py \
  --init-from outputs/MinaDiTV5/run_002/step_90000.pt \
  --output outputs/MinaDiTV5/refine_90k_001 --device cuda --headless
```

中断后仍使用此入口，加载精修目录里的 `last.pt`：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/v5_refinement/run.py \
  --resume outputs/MinaDiTV5/refine_90k_001/last.pt \
  --output outputs/MinaDiTV5/refine_90k_001 --device cuda --headless
```

不要使用普通 `train_dp --resume` 恢复精修：它没有本工具的两段学习率及进度门槛。
`--updates 100000` 可以只运行第一段；该参数始终表示累计目标，不是新增步数。

`best.pt` 从 90k 的真实闭环结果初始化，按完整成功数、取豆成功数、首四步线／角速度相对误差之和选取。
`selection.json` 记录最佳步数与指标；`refinement.json` 记录父 checkpoint、原配置、工具与基线证据哈希。
每个新 checkpoint 保留这些来源信息及门槛状态，续训会检查来源、配置、归一化和门槛证据。
工具源码改变后不能直接恢复同一精修运行。原运行的 `best.pt` 不参与这次选模。

恢复时只保留当前 `last.pt` 已提交步数以内的精修日志，之后的未提交日志会被丢弃；
未完成的验证目录保留，重试使用新目录。请保留精修目录内的 `step_*.pt` 和验证结果，以支持最佳模型恢复和证据检查。
本工具不运行 test、不冻结或发布模型。

CPU 小模型验证：

```bash
PYTHONPATH=src:. /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest -q tools/m5/v5_refinement/test_refinement.py
```
