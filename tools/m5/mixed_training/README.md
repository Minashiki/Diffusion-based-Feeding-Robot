# V1 与不同起点数据联合训练准备

`config.json` 使用 MinaDiTV2，基础数据仍为冻结的 V1。训练读取 V1 的
100 条正常轨迹、100 条恢复轨迹，再加入 start_pose_v2 的 200 条正常轨迹。
验证、测试仍使用 V1 各自的 30 条轨迹（正常与恢复各 15 条）。
不配置 correction 数据；报告目录中的 rejected 和 pending 不参与读取。

保留原阶段采样：均匀选择阶段，再在该阶段均匀选择轨迹和动作窗口。
保留 V1 恢复轨迹原有动作掩码。联合 Min-Max 仅从两份 train 数据拟合，
验证和测试使用同一套参数，不参与拟合。训练来源绑定写入新 checkpoint，
恢复训练时重新核验，不能通过旧 checkpoint 的 `--resume` 改换数据。

仅准备和验证（不会创建优化器、反向传播或保存模型 checkpoint）：

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest tools/m5/mixed_training/test_mixed.py tests/test_minadit.py -q
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/mixed_training/prepare.py
```

默认输出为 `outputs/single_bean/mixed_v1_start_pose_v2/preparation_001/`，
包含 config、联合 normalization、两份数据审计、数据统计和零更新前向检查结果。
重复准备时传入新的 `--output` 目录，避免覆盖既有报告。

数据核验直接检查原 V1 冻结证据及新数据完整文件哈希、成功清单和回放证据。
新数据的采集源码快照和报告保持原样；此次训练接入修改了 M5 源码，
不再要求当前训练源码与历史采集时的整个源码集合相同。

以后明确启动正式训练时使用以下命令（本次未执行）：

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp \
  --config tools/m5/mixed_training/config.json \
  --output outputs/single_bean/mixed_v1_start_pose_v2/train_001 \
  --device cuda --headless
```

正式训练会重新核验数据并拟合联合归一化，使用独立输出目录，不改动原模型和数据。
