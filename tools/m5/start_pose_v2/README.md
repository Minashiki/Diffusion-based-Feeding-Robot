# 不同初始位置的正常训练扩充集

固定 Panda 底座和 V1 勺子朝向，候选 TCP 世界坐标范围为
X 0.35–0.55、Y -0.25–-0.04、Z 0.12–0.25 米。40 个位置格各收集 5 条完整成功正常轨迹。
每个候选均检查 IK、自然沉降后的实际位置、至少 0.02 rad 关节余量、完整勺体和整臂的 5 mm 净空，
以及到原教师 ABOVE 点的增量 IK 路径。60 秒时限与原任务成功判据保持不变。

```bash
/home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest tools/m5/start_pose_v2/test_start_pose.py -q
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/start_pose_v2/run.py collect
# 中断或小批试采后，核验原绑定并继续：
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/start_pose_v2/run.py collect --resume
/home/minashiki/anaconda3/envs/feedingrobot/bin/python tools/m5/start_pose_v2/run.py audit
```

数据独立保存在 `datasets/single_bean/start_pose_v2/panda/train/`。
证据保存在 `outputs/single_bean/start_pose_v2/collection_001/`：`report.json`、工具快照、起点分布图。
`pending/` 不作为训练数据；任务成功且严格回放通过后才移动到训练目录。
超时、任务失败、初态筛选失败、回放失败单独保存在证据目录的 `rejected/`，附拒绝原因。
每格最多尝试 150 个候选，不足配额保留 incomplete，不能转移配额或放宽安全/时限。

只有报告为 `ready_train_extension` 且独立 audit 通过时才能使用；训练前应重新运行 audit。
独立统计和归一化仅使用合格 train 标签，ActionWindows 验证 200 个回合均有合法窗口。
未来混合旧 V1 与本扩充集训练，需要另外配置数据入口和重新拟合合并 train 的模型归一化。
本工具不训练模型，不修改旧 V1 数据、权重、教师、物理或 M4/M5 冻结状态。
