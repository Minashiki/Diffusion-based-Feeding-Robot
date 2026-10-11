# M5 纠偏示教数据 v3

本目录保留真实动量制动示教的读取、标签审计和控制器解释代码。
这里的 v3 是数据／教师版本，与已清理的 MinaDiTV3 模型无关。
旧采集、诊断、校准、训练和评估入口及其运行命令已移除。

`controller.py` 复用 `correction_v2.controller`，补充实际动量、制动余量及独立安全停止判据；
`corpus.py` 复用 v2 的窗口池和混合读取器，额外校验 v3 教师版本、速度和高度漂移。
因此 v2 的 controller／corpus 同样保留。目录另保留包入口 `__init__.py`。

原数据保存在 `outputs/single_bean/v1/m5/dit/correction_v3_dataset_001/data/`，包含 12 条接收回合；
原采集报告状态为 `ready_experimental`，不代表模型训练或闭环验收通过。
P1／P2 的停稳与动量各两条，已对齐对照四条；全部训练数据保持原文件、划分及审计绑定。

监督只使用 alignment_teacher／path_teacher 的完整动作，排除 DP 前缀、扰动与外部制动。
保留真实 entry／sweep 衔接、alignment／transition／aligned 三池和历史 75%／25% 混合规则。
v4 数据已包含这 12 条，不应在复用 v4 时重复加入。
当前 MinaDiTV4／V5 配置使用 M4 与 start_pose_v2，不加入此数据；后续复用需核验原报告、哈希和资格。
