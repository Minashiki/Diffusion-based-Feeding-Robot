# M5 纠偏示教数据 v4

**correction_v4 是示教数据／教师版本，区别于 MinaDiTV4 模型。**
本目录保留 `__init__.py`、`corpus.py`、`controller.py` 与本说明。
旧 run、training、evaluation 入口及其命令已移除。

数据保存在 `outputs/single_bean/v1/m5/dit/correction_v4_dataset_001/data/`，
包含 v3 的 12 条原示教及新增 32 条受控方向示教，共 44 条。
新增示教来自真实路径教师前缀及命令扰动，不能称为 DP 前缀或自主制动；
扰动、制动与前缀均屏蔽监督，原文件、划分、回放和审计绑定保持。

`controller.py` 复用 v3／v2 的几何与反馈规则，保留四类别、四象限、toward／away 方向及逐毫秒稳定资格判据。
`corpus.py` 的 `PickupWindows` 读取 alignment／transition／aligned 三池，
覆盖真实路径至完整 pickup_hold，窗口在教师所有权、合法动作及拾豆终点处截尾。
`MixedV4` 保留历史 75% M4／25% 扩展采样；三池等概率，池内依次均衡选回合、教师阶段与窗口。
等待年龄和教师内部阶段只用于索引及审计，不增加策略输入。

保留 v2／v3 的 controller／corpus 是这些数据语义和导入依赖的组成部分。
当前 MinaDiTV4／V5 默认配置使用 M4 与 start_pose_v2，不加入纠偏数据。
后续复用应核验完整文件哈希、原采集报告、标签所有权、split／seed 隔离及回放资格，
并为当前模型另建训练和归一化绑定；不能继承或改写已清理旧模型的 checkpoint 字段。
`ready_experimental` 或零更新审计通过均不代表模型训练、闭环验收或 M5 放行。
