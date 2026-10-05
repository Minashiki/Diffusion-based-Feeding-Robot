# 单豆 M4 状态入口

M4 当前完成状态以匹配输入的 `outputs/single_bean/v1/m4/revision_3/panda/report.json`、`ur5e/report.json`、`freeze_manifest.json` 和 `acceptance_audit.json` 为准；教师门禁与数据完成分别报告。正式数据位于 `datasets/single_bean/v1/m4/panda/`。

教师、观测、采集与重放契约见 [M4 接口](m4_interfaces.md)，执行顺序见 [实施步骤](m4_execution_plan.md)。调试证据保存于 `outputs/single_bean/v1/m4_tuning/`，保留失败尝试与当时源码；不把调试成功率当独立验收成绩。

原单豆 M1/M3 冻结记录保留。旧盘子／15 豆代码及相关输出、标定和诊断报告已清理；来源校验依赖的源资产与许可证保留。M5/M6 训练不属于本轮 M4。

修订 2 原独立基准 Panda 100/100、恢复 10/10，收敛 19/20；仅一个恢复半步长交付确认差异 1.061 秒超出原 1 秒容差。用户授权修订 3 保留此教师并采用 M4 接触确认 1.1 秒容差，其他指标不变。新报告完成前仍未放行，原失败审计保留。
