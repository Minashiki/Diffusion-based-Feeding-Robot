# v3 专用入口交付验证

本轮仅实现专用训练与可视化闭环入口，没有训练实际 100k 模型，没有运行训练后策略验证、正式 test 或冻结。M5 保持 incomplete。

- 最终版本预检通过：`outputs/single_bean/v1/m5/dit/correction_v3_entry_preflight_001/report.json`，`status=preflight_passed`、`optimizer_updates=0`、`model_executed=false`、`source_unchanged=true`。原 v1/v2/v3 工具绑定、数据、配对校准、零更新审计和归一化均通过核验。
- 三池读取保持 alignment 8 回合 / 119 窗口、transition 8 / 1147、aligned 4 / 840；训练采用 75% 原数据与 25% v3 三池，外部制动标签仍排除。
- 宿主只读计算探针通过：RTX 5080、Torch 2.11.0+cu130、CUDA 13.0、BF16 支持；有效亲和性为逻辑核 0–5，Torch CPU threads=6、interop=1。没有创建模型或优化器。
- 宿主 MuJoCo 图形烟测通过：恢复独立校准种子 780003 的低速状态，运行 1.5 秒固定零动作诊断后缀；显示进程正常完成，74 帧发送并显示，`physics_isolated=true`，随后精确回放通过。它验证图形与记录入口，不是模型能力证据。沙箱不能访问桌面 DISPLAY，实际可视化须在宿主桌面执行。
- 宿主新旧相关回归 **118/118 通过**，耗时 134.63s，包含新增 23 项和原 95 项，未排除 DataLoader spawn/mmap 检查。更新测试只训练随机小模型，验证 100k EMA 起步语义、优化器/RNG 恢复及连续/中断结果一致性；实际 100k 权重未更新。真实物理诊断覆盖低速后缀及高度超限后安全制动，两者均精确回放通过。

操作命令见 [README.md](README.md)。训练完成后使用 `evaluate.py small`，默认逐场打开 MuJoCo 窗口并实时显示；先比较自主纠偏、真实拾豆、运输、交付和安全制动介入，再决定扩大到完整 validation。
