# M5 纠偏示教数据 v2

本目录保留历史示教数据的读取、标签审计和控制器解释代码。
这里的 v2 是数据／教师版本，与已清理的 MinaDiTV2 模型无关。
旧采集、校准、训练和评估入口及其运行命令已移除。

保留文件：

- `controller.py`：高度保持、接管几何与稳定资格判据；供后续数据版本复用。
- `corpus.py`：`check_labels`、`pool_indices`、`audit_corpus` 与 `MixedV2`，用于教师所有权审计和三池窗口读取。
- `__init__.py`：包入口。

原数据保存在 `outputs/single_bean/v1/m5/dit/correction_v2_dataset_001/data/`，当前有 10 条接收回合。
原报告状态为 `quota_incomplete`：P1 动量组为零，不能把现有数据宣称为满足 12 条配额的完整语料。
原始报告、哈希、split、seed、动作所有权及回放绑定保持，不能修改状态或配额绕过审计。

合法监督仅包含完整教师动作；DP 前缀、扰动与外部制动不作为标签。
动作窗口按 alignment／transition／aligned 三池读取，原混合读取器使用 75% M4、25% 扩展的规则。
该规则仅描述历史读取器；当前 MinaDiTV4／V5 配置使用 M4 与 start_pose_v2，不加入此数据。
