# 新餐具 M4 重建状态

M1/M3 v2 已放行，原证据与 v1 M1 保留。M4 的几何教师、独立验收与采集门禁、默认桌面显示和训练显示预算判定已实现；真实取餐尚未通过，教师参数及随机范围未冻结，`teacher_gate=not_verified`，正式数据集未生成，M5/M6 未启动。

## 已实现

- reset 后复制实际盘面、勺唇、下凹承载网格、全餐具包络、食物尺寸、关节范围和口腔规范；教师仅接收只读几何与当前／过去策略观测，输出原六维基座 twist。
- 以初始食物位置限定弧形取餐行程，真实承托反馈驱动转勺与抬升；保留原执行链、M3 事件、保护和观测维度。运输、交付及恢复路径为待验证候选。
- Panda 正常 ≥95/100、恢复 10/10；UR5e 正常／恢复各 5/5。每机器人正常／恢复各 5 个收敛用例预先固定，全部验收尝试命令重放，双机器人成功 viewer、M1/M3 与 pytest 共同决定教师门禁。
- 首批正常／恢复各 train/validation/test=100/15/15；失败尝试保留并重放，配额不足仍执行已有尝试的重放，完整版本和 split 检查通过才放行数据。
- 教师验收和采集默认独立 MuJoCo viewer，30 FPS、单帧队列；阶段／结果叠加，窗口关闭后执行继续。训练预算接口预热后测三组等工作量窗口，中位耗时比 >1.5 关闭显示并保存测量。M5/M6 的实际训练入口及实际训练开销测量待后续接入。

## 验证及失败证据

- 全量 pytest：270 通过；日志为 [pytest.log](../outputs/calibration/new_tableware/m4/rebuild_check_01/pytest.log)。后续修订定向回归 [59 项通过](../outputs/calibration/new_tableware/m4/rebuild_check_01/targeted.log)；采集失败配额重放及观察工具等另有 [52 项通过](../outputs/calibration/new_tableware/m4/rebuild_check_01/targeted_extra.log)，与前述用例部分重叠。
- [固定场景报告](../outputs/calibration/new_tableware/m4/rebuild_check_01/report.json)：候选路径离线可达通过；第一个固定回合运行 60 秒，仍停留 ACQUIRE，无 pickup，时间截断。因此校准在取餐步骤失败，后续独立验收未执行。
- [真实命令桌面重放](../outputs/calibration/new_tableware/m4/rebuild_check_01/native_replay.json)：3000 帧实际状态，观测与物理参考误差均为 0，事件和结局一致。该回合失败，不能替代成功 viewer 验收。
- [冻结输入](../outputs/calibration/new_tableware/m4/rebuild_check_01/frozen_inputs/input_hashes.json) 保留该调试版本。后续采集检查修订改变输入哈希，因此此历史报告不能直接用于当前正式采集。

诊断目录：`outputs/calibration/new_tableware/m4/`。`pivot_probe_01` 为有限转勺并提前抬升：食物留盘，未产生 pickup；`roll_floor_incremental_01`、`center_roll_01`、`later_capture_01`、`shallow_arc_01` 为已否决的贴盘转勺候选，食物穿入薄盘底并触桌，M3 判定 food_dropped。这些原始 trace/result 为诊断证据，不是正式示范或验收成功率。

[姿态碰撞诊断](../outputs/calibration/new_tableware/m4/pose_constraints.json) 显示低位水平承载目标受腕部—盘沿约束：独立 IK 的 Panda TCP 高度误差约 20.7 mm、UR5e 约 50 mm，最近障碍为盘沿。已测方向及倾角网格保存在 [feasibility_grid.json](../outputs/calibration/new_tableware/m4/feasibility_grid.json)。结论范围仅限已测路径族。

## 停步与环境变更提案

按主方案遇到几何不可行证据的停步规则，当前暂停运输、完整教师放行和正式采集。下一项建议先独立验证工具安装连接段的长度是否足以在盘上实现承载姿态；若需延长，仅调整连接段，并以实测碰撞余量确定尺寸，重新验证质量、惯量、负载补偿及双机器人 M1/M3。真实勺头曲面保持。

另需独立复核薄盘底在舀取接触下的支撑稳定性：已否决试验出现约 3.2 mm 穿盘接触。盘底碰撞厚度或接触参数若需修改，单独形成环境变更并重新执行 M1/M3；事件容差和保护阈值保持。上述环境变更尚未实施，不能用修改成功标记或关闭保护推进 M4。
