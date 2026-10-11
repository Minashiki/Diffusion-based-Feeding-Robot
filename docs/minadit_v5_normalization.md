# MinaDiTV5：DP／DP3 归一化调研与实现契约

2026-10-09。V5 在 V4 主候选上增加动作及连续观测的逐维拟合归一化，
推理改为 DDIM `trailing / 100`。本文件记录论文、指定源码与本项目数据约束的对应关系；
不代表 V5 已正式训练、通过闭环验收或产生可部署权重。

## 论文与源码依据

Diffusion Policy 的 [RSS 2023 论文附录 A-A](https://arxiv.org/html/2303.04137v4#A1.SS1)
及 [IJRR 2024 论文附录 A.1](https://arxiv.org/html/2303.04137v5#A1.SS1) 都要求
将每个动作维度的最小值、最大值独立映射至 `[-1,1]`。
论文解释：DDPM 对预测干净样本的裁剪与该范围配套，零均值、单位方差的 Z-score
可能让部分合法动作落在裁剪范围外。近常值维度只平移、不放大小变化；
论文另说明部分旋转表示（例如四元数）保持原样。

DP3 的 [论文 v7 附录 Implementation Details](https://arxiv.org/html/2403.03954v7)
中 `Normalization` 段明确写出：动作每维与观测每维均独立以 min/max 映射至 `[-1,1]`。
这直接支持本轮已经确认的“动作及连续观测都拟合”。论文和网络内部的 LayerNorm
是不同层面的处理，V5 只改变数据的仿射预处理。

本轮核对的源码版本固定为：

| 上游 | 提交 SHA | 指定实现 |
|---|---|---|
| Diffusion Policy | `5ba07ac6661db573af695b419a7947ecb704690f` | [normalizer.py](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/normalizer.py) |
| DP3 | `47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97` | [normalizer.py](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/model/common/normalizer.py) |

两份文件的 `_fit` 和 `_normalize` 函数经 AST 比较完全相同。
因此数值规则按用户指定的这两个函数照搬，而不是仅实现一个大致相似的 Min-Max 变换。

## 精确数值规则

采用 `LinearNormalizer.fit`／`SingleFieldLinearNormalizer.create_fit` 的默认 limits 路径：
`dtype=torch.float32`、`last_n_dims=1`、`mode='limits'`、`output_min=-1`、
`output_max=1`、`range_eps=1e-4`、`fit_offset=True`。
先转 FP32，再将输入重排为 `(-1, D)`，沿第 0 轴分别计算每个维度的
`min`、`max`、`mean`、`std`。`std` 使用上游 `torch.std` 的默认样本标准差
（`correction=1`）；它作为统计保存，不参与 limits 的尺度计算。
[DP 源码 `_fit`](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/model/common/normalizer.py#L182-L261)

设某一维的原始最小值为 `m`、最大值为 `M`，规则为：

```text
range = M - m
if range < 1e-4:
    scale = 1
    offset = -m
else:
    scale = 2 / range
    offset = -1 - scale * m

normalize(x)   = x * scale + offset
unnormalize(y) = (y - offset) / scale
```

必须保留乘法及加法的顺序。范围判断是严格 `< 1e-4`，等于阈值仍缩放。
常值 `c` 映射为零；近常值仍保留原微小差值，不强制所有样本为零。
例如 `[5, 5.00005]` 进入近常值分支，输出近似 `[0, 0.00005]`。
论文文字说“平移至零均值”，但指定源码在此分支实际减的是 `min`。
V5 遵循源码的 `offset=-min`，不改成减均值，也不使用把范围强行钳到 `1e-4` 的替代写法。

归一化器自身没有 clamp；超出拟合范围的值允许超过 `[-1,1]`。
上游 `_normalize` 将输入转到参数 dtype/device，保留原始形状；
拟合参数和输入统计被冻结并保存到 `params_dict`。
[DP3 源码 `_normalize`](https://github.com/YanjieZe/3D-Diffusion-Policy/blob/47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97/3D-Diffusion-Policy/diffusion_policy_3d/model/common/normalizer.py#L264-L278)

V4 的 `scale` 是物理半范围，公式为 `(physical-offset)/scale`；
V5 的 `params_dict` 使用上游定义，`scale` 是乘法系数、`offset` 是加法偏置。
正常维虽然数学上可互换，参数名称的语义与 FP32 运算顺序不同，不能直接套用 V4 的 JSON。

## 上游数据集的区别

DP 的 [RealPushTImageDataset.get_normalizer](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/real_pusht_image_dataset.py#L155-L171)
对动作及所有 low-dimensional 观测分别调用 `create_fit`，随后才采样历史窗口；
图像使用固定范围变换。这个路径最接近本项目的动作加连续观测拟合。

上游并非每个数据集都使用相同例外规则。
[RobomimicReplayLowdimDataset](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py#L88-L130)
中，已经归一化的相对动作采用 identity，绝对动作可以只拟合位置而保留旋转和夹爪；
该数据集的连续观测还使用整体最大绝对值缩放。
这些是数据集专用策略，不能称为指定通用 `normalizer.py` 的逐维默认算法。

RealPushT 的 `get_normalizer` 直接读取整个 `replay_buffer`，没有使用
训练／验证 episode mask 筛统计。V5 保留本项目既有的 train-only 边界，
仅从实际训练源数据拟合，validation／test 只应用冻结参数。
因此这里照搬的是**逐维拟合算法及仿射语义**，拟合数据范围按本项目的训练划分约束收窄；
不会声称与上游 full-buffer 数据选择完全相同。

## V5 字段与拟合样本契约

| 输入 | V5 处理 |
|---|---|
| 六维动作：线速度、角速度 | 每个维度独立拟合 limits |
| 连续物理观测：位置、相对位置、力／力矩、关节位置／速度、TCP twist 等 | 每个物理维独立拟合 limits |
| 观测中的旋转矩阵元素 | 每个元素独立拟合；沿用 V4 的矩阵表示 |
| 派生目标线／角速度 | 在原始物理量上先计算，再逐维拟合 limits |
| state／history 帧龄 | 分别从合法查询的有效原始帧龄拟合 limits |
| `stage`、`interaction`、`execution_status` | identity，类别列保持原编码 |
| 有效 mask、布尔标志及 padding | 保持原语义；无效输入行按原规则清零 |

本项目旋转观测是作为条件输入的矩阵元素，不是待生成且需要单位范数约束的四元数动作。
本轮不变换旋转表示。按已经确认的“连续观测全部拟合”，矩阵元素也包含在拟合范围中；
这是本项目字段选择，不等于照搬 DP 中保留四元数动作的特例。
目标速度和角速度的几何计算必须使用原始姿态，不能用归一化后的矩阵做旋转差分。

`last_n_dims=1` 意味着 `(N, T, D)` 的时间轴被并入样本轴，
每个物理维在全部时间位置共享参数。state 与 history 中同一物理字段也共享统计；
不能先展平为 `(N, T*D)`，再为每个历史位置单独拟合尺度。

`fit_dp_limits` 从 V4 的 `pre_action` 原始 train 读取器取得统计：

1. 动作仅统计有效合法起点对应的原始动作行；不重复累计 16 步重叠 horizon 中的未来动作。
2. 物理观测统计实际被合法窗口的 state／history 查询选中的有效源帧。
   同一 episode、同一源帧只计一次；不让历史重叠和窗口重复给该帧增加权重。
3. 派生目标速度按合法条件查询在原始输入上计算，保留既有可用性规则；
   不把同一个速度向量在两个 state token 的复制当成两条独立样本。
4. 帧龄由 `query_tick-source_tick` 决定，必须从对应有效查询特征取得；
   同一源帧在不同查询下的帧龄可以不同，不能简单按源帧去重后推算帧龄统计。
5. 排除无效行、填充和 held-out 数据。保留冻结基础 train 及原 start-pose 扩展，
   不改数据文件、episode 划分、观测边界恢复或合法窗口规则。

Min/max 对重复样本可能不敏感，但 mean/std 会改变，因此保存这四项统计时，
样本去重与有效性选择同样是需要验证的实现部分。

## 保存、训练与推理

归一化记录使用 `kind='dp_limits'`、`schema_version=5`，
各拟合组在 `params_dict` 保存 FP32 拟合得到的 `scale`、`offset`、
`input_stats.min/max/mean/std`，同时记录 train 来源、观测 schema 和各字段拟合行数。
上游提交版本固定在本文；运行 checkpoint 另绑定本地源码哈希和数据来源。
`normalization.json` 与 checkpoint 绑定同一记录。推理只加载绑定参数；
不根据当前场景重新拟合，不让 validation/test 更新参数。

沿用 [DP 训练与推理顺序](https://github.com/real-stanford/diffusion_policy/blob/5ba07ac6661db573af695b419a7947ecb704690f/diffusion_policy/policy/diffusion_unet_lowdim_policy.py)：
训练先归一化干净动作和条件，之后扩散加噪，模型预测归一化 `sample/x0`；
推理先归一化条件，完成 DDIM 后反归一化动作，再交给原物理控制限幅。
`clip_sample=True` 继续裁剪归一化预测干净动作；连续观测预处理不增加裁剪。
逐维动作裁剪不等于线／角速度向量范数限幅，原控制限制继续生效。

| 配置项 | V5 |
|---|---|
| 数据与观测时间 | V4 的基础 train、start-pose 扩展及 `pre_action` |
| 网络 | 6 层、hidden 512、8 heads、MLP 2048、horizon 16，沿用 V4 |
| 扩散训练 | cosine / 100、`prediction_type='sample'`、`clip_sample=True` |
| 推理 | DDIM `timestep_spacing='trailing'`、`inference_steps=100` |
| 精度与优化器 | 沿用 V4 的训练／推理 FP32 及优化器、EMA、验证策略 |
| 省略 `--model`／`--config` 的默认行为 | MinaDiTV5；V4 需要显式选择 |

DP3 论文实际使用训练 100 步、推理 10 步；V5 的 `trailing / 100` 是本轮用户指定设置，
不能作为照搬论文推理超参数的声明。
重新拟合输入／动作参数改变了模型看到的表示，V4 或更旧权重不能直接作为 V5 续训检查点。
首次 V5 训练应从头初始化；同一次 V5 运行续训则必须复用原 checkpoint 绑定的统计。

在项目根目录显式启动 V5，并使用新输出目录：

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m feedingrobot.scripts.train_dp \
  --config configs/minadit_v5.json --model MinaDiTV5 \
  --output outputs/single_bean/v1/m5/dit/MinaDiTV5/run_001 \
  --device cuda --updates 100000 --headless
```

上述是执行入口，不是已经完成训练的记录。

## 已完成验证及后续工作

对两份固定上游源码的原始 `_fit`／`_normalize` 直接执行 76 组数值对照，
覆盖 NumPy／Torch、FP32／FP64／整数、二维至四维、非连续输入、常量、近常量和阈值边界。
尺度、偏置、四项输入统计、归一化及逆变换均逐值完全一致。
参数契约另外通过 2,000 份随机完整 schema、10,000 组字段压力检查；
20 次一个 ULP 的偏置篡改均被拒绝。此对照在 CPU 上执行。

归一化实现时的模型与数据管线回归为 **88 passed，3 skipped**，跳过项需要 CUDA；其中包含现已移除的旧模型测试。
其中新增 V5 测试 11 项，覆盖固定上游数值基准、共享观测参数、类别与 mask、
有效源行及 held-out 排除、start-pose 扩展、归一化动作往返、完整 `99..0` 的 100 次推理、
外层 autocast 下的 FP32，以及连续四步训练与两步加恢复两步的模型／EMA 完全一致。
恢复测试同时确认不重新拟合统计。小模型训练仅用于管线验证。

真实数据预检已完成：冻结 M4 父数据及 start-pose 扩展审计通过，
基础与扩展各 200 个 train 回合，共 **338,555 个合法动作起点**，全部完成 V5 拟合与参数契约检查。
基础有 113,136 个窗口，扩展有 225,419 个；历史传感字段按源帧去重后每维统计 1,020,901 行，
其余物理观测每维统计 340,427 行。统计与审计报告另存于
`outputs/single_bean/v1/m5/dit/MinaDiTV5/preparation_001/normalization.json` 和 `report.json`；
没有更新权重或改写冻结数据。

```bash
PYTHONPATH=src /home/minashiki/anaconda3/envs/feedingrobot/bin/python -m pytest \
  tests/test_minadit_v5_normalization.py tests/test_minadit_v5.py \
  tests/test_dp_normalization.py tests/test_minadit_v4.py \
  tests/test_diffusion_diagnostics.py tests/test_diffusion_data_audit.py \
  tests/test_m5.py -q
```

正式训练、离线动作质量与 validation 闭环验收仍需随后执行。
当前环境没有可用 CUDA，本轮没有训练完整 V5 权重，也没有进行物理闭环验收。
