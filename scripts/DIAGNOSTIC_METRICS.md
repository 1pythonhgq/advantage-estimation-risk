# 两个诊断指标：总体风险 R 与切分近似误差 A

实现文件：`scripts/compute_diagnostic_metrics.py`。只使用 Python 标准库，在 CPU 上读取现有 JSONL，不加载模型、不做回滚、不需要 numpy 或 GPU。当前支持 GRPO、VinePPO、SPO-int5，以及 GRPO 未标准化对照；没有 PPO Critic 的优势结果，因此不虚构 PPO 指标。

## 定义

每题有 M 轮，每轮同一条轨迹上包含方法估计 a，以及两组独立参考 z1、z2。在每轮、每个词元先计算乘积，再对词元及轮次求平均：

```text
R = mean((a - z1) * (a - z2))
A = mean((z1 - 所属段的z1均值) * (z2 - 所属段的z2均值))
E = R - A
```

R 是总体优势估计风险；A 是切分近似误差。E 为派生的估计器误差，方便分解。所有量越小表示相应误差越小，但有限样本估计允许为负，不取绝对值、不截断为零。不能将两组参考先求均值后做普通 MSE，也不能先对多轮优势求均值后再算乘积。

分段来源：GRPO 整条轨迹一段；VinePPO 使用结果中的 `token_step_indices`，保留项目真实 token 归属（没有 token 的字符步骤贡献为零）；SPO 由项目原始 `cutpoints` 构造，切点 c 表示段末 token 下标，下一段从 c+1 开始，-1 只表示初始基线。首段、没有切点时的整段都保留，不因为优势为零而删掉，也不把相邻相同优势的段合并。

脚本检查每段方法优势恒定，并验证 R=A+E 的投影分解。VinePPO 的 25 步上限改变其估计优势，不改变用于 A 的原始分段。SPO 概率掩码和批次白化必须关闭。

## 必须明确参考目标与评价范围

默认 `--comparison strict --token-scope full`：要求两边元数据有内容相同、非空的 `target_protocol` 描述，且没有缺失的参考词元。现有生成脚本尚未提供此统一描述，旧参考和最新项目估计的前缀处理、回滚长度也确有区别，所以不能声称默认严格检查已经满足。

已有文件可使用 `--comparison reference-target`：明确把输入的两组参考定义为评价目标，计算项目估计器相对于该目标的风险。R、A 公式不变，但 R/E 可能同时包含切分、MC、首段规则、25 步上限以及前缀和预算协议差异的影响；不能将方法风险差全部解释为切分优劣。A 则只使用相同参考与每种实际分段，不依赖方法 MC 数值。为了研究原项目估计器这是可报告的诊断比较；若要求项目与参考定义同一目标，应先另行统一参考协议。不要手改元数据假装相同。

项目文本重新分词可能不保留采样时末尾不可见 EOS；此时用 `--token-scope project` 明确只评价项目实际输出优势的 token 集合。脚本要求项目 IDs 与原始参考 IDs 的前缀完全一致，仅允许尾部至多一个被准备阶段明确排除的 token，并保存排除索引。**不会给 EOS 编造优势，不会将其参考优势合并到前一个 token，也不会修改原始参考的终局奖励位置。** 这衡量的是原参考目标在项目词元子集上的指标，不是把原参考改造成另一套无 EOS 的目标；分母使用实际被评价的词元数。若全部 tokens 一致，两种范围取值产生相同结果。

## 使用现有结果

将 `compute_diagnostic_metrics.py` 复制到 `/root/SPO/scripts/`。例如原参考命令使用 limit=1，而方法结果有两题，脚本会报问题集合不一致；应提供同一批题的完整文件，不会偷偷取交集。

```bash
python /root/SPO/scripts/compute_diagnostic_metrics.py \
  --references-path /root/data/token_reference_smoke.jsonl \
  --estimates-path /root/data/method_advantages_project_v2.jsonl \
  --output-dir /root/data/diagnostic_metrics_smoke \
  --comparison reference-target \
  --token-scope project \
  --bootstrap 1000 \
  --confidence 0.95 \
  --seed 42
```

输入旁边必须有运行生成的同名 `.meta.json`，两边状态均为 complete。这里不需要轨迹原文件、切分文件或 rollout 审计文件。输出目录必须不存在；不会覆盖旧结果。`seed=42` 只控制 bootstrap，与已完成的模型采样无关。正式数据替换输入与输出路径即可，不改变公式。

## 输出

- `summary.json`：四种输出口径各自 R/A/E、95% bootstrap 区间、所有方法间配对差、输入哈希、协议解释、排除词元数与种子检查。
- `summary.csv`：可用 Excel 打开的汇总表。`question_mean` 是主结果，`token_weighted_mean` 是次要结果，置信区间对应主结果。
- `per_question.jsonl`：每题每轮指标、题内多轮均值、实际评价词元数、段数、原参考映射与排除索引。

主结果先在题内平均 M 轮及 T 个词元，然后题目等权平均。长轨迹不会在主结果中占更大权重。bootstrap 每次按题目有放回抽样，并将该题全部轮次、两组参考和各方法一起保留；所有方法共用同一组重采样索引。配对差定义为 left-right，负值倾向 left。所有两两差都输出，没有事后选择“较差方法”，也没有多重比较校正。

至少两题才给 bootstrap 区间，一题的区间为 null。两题 smoke 的区间主要用来检查流程，不能作为正式统计结论。

GRPO 标准化与未标准化结果分别叫 `grpo`、`grpo_unstandardized`，它们的 A 应相同，因为切分都是整轨迹一段；R/E 可以不同。比较优势尺度时同时报告二者。

## 输入检查

按 row_index 连接，检查题集、source_index、selected_index、prompt IDs、response IDs、文本、终局奖励、轮次、MC 样本预算。两组参考必须齐全；检查价值等于 reward_samples 的均值、逐 token 参考优势等于相邻价值差加终局奖励。检查参考请求种子彼此不重复，且不与估计器请求重叠。项目内部同方法同轮 query 去重允许共享估计器种子。种子检查是记录层证据，不能独立证明 RNG 实现的统计独立性或模型目录未被替换。

JSONL 逐物理行读取，保留 GSM8K 文本中的 U+2028。遇到 NaN、非有限优势、长度错误、重复题目、缺轮次、非恒定段优势，直接报错，不用默认值补齐。

## 测试

复制 `test_diagnostic_metrics.py` 后运行：

```bash
cd /root/SPO
python -m unittest discover -s scripts -p 'test_diagnostic_metrics.py' -v
```

测试覆盖手算指标、负值、投影恒等式、int5 下标转换、空 token 步骤、跨轮乘积计算顺序、乱序对齐、EOS 范围、协议冲突、参考污染、配对 bootstrap、JSONL Unicode 以及 CLI 写出。当前 Windows 工作区没有安装 Python；尝试运行返回 `No installed Python found!`，尚未宣称测试通过或实际计算出了实验结果。
