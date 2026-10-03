# 项目原始优势估计：GRPO / VinePPO / SPO-int5

本版替换旧版 `partition_only_all_boundaries`。切分、回滚请求、价值聚合、优势计算直接复用项目代码，SPO 概率掩码按用户原先的明确要求关闭。输出是 episode generator 产生的优势，不包含 PPOTrainer 后续按训练 batch 白化的系数。

## 实际调用关系

| 内容 | 项目方法 |
|---|---|
| 轨迹重新分词及 offsets | `EpisodeGeneratorWithRewardFunction._tokenize_trajectory` |
| VinePPO 字符切分 | `GSM8K.split_solution_into_intermediate_steps` |
| VinePPO 回滚字符前缀及步数上限 | `VinePPOEpisodeGenerator._get_value_estimation_queries` / `_create_step_query` |
| VinePPO 尾部价值补全、步骤优势、token 广播 | `_compute_step_advantages` / `_compute_token_advantages` |
| SPO 概率候选和 int5 | `MathEpisodeGeneratorWithMCAdvantages._get_response_probs_and_cutpoints` |
| SPO 回滚前缀 | `_create_cutpoint_query` |
| SPO 首段基线、终局 score、反向广播 | `_compute_token_advantages(..., apply_probability_mask=False)` |
| GRPO | `MathEpisodeGeneratorWithGroupAdvantages._compute_group_advantages` |
| 两种方法的 MC 奖励均值及 SPO 标准差 | 各自 `_compute_mc_value` |
| 回滚答案抽取 | `IdentityWithSolutionPrefix.extract_from_node` |
| 续写长度预算 | `EfficientIIDExpander._compute_max_tokens` |

为避免诊断脚本复制训练算法，这次在两个项目源文件中做了保持默认训练行为的重构：把 VinePPO 的请求枚举和 SPO 的切点计算提取为共用方法；给 SPO 的概率掩码新增默认值为 `True` 的关键字参数。训练调用不传参数时仍执行原概率掩码；诊断显式传 `False`。没有用修改后的新公式替代原训练公式。

## 已恢复的行为

VinePPO 保留全部原始字符步骤及步骤编号，不把字符边界吸附到 token 边界，不用 `set()` 合并没有独立 token 归属的步骤。回滚 query 是项目在该字符边界截取的文本。token 优势仍由项目按照起始字符归属步骤。当前 Rho GSM8K 配置继承 `max_step_for_value_estimation=25`：估计 query 状态及至多 25 个中间步骤结束状态；更后面的价值通过项目从终局向前补全，保留由此产生的尾部零优势。

SPO 仍使用 `np.where(np.exp(logprobs)<0.9)[0] - 1` 后的 `[::5]`，不额外添加 s0 回滚。首个切点不是 -1 时首段为零；没有切点时不进行 SPO 回滚、全部优势为零；首个切点是 -1 时以其 MC 值作为基线。末端放置轨迹 score，相邻已知值做差，再按项目反向广播。仅移除高概率 token 置零的掩码。

GRPO 第 0 轮复用现有 G 条轨迹奖励，标准差采用 `np.std` 的总体标准差，epsilon 为 `1e-8`。输出标准化和未标准化两种向量。多轮诊断保持选中轨迹固定，从第 1 轮起独立采样其他 G-1 条。这是原实验的重复估计设计，不是重新执行完整 RL 训练。

同一方法、同一轮中 query 文本相同的价值请求复用一次采样，保留每个原始 value_index 的映射，符合项目的 query 去重行为；不同方法、不同轮次、两组参考不会共享估计器样本。

## 配置、引擎和奖励

直接用项目依赖 `_jsonnet` 解析 `polIter_rho1bSft2_vineppo_GSM8K.jsonnet`、`polIter_rho1bSft2_spo_chain_GSM8K.jsonnet` 和 `episode_generators/interval5.jsonnet`。当前值为 temperature=0.6、top_p=0.9、MC=9、每次续写 max_tokens=1024、model_context_size=2047。元数据与配置不一致时停止；不会猜测或静默覆盖。

长度实际由项目函数计算：

```python
min(1024, 2047 - len(tokenizer.tokenize(query_text)))
```

它与旧版的 `1024 - 已有响应前缀长度` 不同。也不能用实际 vLLM 模型的 2048 直接替代配置中的 2047。编码后的请求若超过引擎实际窗口则报错，不再次缩短预算。

VinePPO 使用字符前缀文本，SPO 使用项目解码 token 前缀得到的文本。将该文本按 vLLM 文本 API 的默认 `tokenizer.encode(query_text)` 编码后提交给同版本 vLLM；不再强制这些回滚请求必须等于原始轨迹的 token 切片。保存实际 query_text 和送入引擎的 IDs 供检查。该变化遵循用户最新的“按照项目逻辑”要求，取代旧版固定 token 前缀协议。

训练口径的 SPO 概率依然通过实际 `PPOTrainer._forward_pass_actor` 计算，使用 BF16、FlashAttention2、关闭 dropout；logits 转 float32 并除以 temperature 后做全词表 softmax，没有 top-p 重归一化。HF 仅做概率前向，全部生成使用 vLLM，准备与生成分两个进程。

MC 判分用 vLLM 返回的文本，由项目答案抽取器从完整 query+completion 中抽出解答，并调用项目 reward 和 MC 聚合函数。不再自行解码生成 ID 后重建答案，也不自行把跨固定前缀的 stop 字符串重新解释为终止。项目奖励仍为长度截断 0、多答案标记 -2、错误/无答案 0、正确 1。原组奖励会重新核实，发现不一致时报错。

## token 对齐和已有参考优势

当前项目 `_tokenize_trajectory` 对完整文本重新分词，追加 BOS/EOS 的代码实际被注释。不能根据配置中的 append_eos=true 就在诊断结果里人为加 EOS。

准备阶段核对项目重新分词与保存 IDs：完全相同则直接对齐；若唯一差异是原始采样多出一个不可见的末尾 EOS，按项目不保留该 EOS，输出明确的 `reference_token_indices` 与 `excluded_reference_token_indices`。其他无法一一对齐的差异报错，绝不按新 IDs 静默复用旧参考。共同样本中切分器失败时也报错，避免训练代码原有的跳过行为悄悄改变诊断题集。

**这些索引只证明 token 的对应关系，不证明已有参考优势的估计目标相同。** 若旧参考按 `1024-prefix_length` 回滚，则与本版项目预算不同。若去除了末尾 EOS，终止状态的位置也发生变化，简单丢掉参考向量的 EOS 项不能修复最后一个可见 token 的参考优势。要声称参考与项目估计器使用严格一致的协议，需要按对应预算和终止定义重新计算参考；不能直接把旧文件按下标拼接后当作已完成的一致性验证。本次改动没有替换或重新计算已有参考文件。

## 需要同步的文件

将下列文件按原路径同步到服务器 `/root/SPO`：

- `scripts/project_method_adapter.py`（新增，替代已删除的 `method_advantage_utils.py`）
- `scripts/prepare_method_partitions.py`
- `scripts/compute_method_advantages_vllm.py`
- `src/treetune/episode_generators/vineppo_episode_generator.py`
- `src/treetune/episode_generators/math_episode_generator_with_mc_advantages.py`

保留之前已有的 `scripts/diagnostic_reference_utils.py`、`scripts/compute_token_reference_advantages_vllm.py`、`scripts/reference_request_scheduler.py`。不要只同步主脚本而漏掉两个项目源文件。

旧版切分和方法估计必须重算；脚本通过 schema=`project_methods_v2` 拒绝误用旧结果。一般可以复用已有采样轨迹文本和 IDs，不需要重新抽取诊断轨迹。输出文件必须使用新路径。

## 运行

```bash
export OMP_NUM_THREADS=1
python /root/SPO/scripts/prepare_method_partitions.py \
  --trajectories-path /root/data/diagnostic_trajectories_smoke_vllm.jsonl \
  --output-path /root/data/method_partitions_project_v2.jsonl \
  --limit 2

python /root/SPO/scripts/compute_method_advantages_vllm.py \
  --trajectories-path /root/data/diagnostic_trajectories_smoke_vllm.jsonl \
  --partitions-path /root/data/method_partitions_project_v2.jsonl \
  --output-path /root/data/method_advantages_project_v2.jsonl \
  --limit 2 --mc-samples 9 --rounds 1 \
  --max-pending-requests 16 --max-num-seqs 256
```

仍支持同一 vLLM 引擎的并发调度与 `--backend serial`。更改并发批次可能产生浮点采样差异，不承诺相同种子下逐位相同。脚本保存实际随机种子、原始 completion、每个 value_index 的去重映射和奖励，便于审计。

结果仍从 `record['rounds'][r]['grpo'/'vineppo'/'spo_int5']['token_advantages']` 读取。VinePPO 另存原始 sampled_values（含未采样位置 None）、filled_values、segment_advantages、token_step_indices；SPO 另存 cutpoints 和 cutpoint_values。SPO 的 token_values 非切点处 None 是项目原有的稀疏中间结构，不是缺失优势；最终 token_advantages 必须全部是有限数。

结果按完成顺序写入，以 row_index 连接，核对原始 IDs、项目 IDs 与映射；不要按文件行号连接。旁边的 meta 文件记录项目解析配置、源码指纹、运行状态和生成 token 数。去重请求的 token 数只计一次，记在其首个目标轨迹，避免总成本重复计算。

## 验证

```bash
cd /root/SPO
python -m unittest discover -s scripts -p 'test_method_advantages.py' -v
```

CPU 测试需要 numpy，直接执行从项目源码提取的实际方法，覆盖跨 token 字符边界、没有 token 归属的中间步骤、25 步上限与尾部补全、SPO 首段和空切点、默认掩码保持训练行为、负奖励、EOS 对齐、原始答案抽取、项目续写预算、请求去重、乱序并发与随机流隔离。当前本地环境没有可用 Python，尝试运行测试返回 `No installed Python found!`；测试及 GPU 两题运行均尚未验证通过。
