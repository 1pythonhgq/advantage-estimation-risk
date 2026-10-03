# 全词元 MC 参考优势

运行脚本 `compute_token_reference_advantages_vllm.py`，同时保留同目录的
`diagnostic_reference_utils.py` 和 `reference_request_scheduler.py`。
兼容项目使用的 vLLM 0.4.0.post1，保持原模型权重冻结。

有限 K 的输出是参考估计 z，不是精确真实优势。对下标 i=0,...,T-1 的响应 token：

    s_i = prompt_token_ids + response_token_ids[:i]
    V_j[i] = mean(K independent continuation rewards from s_i)
    V_j[T] = 0
    z_j[i] = r[i] + V_j[i+1] - V_j[i]
    r[T-1] = R; other r[i] = 0

固定 token 后的转移是确定的；gamma=1，无 KL shaping、GAE 或标准化。
每组复用相邻状态价值；两组分别调用生成器，不共享续写。
每组必须满足 `sum(z_j) == R - V_j[0]`，代码自动检查。
只处理输入中每题的 `selected_rollout`，不重新选择或生成公共轨迹。

## 运行

输入 JSONL 旁必须保留轨迹脚本写出的 `.meta.json`；可用 `--metadata-path` 指定。
模型路径、温度、top-p、停止字符串、长度预算、dtype、prefix caching 从该文件读取，
并检查 vLLM 版本一致。相同路径不能证明权重未改变，应使用生成轨迹时同一个冻结 checkpoint。

在已安装项目训练依赖的服务器环境执行（复用项目的 {-2,0,1} 奖励）：

```bash
export OMP_NUM_THREADS=1
python scripts/test_token_reference_advantages.py
python scripts/compute_token_reference_advantages_vllm.py \
  --trajectories-path /root/data/diagnostic_trajectories_smoke_vllm.jsonl \
  --output-path /root/data/token_reference_smoke.jsonl \
  --limit 1 \
  --mc-samples 4 \
  --rounds 1 \
  --reward-mode project
```

先用一条完整轨迹验收，不截取响应的前几个 token（那会错误改变终止状态）。
然后去掉 `--limit 1` 计算全部题目。每条长度 T 的轨迹需要 `2*K*T*M` 次续写。
默认使用一个 vLLM 引擎的连续批处理，同时调度多条轨迹、多个前缀。
通过旧版 `LLMEngine.add_request(request_id, None, sampling_params, prompt_token_ids=...)`
为每个请求设置独立 SamplingParams，然后调用 `step()` 推进；不使用新版 TokensPrompt。

## 并发与速度对照

```bash
python scripts/test_reference_request_scheduler.py
python scripts/compute_token_reference_advantages_vllm.py \
  --trajectories-path /root/data/diagnostic_trajectories_smoke_vllm.jsonl \
  --output-path /root/data/token_reference_parallel.jsonl \
  --limit 2 --mc-samples 4 --rounds 1 --reward-mode project \
  --backend engine --max-pending-requests 16 --max-num-seqs 256
```

`--max-pending-requests` 默认 16，按前缀请求计数；K=4 时最多 64 条逻辑续写分支待调度。
`--max-num-seqs` 是引擎每步的序列容量（默认 256），实际并发仍受 KV cache 和调度器限制。
请求逐题交错入队，完成一个后补入一个；单条轨迹也可以并发多个前缀。
无需同时启动多个模型进程或线程调用同一个 `LLM.generate()`。
从 16 开始；有显存余量可试 32、64，观察生成 tokens/s、GPU 利用率和 preemption/swap 日志。
两种方式使用相同 K、两组参考、长度预算、种子映射与项目奖励，不减少采样来加速。

串行对照加 `--backend serial`，使用另一输出路径；其余输入和参数保持一致。
元数据记录 backend、并发参数、生成 token 数和运行耗时，日志打印生成 tokens/s。
批处理改变 GPU 运算和调度顺序，真实输出不保证逐 token 与串行完全相同；固定同一调度配置复现实验。
测试中的假引擎验证统计组装与路由等价，不证明真实 GPU 数值逐位一致或独立随机数实现。
性能测试需先预热，分别重复测量，不能由并发数推断固定加速倍数。

输出按完成顺序写入，题目顺序可能改变；必须按 `row_index` 对齐不同文件。
每题内部 token、参考组与轮次仍按原始坐标保存。回滚日志亦按完成顺序保存。
日志批量缓冲，每 10 个完成请求或报告进度时 flush，每条完整轨迹完成时也 flush。

## 奖励口径

- 默认 `project`：直接调用项目 `MATHRewardFunction` 与 original-format `GSM8K` 判分。
- `trajectory` 仅保留为旧命令的别名；`binary` 已移除，使用时直接报参数错误。
- `finish_reason=length` 优先记 0；正常结束且多个 `####` 记 -2；
  缺少 `####` 或答案不匹配记 0；答案匹配记 1。
- 预测答案取最后一个 `####` 后的文本，strip、去逗号、lower；标准答案仅 strip、lower。
  使用字符串匹配，不将 `1.0` 与 `1` 判为等价。gold_answer 必须是项目预处理后的最终答案。

旧 JSONL 的 reward 不会改写；输出同时保存原 reward 和本次 terminal_reward。
若保存的奖励与项目重算不符，报错，先按项目规则重新判分，再回滚。
已按 binary 生成的参考结果需重新计算，不能仅修改其元数据。
后续 GRPO、critic 与步骤级 MC 同样使用此奖励口径。
此时 V 是平均终局奖励，不再是成功比例；可能为负。附件中的纯 0/1 描述须随之修正。
本适配器复用当前仓库 Rho GSM8K SPO-chain/VinePPO 的奖励配置，不动态解析自定义 Jsonnet 覆盖。

续写输入直接拼接保存的 token IDs；剩余预算是原 max_new_tokens 减响应前缀长度。
判分会解码完整 prompt+response token 序列，去掉 prompt，再应用文本 stop。
这能识别停止字符串跨越固定前缀与新续写的情况；此时引擎可能多生成若干 token，
但它们不进入判分。文本与 token 不一致的原始轨迹会被拒绝，不重新分词修补。
要求原 prompt 长度加完整响应预算不超引擎上下文窗口，否则报错而不静默减预算。

## 输出与核查

- `token_reference_smoke.jsonl`：每题一行；`rounds[r].groups[0/1]` 分别为两组。
  `values` 长 T+1，`token_advantages` 长 T，与 response_token_ids 逐一对应。
  保存每个前缀的 K 个奖励、请求 seed、标准误、望远镜检验残差。
- `token_reference_smoke.rollouts.jsonl`：逐前缀保存全部续写 token IDs、文本、停止原因及奖励。
- `token_reference_smoke.meta.json`：源数据 SHA256、完整采样配置、奖励模式、运行状态和时间。
- 失败时保留 `.partial.jsonl` 与回滚日志；本版不自动恢复。确认 metadata 的 status 为 complete
  后才使用最终 JSONL。重复运行默认拒绝覆盖，可指定新的输出文件。

参考种子：`group*2**48 + seed + round*2**36 + row_index*2**16 + prefix_length`。
group 为 1、2；其保留空间与现有 seed+row、seed+1000000+row 分离。
后续步骤级估计器须使用独立空间，不能使用这两个参考空间。
不同组输出可以自然重复；不得为了得到不同文本而重采样或去重。
独立随机数生成器的隔离并不意味着文本必不相同。

标准误基于样本方差/K，K=4 时很不稳定，全部成功/失败可能给出 0，不能当作无误差证明。
相邻 token 的优势共享价值估计，存在相关性；最终比较仍按题目 bootstrap，保留两组配对。
后续交叉乘积指标必须使用两组各自的 z，不要先合并两组。

测试不加载模型，但奖励测试需要项目训练依赖，直接调用项目判分方法。
覆盖末尾 token、-2 奖励、length 覆盖、字符串判分、望远镜恒等式、seed 空间、跨前缀 stop，
以及假引擎下的全前缀覆盖、剩余预算和两组分离。真实 vLLM 行为仍需服务器 smoke test。
独立的 `test_reference_request_scheduler.py` 无需训练依赖，覆盖跨题并发、乱序/中间输出、
并发上限、队列补充、串行/并行等价及异常清理。
