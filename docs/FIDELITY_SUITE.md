# Fidelity Gate v4

Fidelity Gate 只回答一个问题：在明确限定的原版迁移范围内，模拟环境是否有足够、可复算且来源独立的证据。它不决定 PPO 配方，也不授权视觉策略或原版部署；这些由独立的 Training Gate 和后续视觉阶段负责。

当前可授权策略是 `original-transfer-jungle2-v4`，范围严格限定为：

- `ZumaRevenge-v0`；
- 普通难度 `Jungle2`；
- `profile_mode="tutorials_completed"`；
- 单曲线共享 Board；
- 状态型 `actor` observation；
- 原生 100 Hz、`frame_skip=1`。

v1/v2 只保留历史复现资格。v3 首次加入完整 fruit 契约，但错误要求 Jungle2
自然产生权重为零的 slow powerup，属于不可达验收条件。v4 保留其余严格要求并
移除这一项；即使旧报告为 `OPEN`，训练入口也不会把历史策略当作当前许可。

## 四条独立证据链

### 1. Source authenticity

原版来源可以是两类：

- `pc_golden`：两次独立原版回放确实逐 tick 一致时，保留更强的 PC Golden 证明；
- `pc_source`：独立验证一份未修改的原版进程轨迹，不要求另一进程与它像素或 RNG 状态完全相同。

`pc_source` 会重新核验原版 EXE、派生 runtime、DMO、进程创建身份、逐 tick 轨迹、外部输入隔离以及采集前后存档/注册表状态根。两份轨迹即使从第一帧开始进入不同的合法随机分支，也可以分别成为真实来源；重新包装同一份轨迹仍只算一个来源。

当前 C109 与 C110 已分别通过 812 tick 单来源验证。它们的 source fingerprint 不同，且两次主机状态都精确恢复。Gate 不再把两者之间的 RNG/像素差异误判成“原版来源无效”。

固定最低要求：8 个独立原版来源、3 个 gameplay seed。当前全量复算达到
10 个认证原版来源和 36 个独立 seed，因此该 lane 为 `PASS`。

### 2. Dynamics fidelity

每项动力学机制必须同时具备：

1. 真实原版来源中的无注入机制证据；
2. 模拟器对该来源的逐 tick 差分证据。

固定矩阵覆盖发射、弹丸碰撞、前后插入、三/四连消、rollback、gap shot、tunnel、
换球、输入节奏、自然胜负、Zuma 过渡、Jungle2 可达 powerup、pending/rejection
RNG 和完整 fruit 生命周期，并额外要求不少于 500 tick 的长时无漂移窗口。
仅有 `PASS` 字样或 suite 手写 feature 标签不能授权机制；Gate 会从报告正文重新推导。

### 3. Stochastic distribution

跨进程完全相同不是随机系统的合理验收标准。当前 distribution protocol v3
使用预注册的、玩家可见的起始双球颜色分布：

- 代码固定并内容寻址两组互不重叠的 gameplay seed：原版 32 个、模拟器
  256 个；不再允许 suite 自选连续 seed 区间；
- 每个原版槽位从同一个基准 DMO 派生，且只允许 DMO 头部偏移 8–11 的
  `random_seed` 字段变化；基准、派生文件及逐字节 provenance 均须绑定；
- 同一 DMO 在多个自然启动进程中重复播放会被拒绝。DMO 会恢复游戏使用的
  全局 MTRand；随进程变化的 CRT startup seed 与 gameplay 分布无关，不能
  冒充独立 gameplay 样本；
- 原版仍以 natural strict command broker 启动：不覆盖寄存器 seed，不写
  gameplay/RNG 内存；完整 strict replay 和有限的字体缓存服务凭据必须进入
  每个 source 的独立验证报告；
- 指标为 `(current_color_id, next_color_id)`，共 16 个低基数类别；
- 经验总变差 `TV <= 0.35`；
- 相对固定置换零分布中位数的超额 TV `<= 0.10`；
- 1,999 次固定种子置换的兼容性 `p >= 0.05`；
- 每个已启动的原版进程都必须保留，观察结果后不得替换样本；
- 32 个 PC 样本 fingerprint 必须全部绑定到同一 suite 中已认证的原版来源。

这样既不会用 0–623 的高基数 MTRand 索引制造稀疏样本假失败，也不会把“未发现显著差异”单独冒充等价证明。阈值和样本数写死在代码中，suite 无法放宽。

### 4. Actor interface

状态策略必须通过 `actor_no_hidden_state` 与 `fruit_actor_observation`：tunnel 内身份、
内部计时器、pending 状态等特权信息不能泄漏进 observation，同时 fruit 的玩家
可见状态必须进入 actor。兼容列表由代码固定，suite 不能自行声明。

`actor_visual_derivability` 不混入当前状态环境的物理 Fidelity Gate。它会在真正
加入像素输入时成为独立视觉阶段 Gate；因此 v4 的 `OPEN` 只代表状态型 Jungle2
环境，不代表视觉迁移已完成。

## 证据和状态

所有 suite 输入都使用规范相对 POSIX 路径和 SHA-256，拒绝路径逃逸、重复 ID、重复路径、重复 JSON 键、NaN/Inf、未知字段和超大报告。状态含义：

- `OPEN`：四条 lane 全部满足固定 policy；
- `CLOSED`：输入有效，但证据不足或某项验收失败；
- `INVALID`：suite、路径、schema 或内容寻址无效。

当前 suite：

```powershell
& .\.venv-win\Scripts\python.exe -m zuma_rl.fidelity_gate `
  D:\ZumaGolden\diagnostics\fidelity-suite-c200-candidate-v24.json `
  --suite-root D:\ZumaGolden `
  --original-root "D:\SteamLibrary\steamapps\common\Zuma's Revenge"
```

当前为 `OPEN`。c218 最新全量复算中：

- Source authenticity：`PASS`，8 个 PC Golden、43 个 PC source evidence、
  17 个认证原版来源、39 个独立 gameplay seed；
- Actor interface：`PASS`；
- Stochastic distribution：`PASS`，1 份冻结分布审计；
- Dynamics：`PASS`，21 个 simulator diff 被接受；
- 固定 policy 的 31 项要求全部通过，missing 与 rejected evidence 均为 0；
- 冻结报告：`fidelity-gate-c218-candidate-v28-report.json`，SHA-256
  `a08d64bfd13bdbfdcb74c65163aa9e9b6a004e9d8c8fc5ff35376d252655c7d1`。

此 `OPEN` 只授权当前状态环境进入独立 Training Gate；它本身不决定训练配方。

实现入口：

- `src/zuma_rl/fidelity_gate.py`：固定 policy、四 lane 汇总和总 Gate；
- `src/zuma_rl/pc_source.py`：单份真实原版 source 验证；
- `src/zuma_rl/distribution_fidelity.py`：预注册分布兼容性复算。
- `tools/manage_startup_distribution.py`：冻结 seed 日程、逐槽派生 DMO、生成
  采集任务和采集前验证报告。
