# AlphaZuma-55 未攻克关卡病因定位裁定书 (2026-09-01)

| 项 | 值 |
|---|---|
| 文件 | `diagnostics/alphazuma-55-unconquered-cause-localization-20260901-VERDICT.md` (新建; 不修改任何既有文件) |
| 写入 (UTC) | 2026-09-02T02:40:00Z |
| 配套机器收据 | `diagnostics/alphazuma-55-unconquered-cause-localization-20260901-receipt-v1.json` |
| 性质 | 只读综合裁定: 未启动任何训练 / 采集 / 评估; 未消耗正式种子 (formal_seed_consumption=false); CPU 全程留给 AlphaDiablo 战役 |
| 范围 | 51 关 = 55 面板中 46 个非丛林未攻克关 (village 9 / city 9 / coast 9 / grotto 9 / volcano 10) + 5 个面板外移动青蛙关 (Jungle5, village9, city8, Coast8, grotto8) |

## 0. 案由

主席令: **"祖玛不着急施工,只需要把原因定位出来"**。

据此, 本文件只做一件事: 对每个尚未攻克的关卡, 用 `D:\ZumaTraining` 与祖玛仓库中**已经存在**的工件回答"为什么没过", 给出主因 / 次因 / 置信 / 关键证据 / 战役内已验证的解法; 不制定施工计划, 不启动任何 run; 只对确需判别的关卡列出最小探针, 且全部为建议, 均未启动。丛林区 (Jungle1-4, 6-10) 由 `diagnostics/alphazuma-55-jungle-campaign-closing-summary-v1.json` 处理, 不在本文件范围; Jungle5 作为丛林区的移动青蛙成员计入。

方法链: 四条证据线 (§1: 闭环全景 pano / 教师 teach / 关卡结构 feats / 离线 BC 校准 offline) → 逐关诊断 (primary / secondary / confidence / evidence / validated_remedy / discriminating_probe) → 逐关怀疑者复核 (逐条重开工件, 标出误引与不成立的推断, 给出 corrected_cause / corrected_confidence) → 本综合 (采用复核后的 final_cause)。每个数字都可回溯到 §1 的文件与键; 标 "campaign memory" 者来自已验证的战役记忆; 标 "推断" 者是解释而非测量。

病因分类 (taxonomy):

| 码 | 名称 | 定义 |
|---|---|---|
| A | A_interface_ceiling 接口天花板 | 宏接口 / 模拟器无法表达该关所需动作 (双位 hop / 移动青蛙); teacher-through-macro 本身失败或根本无法运行 |
| B | B_weak_init 弱初始化 | 40 集 bc49 BC (单 checkpoint) 欠训练: 动词头在 knife-edge 上按种子翻入不开火或触发过度, aim 头近乎随机 |
| C | C_long_horizon_drift 长时程漂移 | 学生在数百决策内表现如教师, 随后进入 40 集教师走廊未覆盖的状态 (长链 / 清场阶段) 而无法恢复 |
| D | D_multi_chain_scheduling 双链调度缺陷 | 两条同时推进的链需要注意力调度, 40 集 BC 未捕获 (Jungle9 病) |
| E | E_verb_miscalibration 动词权重失准 | 类权重导致不开火 / 触发过度 (本文件中只作次因、信号或杠杆, 无一关为主因) |
| F | F_teacher_weak 教师弱 | 教师本身在该关失败 |
| G | G_never_evaluated 从未评估 | 无任何学生闭环工件 |
| H | H_tooling 工具 / 数据伪影 | 预算 / 截断 / sha / loader bug (51 关全部排除) |
| X | X_unknown 未知 / 表征 | 现有工件无法解释 |

## 1. 证据来源清单

### 1.1 闭环全景 (pano)
- `D:\ZumaTraining\alphazuma-55-macro-bc49\<level>\eval8\seed-1495002000..1495002007\completion.json` — 49 关 × 8 seeds, schema zuma-rl.alphazuma-55-macro-native-eval-v1, hold_ticks 8, stack [0,4,8], max_ticks 30000, 2026-08-19; 49/49 关 source.model_sha256 == final_model.sha256 (核对)。
- `D:\ZumaTraining\alphazuma-55-timepolish-baselines\<panel>\seed-1494001000..1494001015\completion.json` — 15 个丛林面板 (jungle3/7/8/9/10-240bc, jungle7-beta05bc, jungle7-beta05r2bc, jungle9-daggerbc, jungle9-beta05bc, jungle9-beta05bc-hold8, jungle9-beta05r2bc, jungle9-beta05r3bc, jungle9-widelagbc, Jungle2, jungle6), hold 4 (仅 -hold8 为 hold 8)。
- `D:\ZumaTraining\alphazuma-55-macro-ppo-s99081680-v1\eval-macro-native-ckpt{786k,1572k}`, `...-macro-ppo-s99081682-v1\eval-ckpt{786k,1572k}`, `...-macro-eval12-s99081674-v1`, `...-macro-eval12-dagger-s99081675-v1` — pre-bc49 宏接口学生, 12 关探针种子 1400920000-011, 全负。
- `D:\ZumaTraining\alphazuma-55-macro-bc-jungle{1,2}-s9908169{0,1}-v1-eval8`, `...-sweep-invfreq-v1-eval8` — 动词权重杠杆收据 (无权重 0/8 → invfreq 8/8, 7/8)。
- `D:\ZumaTraining\alphazuma-55-record-*`, `alphazuma-55-ab-hold{4,8}-s9908170{3,2}-v1` — 种子流扫描 (仅丛林)。
- `D:\ZumaTraining\alphazuma-55-park-settle-b-probe55-s99081664-v1\student_only.json` — pre-macro per-tick 学生 (0/55), 其 0 射集用作 "无阻漏斗" 实测 tick 参考。
- 中间产物 (只读推导): `scratchpad\zuma_evidence\panorama.json`, `closed_loop_groups.json` (250 组, unparsed=[])。

### 1.2 教师 (teach)
- `D:\ZumaTraining\alphazuma-55-macro-teacher55-s99081681-v1\{completion,config}.json` — 55 关 × 1 seed (1400920000 + index), hold 8, 50/55 (负: city3, Coast3, Coast7, Coast10, grotto3), completion sha256 a35dd7e1…。
- `D:\ZumaTraining\alphazuma-55-macro-teacher12-s99081678-v1\completion.json` — 12 关 × 1 seed, 10/12 (负: village3, volcano9)。
- `D:\ZumaTraining\alphazuma-55-macro-decisions-full49-s99081698-v1\{completion,config,episodes_manifest,status}.json` + `episodes\<level>-seed<seed>.npz` (键 macro_actions / teacher_per_tick_actions / fallback_codes) — 49 关 × 40 seeds (1495000000+), 1925/1960 教师胜, 0 截断, 0 fallback。
- `D:\ZumaTraining\alphazuma-55-park-settle-episodes-s99081659-v1\completion.json` — per-tick 教师 (hop 可用), 55 关 × 3 seeds (1543004000+), 164/165。
- `D:\ZumaTraining\alphazuma-55-macro-decisions-s99081689-v1\completion.json` — Jungle1+Jungle2 80 集 (应用权重 [1,3.7,8.3] 的实际来源)。
- `D:\ZumaTraining\alphazuma-55-macro-bc-jungle{3,7,8,9,10}-240-*\completion.json`, `...-bc-jungle7-beta05{,r2}-*`, `...-bc-jungle9-{dagger,beta05,beta05r2,beta05r3,widelag}-*`, `...-macro-dagger-jungle7-beta05-s99081748-v1\config.json`, `...-macro-dagger-jungle9-r1-s99081737-v1` — 240-ep / DAgger 收据 (含 verb_class_weights, teacher_drive_prob, sidecar 标签语义)。
- `C:\Users\Laure\Documents\祖玛\diagnostics\`: `alphazuma-55-macro-ppo-s99081680-preregistration-v1.json` (teacher_macro_ceiling_map_55.loss_detail, dual_position_note, levels_scope), `alphazuma-55-macro-ppo-jungle1-s99081683-preregistration-v1.json`, `alphazuma-55-jungle9-park-verdict-v1.json` (+ `-addendum-dual-curve`), `alphazuma-55-jungle-campaign-closing-summary-v1.json`, `alphazuma-55-jungle7-record-scan-verdict-s99081761-v1.json`, `alphazuma-55-weekend-s81081401-preregistration-v1.json`。

### 1.3 关卡结构 (feats)
- `D:\SteamLibrary\steamapps\common\Zuma's Revenge\main.pak` → `levels\levels.xml` 与 `levels\**\*.dat` (经 `src/zuma_rl/original_data.py` OriginalGameCatalog 只读解析, hard=False): Gun type / gx / gy, curveN, 每路点 in_tunnel 位, CURV 头 (colors, speed, start%, zuma_score, slow_distance / slow_factor, zuma_back)。
- `C:\Users\Laure\Documents\祖玛\src\zuma_rl\`: `alphazuma_55.py` (INCLUDED_LEVELS L10-66 = 序号来源; EXCLUDED_MOVING_FROG_LEVELS L68-74; DUAL_POSITION_LEVELS L76-83), `revenge_core.py` (from_installed L1256-1420 fail-closed; score_target L1216-1222; _update_zuma_bar L4115-4147 清场阶段机制; 胜负判定 L4180-4238), `revenge_env.py` (decode_action L327-338), `park_settle_action_wrapper.py` (L16-17, L35-38, L60, L181)。
- `C:\Users\Laure\Documents\祖玛\tools\`: `distill_alphazuma_55_macro_decisions_v1.py` (L676-757 无 teacher_won 过滤; ~L1049-1145 校准键), `collect_alphazuma_55_park_settle_episodes_v1.py` L224-236 _validate_levels, `probe_alphazuma_55_macro_teacher_v1.py` (L166-174, L258), `collect_alphazuma_55_macro_dagger_decisions_v1.py`, `audit_alphazuma_55_environment.py` L152-163。
- 中间产物: `scratchpad\zuma_evidence\features.json`, `structured_payload.json`。

### 1.4 离线 BC 校准 (offline)
- `D:\ZumaTraining\alphazuma-55-macro-bc49\<level>\{completion,config}.json` — 49 个 40 集 BC: calibration (== optimization.epochs[-1].holdout), split, dataset (含 teacher_losses_flagged), train_config.loss.verb_class_weights [1.0,3.7,8.3], 20 epoch holdout 轨迹, final_model.sha256; `epoch_20_model.zip` 与 `final_model.zip` 成员哈希核对 (仅 zip 时间戳不同)。
- 中间产物: `scratchpad\zuma_evidence\offline.json`, `bc49_raw.json`, `bc49_derived.json`。

(scratchpad = `C:\Users\Laure\AppData\Local\Temp\claude\D--ZumaGolden\d4a98326-10cc-4022-b885-c1d9179a2110\scratchpad`; 运行目录清单 = 同会话 `tool-results\bke5ncveh.txt`。)

### 1.5 §2 表格中的缩写
- `bc49/<L>` = `alphazuma-55-macro-bc49\<L>\completion.json` (calibration.*, optimization.epochs[], split, dataset)
- `eval8/<L>` (或 "eval8") = `alphazuma-55-macro-bc49\<L>\eval8\seed-1495002000..007\completion.json` episodes[0]
- `full49` = `alphazuma-55-macro-decisions-full49-s99081698-v1\completion.json` (per_level.<L>, episodes[level_id==L])
- `teacher55` / `teacher12` = 对应 completion.json episodes[level_id==L]
- `park-settle` = `alphazuma-55-park-settle-episodes-s99081659-v1\completion.json` per_level.<L> / episodes[]
- `timepolish/<panel>` = `alphazuma-55-timepolish-baselines\<panel>\seed-1494001000..015`
- `prereg` = `diagnostics\alphazuma-55-macro-ppo-s99081680-preregistration-v1.json`
- `levels.xml` = main.pak 成员, 经 OriginalGameCatalog
- 漏斗 = 无人射击时链头到达骷髅的 tick 数: 关卡参数推导值为推断性参考, probe55 的 0 射集为实测
- 解法码: **R1** = 240-ep 强化初始化 (同 bc49 配方, invfreq 权重; 收据 J3 2/8→16/16, J10 1/8→10/16); **R2** = J7 路线 (240-ep → 迭代 beta-0.5 DAgger, 每轮合并后重算 invfreq; 收据 0→0→4→10/16); **R3** = PPO 打磨, 仅从获胜初始化; **R0** = 无已验证解法
- aim_w3_fire = calibration.fire_decisions.aim_within_three_accuracy; ratio = fire_rate_calibration.student_to_teacher_fire_ratio; verb_acc = calibration.verb_accuracy

## 2. 分关裁定表 (按 INCLUDED_LEVELS 序号 #; 面板外 ordinal 0 置于表末)

| 关卡 | 区 | 已有闭环证据 | 主因 | 次因 | 置信 | 被质疑? | 关键证据 | 已验证解法 | 需探针? |
|---|---|---|---|---|---|---|---|---|---|
| village1 (#10) | village | eval8 1W/7L (hold 8) | B | E (信号) | 高 | 否 | eval8: seeds 2006/2007 shots=0, 621×wait_hold 同死于 4977 ticks; 闭环 fire 0.048 vs 教师 0.199, 而 bc49 离线 ratio 1.32 (over-fire); train_rows 10152 区内最少; best verb_acc 0.736@ep19 vs 出厂 0.679; full49 npz 教师首射 decision 25-26 (in-distribution); 自有 invfreq [1,3.56,7.67] ≈ 应用权重 | R1 (J3 同签名: 2/8 含零射集 → 16/16 零零射) | 否 |
| village2 (#11) | village | eval8 0/8; 4/8 死于 score>2700 (清场阶段, revenge_core L4117-4147) | B | C | 中 | 否 | bc49: aim_w3_fire 0.521 (区内第 3 差), aim CE 4.57 (chance ln180=5.19); eval8: 闭环 fire 0.171 ≈ 教师 0.189, 中位死亡 7399 ticks > 教师中位胜 5094, 509 决策/集 vs 376; full49 40/40; 教师观测为 actor 视角 (隐藏隧道路点非特权信息) | R1 (J3 aim_w3 0.767→0.880); 若 0/16 则 R2 | 否 |
| village3 (#12) | village | 无学生; teacher55 seed 1400920011 win (74 hop fallback); teacher12 seed 1400920002 loss (738 fallback, 805/821 wait_hold, score 880) | A (软) | G | 高 | 否 | levels.xml Gun gx1=175/gx2=610 双位; park_settle_action_wrapper L181 hop_verb_exposed=False; full49 config 无 village3, bc49 无子目录; park-settle 3/3 wins 含有效 hop 17/4/20; 6 个 pre-bc49 学生同 seed 全负 (0-29 shots) | R0 (需 hop 宏接口; 未验证: 仅用无 hop 可胜种子采集) | 否 (可选 teacher-only 16 seed) |
| village4 (#13) | village | eval8 0/8; 7/8 死于 score≥4280 (>目标 3100, 清场阶段) | B | E (信号); C (预期下一阶段) | 中 | 否 | bc49: ratio 2.19 (49 关最差), verb_acc 0.744@ep12→0.575@ep20 (-0.169), aim_w3_fire 0.458, aim CE 5.66 > chance; eval8: 闭环 fire 0.482 (2.5× 教师), 中位死亡 9215 (1.9× 教师中位胜 4820); 自有 invfreq [1,3.79,8.25] ≈ 应用 → 非权重失配; policy.pth 成员哈希 == epoch_20 (H 排除) | R2 (J7 同签名: 240-ep 独用 0/16, beta05 4/16, r2 10/16) | 否 |
| village5 (#14) | village | eval8 1W/7L (win seed 2006: 19469 ticks = 3.2× 教师中位 6106) | B | E (信号) | 高 | 否 | eval8: seeds 2000/2001/2007 仅 20/23/12 shots (近不开火), 闭环 fire 0.122 = 0.58× 教师; bc49: ratio 1.29 (离线 over-fire), aim_w3_fire 0.741 (区内最佳), best verb_acc 0.746@ep14 vs 0.690; 自有 invfreq [1,3.35,7.31] < 应用 → 权重预测过火, 与观察相反 | R1 (J10 1/8 fire 0.113 → 10/16); 之后 R3 恢复节奏 | 否 |
| village6 (#15) | village | eval8 0/8; 中位死亡 3564 ticks = 教师中位胜 10154 的 35%; 185 决策/集 vs 教师 673 | D | E 降为信号; B 背景 | 中 | 是 (主因存续; 论据 "J9 全链保持全局权重" 被证伪: bc-jungle9-beta05r2/r3 权重 [1,2.83,4.12]/[1,2.87,4.09] 仍 swap 0.21/0.41, 4/16 与 3/16) | levels.xml curve1=Village6-1, curve2=Village6-2 (区内唯一双曲线); bc49 verb_acc 0.762 区内最佳、epoch 衰减最小, 却闭环 swap 0.280 (教师 0.145)、fire 0.302; 全局权重相对自有 invfreq [1,2.38,4.14] 超权 fire 1.56× / swap 2.0× (区内最大); twin Jungle9 全链 (240 / dagger / beta05×3 / widelag) 最佳 4/16 | R0 (J9 最佳 4/16; 唯一产生胜场的杠杆为迭代 beta-DAgger); 禁 plain DAgger | 是 (P-village6) |
| village7 (#16) | village | eval8 1W/7L (win seed 2002: 14476 ticks = 2.7× 教师) | B | E (信号) | 中 | 否 | eval8: 同一 checkpoint 按种子分裂 — seed 2006 12 shots/403 决策 (score 70) vs seeds 2005/2004/2001 186/111/87 shots 且死于 score 7130/4060/3870 (>目标 3700 清场阶段); 闭环 fire 0.244 = 1.37×; bc49: ratio 1.64, precision 0.513, best verb_acc 0.759@ep16 vs 0.692; 自有 invfreq [1,4.17,8.95] > 应用 → 权重预测欠火, 与观察相反 | R1; 若 0/16 且过火则 R2 | 否 |
| village8 (#17) | village | eval8 2W/6L (非丛林最佳; win seed 2001 7497 ticks ≈ 教师中位 7670) | B | E (信号) | 高 | 否 | bc49: ratio 1.90, recall 0.923, precision 0.486, best verb_acc 0.793@ep17 vs 0.672 (-0.121), aim_w3_fire 0.786; eval8: 闭环 fire 0.316 (1.5× 教师), swap 0.141 vs 0.097; 自有 invfreq [1,3.30,7.17] 仅超权 1.12×/1.16×; full49 40/40; pre-bc49 学生同 seed 全负 | R1 (bc49 J8 已 7/8 + 1 截断, 非拯救收据; 适用收据 J3/J10), 后 R3 | 否 |
| village10 (#18) | village | eval8 0/8; seeds 2000/2002/2003/2004 shots=0 (507×wait_hold, 4065 ticks, 四条字节相同轨迹) | B | E (信号) | 中 | 否 | bc49: train_rows 21966 区内最多, ratio 1.15、precision 0.698 区内最佳校准, verb_acc 0.760, 却 aim_w3_fire 0.493、aim CE 4.52; 闭环 fire 0.025 (0.11× 教师); full49 npz 教师首射 decision 4 (5/6 集) → 崩溃发生于 in-distribution 早期状态; 应用权重超权 fire (3.7 vs 自有 2.99) 方向相反; 教师 109 shots 经同一契约 (H 排除) | R1 (J3: 零射集 → 0); 667 决策/集 → 预期需 R2 第二阶段 | 否 |
| city1 (#19) | city | eval8 0/8; 中位死亡 5714 ≈ 教师中位胜 5450, 447 决策/集 > 教师 380, 但 score 中位 1415 vs 4825 | B | E (信号) | 中 | 否 | bc49: train_rows 12175 区内最少, aim_w3_fire 0.525, exact 0.108, aim CE 4.78, best verb_acc 0.790@ep16 vs 0.709 (逐 epoch 含噪 0.58-0.79); eval8: 闭环 fire 0.136 vs 教师 0.207 而离线 ratio 1.24 (regime split), seed 2006 24 shots/538 决策; full49 40/40 | R1 (适用收据 J3/J10; record-jungle3 2946/3000 为 golden-opening seeds, 非随机面板) | 否 |
| city2 (#20) | city | eval8 1W/7L (win seed 2007: 16237 ticks / 945 决策 / 214 shots, 超教师最慢胜 14450); 7 负死于 4712-7565 (教师中位胜 6457) | B | C | 高 | 否 | bc49: aim_w3_fire 0.805 (区内最佳), exact 0.370, aim CE 2.79 (最低), ratio 1.63; eval8: 闭环 fire 0.191 = 教师 0.189, swap 0.126 vs 0.094; 520 决策/集; full49 40/40 | R1; 若面板仍有教师时程附近的活跃失败则 R2 | 否 |
| city3 (#21) | city | 无学生; teacher55 seed 1400920020 loss 10689 ticks, 1157/1273 决策为被拦截 hop, 17 shots, 未截断 | A | 无 | 高 | 否 | levels.xml Gun (210,300)/(590,300); teacher55 config adapter.verb_mapping.hop='wait_hold (fallback)'; park-settle 同教师 3/3 wins (raw hop 343/347/603, 有效 32/32/55); full49 config 49 关无 city3, bc49 无子目录; prereg loss_detail.city3 'dual-position, hop-blocked' | R0 (仅接口扩展 hop 后 teacher-through-extended-macro) | 否 |
| city4 (#22) | city | eval8 0/8; 死于 2330-3982 (中位 3006 = 教师中位胜 12559 的 24%), 126.8 决策/集 | B (必要, 非充分) | E; D 三级待定 | 低 | 是 (排序被驳: 触发过度签名离线已在 (ratio 1.56, 学生 fire 0.414 vs 教师 0.265) 且闭环 fire 0.379 / swap 0.305; 同类 J7/J9 240-ep 独用均 0/16, 所引 "J3/J8 收据" 不适用) | levels.xml 双曲线 city4-1v2/-2v2; bc49: aim_w3_fire 0.536, exact 0.087, aim CE 4.87, best verb_acc 0.818@ep18 vs 0.760; full49 教师 38/40, 2 负集 (1162/31289 行) 入训练; 全局权重相对自有 [1,2.29,4.25] 超权 swap 2×; 射击节奏 63.5 ticks/shot = 教师 61.9 | R2 (含 invfreq 重算); 双曲线残余 R0 (J9 park); per-level 权重为廉价未验证 E 杠杆 | 是 (P-city4) |
| city5 (#23) | city | eval8 0/8 双峰: 5 seeds 死于 3847-8023; 3 seeds 死于 11001-12793 (> 教师中位胜 10183), score 5110-6780 | B | C (E 过换 1.5× 亦活跃) | 中 | 否 | bc49: aim_w3_fire 0.793 (区内第 2), exact 0.168, best verb_acc 0.800@ep14 vs 0.752; eval8: 闭环 fire 0.212 ≈ 教师 0.199, swap 0.148 vs 0.098; 748 决策/集; 引用 "3 seed 分数 ≥ 教师中位 5845" 仅 6780 成立; J9 时程读法已被其附录取代 (C 为推断) | R1, 后 R2 | 否 |
| city6 (#24) | city | eval8 0/8; 死于 2982-6556 (中位 5042 = 教师中位胜 9040 的 0.56×), score 中位 1745 vs 5370 | B | E (信号) | 中 | 否 | bc49: aim_w3_fire 0.406 (区内最差), exact 0.115, verb 头无 epoch 漂移 (0.795→0.789), ratio 1.48; eval8: 闭环 fire 0.138 / swap 0.064 (教师 0.184/0.091, 唯一欠换关), seed 2005 27 shots/375 决策; full49 40/40 | R1 (J3/J10 欠火同类; city6 aim 0.41 远低于 J3/J10 的 0.77 → 中置信); X (曲线几何) 未排除 | 否 |
| city7 (#25) | city | eval8 0/8; 4/8 seeds 死于 12706-17209 (> 教师中位胜 10597), score 4830-6520 (< 教师中位胜分 7065); seed 2002 仅 7 shots | B | C (顺序不可由现有工件判别) | 低 | 是 (原 C 主因被驳: 依据 J9 已被双曲线附录取代的 "决策累积" 读法与未经验证的时程外推 (J7 283 需 DAgger 而 J8 299 / J10 294 不需); "champion 1836/3000" 为 PPO ckpt 在 golden seeds 成绩) | full49 813.7 决策/集 (区内最长); bc49: 离线校准区内最佳 (ratio 1.20, precision 0.665, aim_w3_fire 0.730) 但 exact 0.096, aim CE 4.00; eval8 闭环 fire 0.194 ≈ 教师 0.209; full49 40/40 | R1, 后从最佳存活 BC R2 | 是 (P-city7) |
| city9 (#26) | city | eval8 0/8; 死于 2920-6343 (中位 4325 = 教师中位胜 9549 的 0.45×), score 中位 925 vs 5635 | B | C (推断; E 1.17× fire / 1.35× swap 同等适配) | 中 | 否 | bc49: aim_w3_fire 0.499, exact 0.100, best verb_acc 0.788@ep19 vs 0.745, ratio 1.59; eval8: 射击节奏 69 ticks/shot = 教师 70, 280 决策/集 vs 教师 750; 链速 0.9 最快 + 两段中途隧道 [2453,2644]/[3081,3257] (X 未排除, bundle 未重开); pre-bc49 学生同 seed 全负 (0-25 shots) | R1, 后 R2 | 否 |
| city10 (#27) | city | eval8 0/8; 死于 2558-6593 (中位 3980), 196.8 决策/集 (教师 761); 闭环 swap 0.264 (教师 0.149, 1.77×), fire 0.318 (1.24×) | B (必要) | E / D 待定 | 低 | 是 (排序被驳: 过火/过换签名属 J7/J9 类, 240-ep 独用无成功收据; D (结构 + J9 附录, 无 per-curve 读数) 与 E (离线 ratio 1.25, 全局权重超权 swap 2.1×) 无法由工件排序) | levels.xml 双曲线 city10-1/-2 + effect1 (fog, 对特征提取器影响未验证); bc49: verb_acc 0.819 (49 关最佳, 无漂移), precision 0.690, 但 aim_w3_fire 0.676, exact 0.094, aim CE 4.75; 自有 invfreq [1,2.32,4.00]; full49 教师最长胜 18580 | R2; 双曲线残余 R0; per-level 权重为廉价未验证杠杆 | 是 (P-city10) |
| Coast1 (#28) | coast | eval8 0/8; 死于 2656-5525 (中位 3697 = 教师中位胜 8758 的 42%), 247.9 决策/集 (教师 639 的 39%) | B | C (共等) | 低 | 是 (置信与 B>C 排序被驳: 所有 coast 学生同为 40 集初始化, 引用指标在 C 下同样成立; 早死→B 启发式在丛林收据 1/4 命中; "J8→14/16" 误引 (bc49 J8 已 7/8); 时程标尺非单调) | bc49: ratio 1.29, aim_w3_fire 0.533, best verb_acc 0.803@ep14 vs 0.764, 逐 epoch ratio 0.98-1.66 摆动 (仅存 ep20); eval8: 闭环 fire 0.85×, swap 1.32×, score 中位 565 vs 教师胜 5665; full49 40/40; pre-bc49 学生同 seed 全负 | R1 (B/C 首步相同), 若 0/16 则 R2 | 是 (P-Coast1) |
| Coast2 (#29) | coast | eval8 0/8; 中位死亡 8130 (4/8 超 teacher55 胜 7914; 3/8 score 6890-7110 > 目标 4650 清场阶段; 5/8 低于目标) | B | C (共等) | 低 | 是 (原 C 主因被驳: "全部晚于教师" 误引 (4/8); 5/8 死于目标分以下; 单位漏斗存活 2.6× 与 Coast6/7 同带; "超教师时长" 在 J10 (1.23×, 240-ep 治愈) 与 J7 (1.41×, 未治愈) 上不判别) | bc49: ratio 1.47, precision 0.59, aim_w3_fire 0.571, best verb_acc 0.799@ep14 vs 0.741; eval8: 闭环 fire 1.31×, swap 1.71×, 447.8 决策/集 = 教师 579 的 77% (区内最高); 579 决策/集为 coast 最短 → 区级 B-vs-C 最廉价判别关 | R1 (首步), 后 R2 | 是 (P-Coast2) |
| Coast3 (#30) | coast | 无学生; teacher55 seed 1400920029 loss 3971 ticks, score 90, 449/485 决策为被拦截 hop, 3 shots | A | 无 | 高 | 否 | levels.xml 双位 Gun; hop_verb_exposed=False; park-settle 3/3 wins (raw hop 370/196/303, 有效 33/21/29); teacher55/12 七个双位教师集: hop 需求 ≥90% 者全负、≤18% 者全胜; full49/bc49 无 Coast3 | R0 (接口扩展) | 否 |
| Coast4 (#31) | coast | eval8 0/8; 死于 1861-4000 (中位 3096 = 教师中位胜 12400 的 25%), 170.8 决策/集 (教师 896 的 19%, 区内最早死) | B | C | 低 | 是 (同 Coast1; 另 "按 holdout ratio 选 checkpoint" 无收据 — 所有 240-ep/DAgger BC 仅存 epoch 20, 无逐 epoch 评估) | bc49: ratio 1.61, recall 0.92, precision 0.57, aim_w3_fire 0.572, best verb_acc 0.836@ep14 vs 0.746 (区内最大跌幅), 逐 epoch ratio 0.92↔1.93 摆动; eval8: 闭环 fire 1.27×, swap 1.33×, score 中位 400; full49 40/40; 离线指标不预测 (J4 verb 0.579 仍 8/8) | R1, 后 R2; checkpoint 选择子解法标为未验证 | 是 (P-Coast4) |
| Coast5 (#32) | coast | eval8 0/8; seed 2001 470×wait_hold 0 shots; 闭环 fire 0.047 (0.26× 教师 0.177), swap 0.38× | B | E (信号) | 中 | 否 | bc49: ratio 0.86 (49 关唯一 <1; ep19 1.63/recall 0.88 → ep20 0.86/0.64, 仅存 ep20), verb_acc 0.814、aim_w3_fire 0.79 区内最佳; 自有 invfreq [1,4.15,8.50] ≈ 全局 → 非权重失配; 链起点 67% 最深; J3/J10 欠火同类 (240-ep → 16/16, 10/16) | R1; [1,8,12] 强权重臂从未闭环评估 (标未验证); "epoch 抽签" 为推断 | 否 |
| Coast6 (#33) | coast | eval8 0/8; 死于 4087-6329 (中位 4841 = 教师中位胜 12538 的 39%), 310.6 决策/集 (37%) | B | C (共等) | 低 | 是 (同 Coast1: "欠训练 aim 头" 不判别 — J8 40-ep aim 0.598 胜 7/8, J9-240 aim 0.859 负 0/16; B 与 C 在此为纯平局) | bc49: aim_w3_fire 0.502 (区内最低), ratio 1.49, best verb_acc 0.768@ep18 vs 0.718; eval8: 闭环 fire 0.97× (区内校准最佳), swap 1.25×, score 中位 1245 vs 教师胜 9595; 链速 0.925 区内第 2 快; full49 40/40 | R1, 后 R2 | 是 (P-Coast6) |
| Coast7 (#34) | coast | eval8 0/8; 死于 4040-7023 (中位 4841 = 教师中位胜 9676 的 50%), score 中位 1465; 教师: teacher55 timing loss 13750 ticks / 9050 分 / 0 fallback, full49 39/40 (负集 seed 1495001157 入训练) | B | C (由 E 改); F 次要天花板 (39/41) | 低 | 是 (次因 E 被驳: 战役无手工重加权收据, 5-16% 权重差解释不了 1.44×/1.81× (同权重 Coast6 为 0.97×); J7/J9 触发过度类只靠 beta-DAgger 治愈; "剔除教师负集" 与 jungle10-240 实践 (含 4 截断集 → 10/16) 相悖) | bc49: ratio 1.75 (区内最高), precision 0.523 (最低), verb_acc 0.662 (最低; 0.779@ep16 → 0.662); eval8: 闭环 fire 1.44×, swap 1.81× (区内最高); prereg loss_detail.Coast7 'NON-dual-position timing loss ... KEPT'; 338-wp 末段隐藏隧道 (关联为推断) | 240-ep 采集 (标准 flag-and-include) + 以该集 invfreq 重算权重 BC, 后 R2; 16 seed 教师对照界定 F | 是 (P-Coast7) |
| Coast9 (#35) | coast | eval8 0/8; 死于 2338-4615 (中位 3439 = 教师中位胜 14850 的 23%), 293 决策/集 (教师 1115, 49 关第 3 长) | B | C (推断) | 中 | 否 | bc49: ratio 1.16、recall 0.740 而闭环 fire 0.62× (欠火 regime split, 与 J10 0.113→10/16 同类), aim_w3_fire 0.567, ep19 ratio 1.72 → ep20 1.16; 链速 1.0 最快; full49 39/40 (1 timing loss 入训练); pre-bc49 学生同 seed 全部近不开火 (0-19 shots); 自有 invfreq [1,4.19,8.65] ≈ 全局 | R1 (J10 收据), 后 R2; 16 seed 教师对照界定 F | 否 |
| Coast10 (#36) | coast | 无学生; teacher55 seed 1400920035 loss 3699 ticks, score 550, 426/451 决策为被拦截 hop, 3 shots | A | D (仅由 curve_count=2 推断; 诚实标签为 G) | 高 | 否 | levels.xml 双位 + 双曲线 (Coast10-1/-2); park-settle 3/3 wins (raw hop 859/676/596, 有效 76/57/51, coast 最依赖 hop); 无任何 macro 学生工件; prereg loss_detail.Coast10 'dual-position, hop-blocked' | R0 (接口扩展; 扩展后预期需 J9 双链解法) | 否 |
| grotto1 (#37) | grotto | eval8 1W/7L (win seed 2006 10772 ticks); 5/8 集超教师中位胜 9676; 2 负 score 4610/5440 > 目标 4300 (清场阶段, 由分数推断) | C | B | 中 | 否 | bc49: aim_w3_fire 0.780 (区内最佳), verb_acc 0.794, recall 0.892; eval8: 闭环 fire 0.226 ≈ 教师 0.216, 中位决策 593 = 教师 711 的 83%, 分/射 26.8 vs 教师 34.2 (0.80, 区内最佳); 损失既非不开火亦非触发过度 (J3/J7 签名均不符); full49 40/40 | R1 + R2 (J7 0→0→4→10); 唯一可考虑 R3 门槛的 grotto 关 | 否 |
| grotto2 (#38) | grotto | eval8 0/8; 死于 2707-7574 (中位 3631 = 1.9× 漏斗 1916), 215 决策/集 (教师 924 的 23%) | B | C (共等规划) | 中 | 否 | eval8: 分/射 20.4 vs 教师 42.5 (0.49), 闭环 fire 1.23×, 13.4 vs 13.9 shots/1k ticks; bc49: precision 0.575, aim_w3_fire 0.654, holdout verb_acc 0.817@ep15 → 0.735 为逐 epoch 方差 (aim 仍在升, 非过拟合); full49 40/40 | R1 (必要); 924 决策/集 → R2 作共等规划 (J7/J9 240-ep 独用 0/16) | 否 |
| grotto3 (#39) | grotto | 无学生; teacher55 seed 1400920038 loss 3769 ticks, score 0, 470×wait_hold, 460 hop fallback, 0 shots (双位关中唯一 0 射) | A | 无 | 高 | 否 | levels.xml Gun (200,215)/(600,425) resid GROTTO3CHUTE2; teacher55 config adapter.mask_rule 'hop always masked off'; park-settle 3/3 wins (raw hop 459/325/410, 有效 40/26/33); full49/bc49 无 grotto3 | R0 (接口建设; 可选: 1 集 per-tick 禁 hop 教师验证位置 1 单独可胜性) | 否 |
| grotto4 (#40) | grotto | eval8 0/8; 死于 2428-3448 (中位 2679 = 1.73× 漏斗 1552), score 130-530, 198 决策/集 (32%) | B | C; X 共候选 | 中 (原高) | 是 (置信被降: 离线 aim_w3_fire 0.630 ≈ grotto2 0.654 / grotto10 0.646, 但闭环分/射 6.9 (12% 教师) 比其差 3-5×; 闭环 wait 0.75 区内最高 → 数据量解释不了; "J8" 误引) | bc49: verb_acc 0.791, recall 0.828, ratio 1.26 而闭环 fire 0.81× (regime split), ep1 aim_w3_fire 0.090 (区内最慢学习); levels.xml gun (400,490) 区内最低 + 入口隧道 135 wp 最长 (关联为推断); full49 40/40, 区内最快清场 (中位 7500) | R1; 若分歧低仍失控 → 升级 X (表征/几何) | 是 (P-grotto4) |
| grotto5 (#41) | grotto | eval8 0/8; 死于 2189-3352 (中位 2650 = 1.81× 漏斗 1464, 区内最短), 160 决策/集 (19%), 分/射 10.7 = 20% 教师 | B | C (F 降为后期天花板注记) | 中 (原高) | 是 (次因 F 被驳: 5 个教师负集为 482-1029 决策 / 2910-7760 分的长局, 学生 ~375 分即死, 12.5% 负集份额产生不了 20% 效率学生; "J8" 误引; 852 决策/集无 240-ep 独用先例) | bc49: verb_acc 0.727、precision 0.524、ratio 1.75 (区内最差三项), aim_w3_fire 0.595; dataset teacher_wins 35 / losses_flagged 5 (49 关第 3 弱); levels.xml 链起点 64% + 末段隐藏隧道 [3145,3283] | R1, 后 R2; 胜集过滤未验证 | 是 (P-grotto5) |
| grotto6 (#42) | grotto | eval8 0/8; 6/8 死于 4806-9652 (2 负 4360/4530 距目标 4700 仅 170-340 分), 2/8 早崩 (282/285 决策) | C | B | 中 | 否 | bc49: aim_w3_fire 0.782 (区内最佳并列), recall 0.795 (区内最低); eval8: 闭环 fire 0.97× (区内校准最佳), 分/射 31.5 vs 教师 48 (0.65), 中位 331.5 决策 (39%); levels.xml speed 0.945 最快 + end_wp 4530 最长单曲线; full49 40/40 | R1 + R2 | 否 |
| grotto7 (#43) | grotto | eval8 0/8; 死于 3615-10420 (中位 6688 = 2.94× 漏斗 2278, 区内最宽容); 2/8 score 3990/4900 > 目标 3500 (清场阶段); 最长负局 604 决策 ≈ 教师全长 | C | B (近共主因) | 中 | 否 | bc49: aim_w3_fire 0.729, 但 verb_acc 0.739 / precision 0.560 / ratio 1.585 与 grotto2 (B) 同型; eval8: 闭环 fire 1.11×, 分/射 29.7 vs 45.5 (0.65), 中位 424 决策 (50%); full49 40/40 | R1 + R2 | 否 |
| grotto9 (#44) | grotto | eval8 0/8; 死于 2282-2931 (49 关最紧死亡带, 1.33× 漏斗 1949), score 100-350, 分/射 5.1 = 9% 教师 | B | C; X 共候选 | 中 (原高) | 是 (置信被降: "aim CE 4.59 ≈ chance" 误读 — 训练 aim_w3 0.540 远高于 7-bin 窗口随机 0.039, J8 CE 4.98 仍 7/8; 真异常为 train_rows 30149 (区内第 2 多) 却 holdout aim_w3_fire 0.550 最差, 同速 grotto6 0.782; J9-240 aim 0.72→0.86 仍 0/16) | full49 教师 39/40 (1 负集入训练), 931 决策/集; levels.xml speed 0.945 最快; pre-bc49 学生同 seed 全部不开火 (0-11 shots); 闭环 fire 1.21×, 13.85 vs 13.84 shots/1k ticks | R1, 后 R2; 离线伴随判别: 240-ep holdout aim_w3_fire < 0.65 → X | 是 (P-grotto9) |
| grotto10 (#45) | grotto | eval8 0/8; 死于 2392-7979 (中位 4026 = 2.90× 曲线 1 漏斗 1389), 205 决策/集; 闭环 swap 0.281 (教师 0.143, 1.97×), fire 1.14× | C / D 共主因 | E (swap 1.97× 已验证); B 背景; F 排除 | 中 | 是 (原 B 主因被驳: verb_acc 0.803 区内最佳, 分/射 0.61× 区内第 2, 漏斗倍数 2.90× = grotto7 (C 标); "18% 时程" 系 49 关最长中位胜 17338 所致; "1112 为最长时程" 误引 (第 4); 7 教师负集皆 >9400 目标分的清场失败, 非学生 <4240 分即死之因) | levels.xml 双曲线 grotto10-1 (6 色, 0.725) / -2 (5 色, 0.705) 不对称, 目标 9400; full49 教师 33/40 (49 关第 2 弱); 全局 swap 权重 8.3 相对本关 0.143 份额超权 ~1.6× (推断); eval8 记录无 per-curve 字段 → D 为推断 | R1 + J9 双链路线 (R0 级: J9 park); per-level 权重为廉价未验证杠杆 | 是 (P-grotto10) |
| volcano1 (#46) | volcano | eval8 0/8; 死于 2515-4803 (中位 3450 = 0.90× 实测漏斗 3848), 226.6 决策/集 (教师 839 的 27%), score 中位 815 = 10% | C | B | 中 | 是 (原 B 主因被驳: 引用的离线数字 (verb 0.75, aim_w3_fire 0.44, loss 仍降, holdout 摆动) 被闭环获胜的 bc49 关持平/超过 (J4 8/8 verb 0.579, J8 7/8 aim 0.598); 49 关中唯一分隔胜负的量是决策时程; 采集成本误引 4-7×) | eval8: 射速 12.6 vs 教师 13.9 shots/1k ticks 而转化 10%; full49 40/40; 全域时程分裂: <645 决策/集 40/160 胜, ≥645 1/232; J7-240 / J9-240 同型 (离线健康, 教师速射, 0/16) 仅由 beta-DAgger 治愈 | R1 (必要) + R2; 240-ep 采集实测 3.1-5.6 h wall (136-243 ticks/s), 非 45 min | 是 (P-volcano1) |
| volcano2 (#47) | volcano | eval8 0/8; 死于 1682-3695 (中位 2644 = 0.81× 实测漏斗 3248, 比不开火更早死); 3/8 近不开火 (7/9/10 shots) | B | C | 中 | 否 | bc49: verb_acc 0.829 (49 关最佳), ratio 1.08 (第 2 佳校准) 却闭环 fire 0.080 (0.47× 教师) — bc49 J3 同签名 (fire 0.050, 240-ep → 16/16); aim_w3_fire 0.459; full49 教师 37/40 (3 负集为全长晚负, 未过滤入训练); levels.xml 链起点 70%、可视仅 1001 wp (最短); 自有 invfreq [1,3.26,6.5] < 全局 → 权重预测过火 | R1 (J3 直接先例); 645 决策/集 → 预期 C 随后 (J7-240 同型) | 否 |
| volcano3 (#48) | volcano | 无学生; teacher55 seed 1400920047 WIN 13029 ticks / 7130 分 / 188 shots, 仅 13/936 决策 hop fallback (1.4%) | G | A (种子依赖风险) | 高 | 否 | alphazuma_55.py DUAL_POSITION_LEVELS 范围规则排除 (prereg L71 dual_position_note: village3/volcano3 无 hop 亦胜, 仍排除); full49/bc49 无 volcano3; park-settle 3/3; 唯一学生数据为 pre-macro 不开火 (probe55 4184 ticks, 0 射) | R0 (无双位验证解法); 若 teacher-only 探针 ≥12/16 → 修订 prereg 后进入 R1 轨道 | 是 (P-volcano3) |
| volcano4 (#49) | volcano | eval8 0/8; seeds 2001/2006 shots=0 (497×wait_hold, 死于 3984 = probe55 实测漏斗 3984 精确相等) | B | C | 中 | 否 | bc49: verb_acc 0.819 (49 关第 3), recall 0.80, ratio 1.22 却闭环 fire 0.048 (J3 同签名); aim_w3_fire 0.497; full49 40/40; levels.xml 入口隧道 156 wp (最长, drawtunnel=true); 自有 invfreq [1,4.0,8.1] ≈ 全局 | R1 (J3 直接先例); 924 决策/集 → 预期 R2 (J7/J9-240 0/16) | 否 |
| volcano5 (#50) | volcano | eval8 0/8; 死于 2964-4542 (中位 4342 = 1.13× 实测漏斗 3827), score 中位 935 = 12%; 射速 12.8 vs 教师 12.9 | C | B | 中 | 是 (原 B 主因被驳: aim_w3_fire 0.62 > J8 0.598 (7/8); 0.04 末段 verb 跌幅在 8 集 holdout 噪声内; 1004 决策/集 = 2.3× J9) | eval8: 闭环动词混合 = 教师 (fire 0.203 vs 0.181); full49 40/40; bc49 verb_acc 0.777, recall 0.86; 两段进料侧隐藏隧道 (推断) | R1 + R2 | 是 (P-volcano5) |
| volcano6 (#51) | volcano | eval8 0/8; 中位死亡 6490 (1.56× 实测漏斗 4150, 区内最佳); seed 2002 15567 ticks / 6710 分 (超教师中位胜时 14334, 95% 目标分) | C | B | 中 | 是 (区内最清晰错序: bc49 训练 loss 3.034 为 49 关最低, aim_w3_fire 0.792 为 49 关第 3, 离线 ≥ 获胜的 240bc 锚点 J8-240 0.716; "ratio 1.52 / verb 摆动" 亦见于胜者 J8 1.56, jungle6 1.63) | eval8: 分/射 27% (区内最佳), 452.8 决策/集 (43%); full49 40/40, 1059 决策/集 = 2.4× J9; "仅 BC 即可过关" 无 >320 决策/集先例 | R1 + R2 | 是 (P-volcano6) |
| volcano7 (#52) | volcano | eval8 0/8; 死于 3018-5741 (中位 4365 = 1.10× 实测漏斗 3966), score 11%; 274 决策/集 (教师 1211.4 = 49 关最长) | C | B | 中 | 是 (原 B 主因被驳: 49 关最大训练集 38408 行仍失败 (诊断自称 "行数非限制量"); verb 0.796 高于所有 bc49 胜者; 所引锚点 (无 >320 决策/集 BC-only 成功) 本身是 C 证据) | eval8: 射速 12.8 vs 教师 13.3; full49 39/40 (1 全长晚负 seed 1495000094 入训练), 最长胜 21781 < 30000 (H 排除); 链速 1.0 | R1 + R2 (J7 需两轮); 240-ep 实测 4.5-8.1 h | 是 (P-volcano7) |
| volcano8 (#53) | volcano | eval8 0/8; 死于 3149-4943 (中位 3832 ≤ 0.92× 漏斗; 参考 4176 为下界 — probe55 该集射 6 发), score 7.8% | C | B | 中 | 是 (原 B 主因被驳: verb 0.766 / aim_w3_fire 0.531 在 bc49 胜者范围内 (Jungle2 0.572 4/8, J8 0.598 7/8); 0.815→0.766 末段跌幅在噪声内; 1132 决策/集 49 关第 2) | eval8: 射速 13.8 = 教师 13.8, 净有害射击 (比不开火更早死); full49 40/40; bc49 recall 0.88 | R1 + R2 | 是 (P-volcano8) |
| volcano9 (#54) | volcano | eval8 0/8; 死于 1589-2816 (中位 2144 = 49 关最快, 0.59× 实测漏斗 3618), 98 决策/集 (12%), swap 0.290 (教师 0.141) | D | F (教师 27/40; teacher12 loss 8664/8990, park-settle 2/3 — 三个教师源皆负过) | 中 | 否 | levels.xml 唯一不对称双曲线 volcano 关 (6/5 色, 目标 9600 最高); bc49 dataset 13/40 教师负集未过滤 (distill L676-757), verb_acc 0.731 / ratio 1.68 / loss 5.195 区内最差; 全局权重相对自有 [1,2.46,4.34] 超权 fire 1.5× / swap 1.9×; 六个双曲线关 bc49 eval8 合计 1/48 胜, swap 0.26-0.34; J9 r2/r3 近自有权重仍 churn; prereg L24 'timing loss ... KEPT' | R0 (J9 最佳 4/16); 零成本卫生 (剔负集 + 自有权重) 未验证; 调度感知修复为新干预 | 否 (探针 (a) 需新 per-curve 仪表) |
| volcano10 (#55) | volcano | eval8 0/8; 死于 3380-5241 (中位 4054 = 0.96× 实测漏斗 4218), score 7.2%; 闭环 fire 0.099 (0.59× 教师 0.169), 8.5 vs 13.0 shots/1k ticks | B | C | 中 | 否 | full49 教师 fire 0.169 为 49 关最低 (39/40, 1 全长晚负入训练); bc49: recall 0.780 / precision 0.550 区内最低, ratio 1.42 (离线过火 vs 闭环欠火 regime split, J10 同签名 0.113 → 10/16); levels.xml clump 2 (同色聚集最少); 自有 invfreq [1,4.42,8.9] 略高于全局 (0.84× 欠权) 但离线仍过火 → E 为信号 | R1 (J10 收据); 932 决策/集 → 预期 R2 随后 | 否 |
| Jungle5 (#0 面板外) | jungle | 无任何工件 (0/220 运行目录; 不在 teacher55 / full49 / bc49) | A | G | 高 | 否 | levels.xml Gun type='horiz' (startx 50, starty 540, width 700, "Switchback Slider"); revenge_core.from_installed L1303-1317 fail-closed (复现: NotImplementedError "'horiz' moving-frog type", Jungle4 正常构造, 0 tick); src 无导轨运动学 (word-boundary grep horiz/vert/slider/rail 为空), 动作空间仅角度 (revenge_env L327-338), 宏契约 hop_verb_exposed=False; prereg L16/L98/L116/L133 + audit L152-163 冻结; tests 钉住拒绝 | R0 (需模拟器导轨运动学 + 宏接口位置动作 + prereg 修订); 5 关导轨类中最短 (partime 4000, 目标 2250) | 否 |
| village9 (#0 面板外) | village | 无 | A | G | 高 | 否 | levels.xml Gun type='horiz' startx 50 starty 520 width 700 ("Logroller"); alphazuma_55.py EXCLUDED_MOVING_FROG_LEVELS; revenge_core L1303-1305 拒绝; teacher55 55 关 / full49 49 关 / bc49 均无 | R0 | 否 |
| city8 (#0 面板外) | city | 无 | A | G | 高 | 否 | levels.xml Gun 'horiz' (50,540, width 700); loader 拒绝 (L1311-1313 NotImplementedError unless allow_partial_level); 不在 teacher55 / teacher12 / full49 / bc49 | R0 (需导轨青蛙执行器 + 位置通道) | 否 |
| Coast8 (#0 面板外) | coast | 无 | A | G | 高 | 否 | Gun 'horiz' (60,535, width 680) (bundle 特征, 本轮未重开 levels.xml); _validate_levels 拒绝 (collect_alphazuma_55_park_settle_episodes_v1 L224-235, 被采集器与探针复用); 不在 teacher55 / full49 / park-settle / bc49 | R0 | 否 |
| grotto8 (#0 面板外) | grotto | 无 | A | G | 高 | 否 | levels.xml Gun type='vert' (startx 390, starty 100, height 405, "Pacific Divide") — 唯一垂直导轨; revenge_core.from_installed 拒绝; D:\ZumaTraining 无任何引用 (仅 park-settle 时代 coverage 表) | R0 | 否 |

置信汇总: 高 15 / 中 28 / 低 8。被质疑 (复核 refuted 或修正) 18 关: 主因改判 8 关 (city7 C→B, Coast2 C→B, grotto10 B→C/D 共主因, volcano1/5/6/7/8 B→C); 次因修正 5 关 (village6 E 降级, city4 D→E, city10 D→E/D 待定, Coast7 E→C, grotto5 F→C); 置信下调 11 关 (city4/7/10, Coast1/2/4/6/7 → 低; grotto4/5/9 高→中); village6 主因存续但论据被证伪。

## 3. 按病因归类

### 3.1 A 接口天花板 — 10 关
- 双位 hop 被拦截 (5 关, 面板内): village3 (软天花板: 1/2 seed 无 hop 亦胜), city3, Coast3, Coast10, grotto3。
- 移动青蛙 (5 关, 面板外): Jungle5, village9, city8, Coast8, grotto8。

共同机理与证据: `park_settle_action_wrapper.py` L60 MACRO_VERB_NAMES=('wait_hold','fire','swap'), L181 hop_verb_exposed=False; teacher55 hop 需求表 — 7 个双位教师集中 hop-fallback ≥90% 者全负 (city3 1157/1273=91%, grotto3 460/470=98%, Coast3 449/485=93%, Coast10 426/451=94%, teacher12 village3 738/821=90%), ≤18% 者全胜 (teacher55 village3 74/418=18%, volcano3 13/936=1.4%); 同一教师在 per-tick 接口 (hop 可用, park-settle s99081659) 全部 3/3 (有效 hop: city3 32/32/55, Coast3 33/21/29, Coast10 76/57/51, grotto3 40/26/33, village3 17/4/20)。移动青蛙 5 关: Gun type 'horiz'/'vert', `revenge_core.from_installed` L1303-1317 fail-closed (Jungle5 复现 NotImplementedError, 0 tick), 模拟器无导轨运动学 (shooter_positions 为静态元组 L954-989; word-boundary grep horiz/vert/slider/rail 为空), 动作空间仅角度 (`revenge_env.py` L327-338), prereg + audit 冻结, 0/220 运行目录。

已验证解法: **R0**。campaign memory: 所有 hop 被拦截的双位负关至今未过; 无任何带 hop 动词的接口被训练过。代价性质: 接口 / 模拟器工程 + prereg 修订 (非训练成本); 之后才是 teacher-through-macro 探针 → 采集 → BC 的标准流程。Jungle5 为导轨类中最短 (partime 4000, 目标 2250), 是打通该类的自然首选 (推断)。

### 3.2 G 从未评估 — 1 关: volcano3
唯一 macro 教师数据是一场胜局 (teacher55 13029 ticks, 13/936 hop fallback), 因 DUAL_POSITION_LEVELS 规则从未采集 / 训练 (prereg L71 明记 "WON ... nonetheless remain EXCLUDED"); per-tick 3/3。A 是种子依赖风险 (village3 一胜一负), 不是已观测的天花板。解法: 16 seed teacher-only 探针 (P-volcano3) 决定它属于 A 还是 R1 轨道; 后者需 prereg 修订。

### 3.3 B 弱初始化 — 29 关 (三种闭环签名)
(a) **不开火 / 欠火 regime split** — 10 关: village1, village5, village10, city1, city6, Coast5, Coast9, volcano2, volcano4, volcano10。
共性: 离线 ratio > 1 (1.08-1.48; Coast5 0.86 为唯一例外) 而闭环 fire 仅 0.11-0.66× 教师; 零射集: village10 4/8, village1 2/8, volcano4 2/8, Coast5 1/8; 崩溃始于 in-distribution 早期状态 (village10 教师首射 decision 4, village1 25-26; volcano4 零射集死亡 tick == 无阻漏斗 tick 3984); 自有 invfreq 与应用权重之比 ≤ 1.26× 且多数方向相反 → E 非机制, 只是杠杆。
已验证解法 **R1**: 直接收据 J3 (2/8 含零射 → 16/16 零零射), J10 (1/8 fire 0.113 → 10/16); 240 集在每个锚点 (含 J7-240 / J9-240) 都消除了零射集。风险: volcano2/4/10、Coast9 时程 645-1115 决策/集, 而 J7-240 (283 决策/集) 在零射消除后仍 0/16 → 预期需要 R2 第二阶段。

(b) **触发过度** — 7 关: village4, village7, village8, city4, city10, Coast4, Coast7。
共性: 闭环 fire 1.24-2.5× / swap 1.3-2.2× 教师; 离线 ratio 1.25-2.19, precision 0.42-0.52 (village8 0.49); 清场阶段死亡 (village4 7/8, village7 3/8)。
已验证解法 **R2** (J7 路线): 240-ep 独用在此签名类无成功收据 (J7 0/16, J9 0/16); J7 beta05 4/16 → r2 10/16 (r2 同时重算权重 [1,4.42,10.03])。village8 (2/8 已有胜场) 可能 R1 即够 (J3/J10 收据), 之后 R3。city4 / city10 (双曲线) 的残余为 D, 无验证解法。

(c) **aim 头缺陷 / 长存活不能完成** — 12 关: village2, city2, city5, city7, city9, Coast1, Coast2, Coast6, grotto2, grotto4, grotto5, grotto9。
共性: 闭环 fire 0.85-1.31× (校准), aim_w3_fire 0.50-0.81, 分/射 0.09-0.61× 教师; 或存活超教师时程但清场失败 (village2 4/8, city5 3/8, Coast2 3/8, city7 4/8)。
已验证解法: **R1** 为必要首步; 是否需要 R2 由时程与探针决定。Coast1/2/4/6 与 city7 的 B-vs-C 顺序为低置信 (共享同一 40 集混杂); grotto4 / grotto9 带 X 共候选。

**代价 (R1, 每关)**: 240 × 该关教师中位 tick, 按实测单关采集吞吐 136-243 ticks/s (jungle3/7/8/9/10-240 completion.json total_native_ticks / wall_seconds, parallel_envs 12, cpu; 非 full49 批量的 1007 ticks/s): village 1.0-2.4M ticks → 1.2-5.0 h; city 1.3-3.0M → 1.5-6.2 h; coast 1.8-3.6M → 2.1-7.3 h; grotto 1.8-4.2M → 2.1-8.5 h; volcano 2.2-4.0M → 2.5-8.1 h wall (逐关串行)。BC: bc49 40 集 101-220 s GPU (wall_seconds), 240 集约 6× 行数 (推断)。面板评估: 16 集 × ~40 s/集 (eval8 收据) ≈ 10-20 min CPU。**R2 每轮**: 160 seeds 采集 (macro-dagger-jungle7-beta05 config seeds_per_level 160) ≈ 与 R1 同量级, 10k+ tick 关约为丛林轮的 2×, 再合并重训。

### 3.4 C 长时程漂移 — 8 关 (+ grotto10 共主因)
grotto1, grotto6, grotto7 (原诊断即 C); volcano1, volcano5, volcano6, volcano7, volcano8 (复核由 B 改判)。
共性: 离线校准与获胜的 bc49 / 240bc 锚点持平或更好 (volcano6 训练 loss 3.034 为 49 关最低; grotto1 / grotto6 aim_w3_fire 0.78; volcano1/5/7/8 射速 = 教师), 闭环存活至教师中位时程的 40-100%, 分/射 0.65-0.80× (grotto) 或转化 7-12% (volcano), 部分集进入清场阶段后仍负; 关卡时程 711-1211 决策/集 = 1.6-2.8× Jungle9。全域时程分裂 (49 个 bc49 联表): <645 决策/集 的 20 关 40/160 eval8 胜 vs ≥645 的 29 关 1/232 (仅 grotto1); corr(eval8 胜, 决策/集) = -0.61, 而 corr(胜, verb_acc) = -0.44, corr(胜, aim_w3_fire) = +0.31。
保留意见 (复核): 时程在丛林收据中非单调 (J7 283 失败, J8 299 / J10 294 成功); J9 的 "决策累积" 读法已被其双曲线附录取代 → 无 >439 决策/集 的 240-ep 先例, C 主因是延伸而非直接收据, 故 8 关皆中置信。
已验证解法: **R1** (必要, 提供会开火的学生) → **R2** 从最佳存活 BC 迭代 (J7 0→0→4→10; J9 0→4/16) → **R3** 仅从获胜初始化。禁: 从 0 胜学生 plain DAgger (jungle9-daggerbc fire 0.019 / swap 0.735, 0/16)。代价: 每关 R1 (grotto / volcano 2.5-8.5 h wall) + 每轮 R2 约为丛林轮的 2×。

### 3.5 D 双链调度 — 2 关 (+ grotto10 共主因): village6, volcano9
六个 curve_count=2 关 (Jungle9, city4, city10, grotto10, village6, volcano9) bc49 eval8 合计 1/48 胜, 学生 swap 0.26-0.34 (教师 0.137-0.149), 中位死亡 2144-4764 ticks; 单曲线 43 关中 swap > 0.2 仅 2 关 (Jungle7 0.286, village4 0.25)。全局 swap 权重 8.3 对所有双曲线关超权 ~2× (E 与 D 纠缠), 但 Jungle9 r2/r3 以近自有权重 [1,2.83,4.12] / [1,2.87,4.09] 仍 churn (swap 0.21 / 0.41, 4/16, 3/16) → 权重修正不治 D。volcano9 叠加 F (教师 27/40, 三个教师源皆负过) 与数据卫生 (13/40 负集入训练)。
已验证解法: **R0** (J9 五次进攻后 park, 最佳 4/16; 唯一产生胜场的杠杆是迭代 beta-DAgger)。

### 3.6 E / F / H / X 的位置
- **E**: 无一关为主因; 作为次因 / 信号出现于 12 关 (+ city10 待定); 作为机制被 "自有 invfreq vs 应用权重" 对比排除 (差 ≤ 1.26× 或方向相反); 作为杠杆已验证 (bc-jungle1/2 无权重 0/8 → invfreq 8/8, 7/8), 但 strong [1,8,12] / heavy [1,16,16] 臂从未闭环评估 (离线 ratio 2.2, precision 0.4)。应用权重 [1,3.7,8.3] 实为 s99081689 (Jungle1+Jungle2) 的 invfreq, 非 full49 全局 ([1,3.47,6.92])。
- **F**: volcano9 次因 (教师 27/40); Coast7 次要天花板 (39/41); grotto5 / grotto10 复核降为后期注记 (负集皆为长局高分); 其余 45/49 关教师 ≥ 97.5%。
- **H**: 51 关全部排除 (eval8 model sha == final sha 49/49; epoch_20 与 final 成员哈希一致; 0 截断; in-scope 0 mask_fallbacks; max_ticks 30000 > 教师最长胜 22267; 移动青蛙关的 loader 拒绝是预注册的设计行为而非 bug)。
- **X**: grotto4, grotto9 共候选 (离线中游 aim 却闭环效率差同型关 3-5×; 30k 行仍 holdout aim 最差); city6 / city9 的曲线几何 / 隧道再现问题未排除。

### 3.7 争议与置信汇总
- 复核 refuted 或修正 18 关: 主因改判 8 (city7, Coast2, grotto10, volcano1/5/6/7/8); 次因修正 5 (village6, city4, city10, Coast7, grotto5); 置信下调 11 (city4/7/10, Coast1/2/4/6/7 → 低; grotto4/5/9 高→中); village6 主因存续但论据被证伪。
- 最终置信: 高 15 关 (全部 A/G 11 关 + village1, village5, village8, city2), 中 28 关, 低 8 关 (city4, city7, city10, Coast1, Coast2, Coast4, Coast6, Coast7)。
- 低置信的共同根源: 每个非丛林学生都是同一种 40 集初始化, 因此 "弱初始化" 的离线指标在 B 与 C 两种假设下同样成立; 只有把 40 集混杂去掉 (R1 首步) 或用教师影子探针看分歧的时间结构才能分离。

### 3.8 复核中发现的跨关卡修正 (供主席校准既有认知)
1. 权重来源: [1,3.7,8.3] = s99081689 (Jungle1+Jungle2, 9701/2610/1168) 的 invfreq, 非 full49 全局 (919474/265096/132850 → [1,3.47,6.92]); 对双曲线关 swap 超权 ~2×。
2. "J8 → 14/16" 不是弱初始化拯救收据 (bc49 J8 eval8 = 7 胜 + 1 截断 (seed 1495002001 30000 ticks, 749 shots)); 240-ep 独用的拯救收据只有 J3 (2/8 → 16/16) 与 J10 (1/8 → 10/16), 均为欠火签名、223-295 决策/集; J7 (283) 与 J9 (439) 240-ep 独用均 0/16。
3. J7 链的 r2 步同时改了权重 (bc-jungle7-beta05r2-s99081758-v1 [1,4.42,10.03]) → "迭代 beta-DAgger" 是数据 + 权重重算的组合被验证, 不是 DAgger 单独。
4. J9 park 裁定的 "决策累积" 诊断已被其双曲线附录取代; 时程在收据中非单调 → 以时程外推 C 的主张只能中 / 低置信。
5. 离线指标不判别闭环 (49 关: corr(胜, verb_acc) = -0.44, corr(胜, aim_w3_fire) = +0.31, corr(胜, ratio) = +0.05, corr(胜, 决策/集) = -0.61); "loss 仍在下降" 对包括 8/8 的 Jungle4 (mean_loss 5.55) 在内的每个 40 集 BC 都成立。
6. 采集成本在 volcano 诊断中被低估 4-7×: 单关 240-ep 采集实测 136-243 ticks/s (非 full49 批量 1007 ticks/s) → 2.5-8 h wall / 关。
7. 面板混淆: eval8 = seeds 1495002000-007, hold 8; timepolish 收据 = seeds 1494001000-015, hold 4 (J3-240 hold A/B 0.985 vs 0.9885 界定 hold 效应; 种子集差异未受控; 无任何 coast 关在 eval8 种子上有教师对照)。
8. record-jungle3 2946/3000 (98.2%) 与 record-jungle7 1836/3000 均为 golden-opening seeds (seed_list_tas) 上的成绩 (后者还是 PPO ckpt786360), 非随机面板; J3 的拯救收据是 16/16 面板。
9. 教师负集未过滤进入 bc49 数据 (distill L676-757 只计数): volcano9 13/40, grotto10 7, grotto5 5, volcano2 3, city4 2, Coast7 / Coast9 / grotto9 / volcano7 / volcano10 各 1; 但战役实践 (jungle10-240 含 4 截断集 → 10/16) 表明这不是原因。
10. 引用误差已订正 (不改标签): village2 aim 排名第 3 非第 2; village4 7/8 (非 6/8) 高分死亡; city3 park-settle hop_intents 1293 存在 (raw 343/347/603), 诊断称 "未找到" 有误; city5 仅 1/3 晚死 seed 分数超教师中位; Coast2 4/8 (非全部) 超教师胜时; grotto1 闭环 fire 0.226 非 0.232; grotto4 fire 0.151 非 0.157; grotto10 时程为 49 关第 4 非最长; volcano8 漏斗参考 4176 为下界 (probe55 该集射 6 发); Jungle5 诊断把 "九个兄弟关经宏接口全胜" 误引自 closing summary (应引 teacher55 per_level_outcomes)。

## 4. 结论

51 个未攻克关卡分成两个互不重叠的世界。11 关 (10 A + 1 G) 从未被战役工具触及: 5 个双位关的 hop 被宏契约拦截 (同一教师在 per-tick 接口 3/3 全胜, 经宏接口 3-17 射即负), 5 个移动青蛙关被模拟器 fail-closed 拒绝, volcano3 仅因范围规则未采集 —— 这里没有学习问题, 只有接口 / 范围决定。其余 40 关都有一个真实的 40 集 bc49 学生, 环境 / 教师 / 工具在其中全部被排除 (教师 ≥97.5% 于 45/49 关, 0 fallback, 0 截断, sha 一致; 唯一例外 volcano9 教师 27/40), 失败全在数据侧: 29 关为 40 集单 checkpoint 初始化太薄 (动词头按种子翻入不开火或触发过度, aim 头近乎随机), 8-9 关为学生在 700-1200 决策/集的关卡上先如教师般打几百手再漂入走廊之外 (含清场阶段), 2-3 关为双链调度 —— 战役唯一未治愈的病。已验证解法阶梯 (R1 240-ep 初始化 → R2 含权重重算的迭代 beta-0.5 DAgger → R3 仅从获胜初始化的 PPO 打磨) 覆盖 B 与 C 但不覆盖 A 与 D; 最大的未决问题是 coast / volcano / city 长关上 B 与 C 的先后 (18 关被复核质疑, 8 关低置信), 它们共享同一个混杂 (每个非丛林学生都是同一种 40 集初始化), 只能由廉价的教师影子探针预判, 或由 R1 首步 (即施工第一步) 决定性分离。

## 5. 建议的最小探针清单 (全部为建议, 均未启动)

| 探针 | 关卡 | 问题 | 探针设计 (工具 / 参数) | seeds (panel16) | 预计成本 | 读数判据 | 来源 |
|---|---|---|---|---|---|---|---|
| P-village6 | village6 | D vs E | 在相同 40 集 full49 数据上以 village6 自有 invfreq 权重 [1.0,2.38,4.14] 重训 bc49 配方 (其余不变); probe_alphazuma_55_macro_native_eval_v1, hold 8, max_ticks 30000; 对照 bc49 eval8 (swap 0.280, fire 0.302, 中位 3564, 0/8) | 1494001000-1494001015 (16 集) | GPU ≈ 220 s (bc49 village6 wall_seconds) + CPU 16 集 ≈ 10-15 min | 动词回归 ~0.15/0.25 且存活 / 胜场改善 → E; 动词归一但仍在 ~1/3 教师时长死亡且 0 胜 → D (J9 r2/r3 预测后者); 同一探针跑 Jungle9 可回溯判别 (可选) | 诊断 + 复核 |
| P-city4 | city4 | E vs D vs B | 教师影子回放: collect_alphazuma_55_macro_dagger_decisions_v1 --student-model bc49/city4/final_model.zip --teacher-drive-prob 0.0, hold 8; 读数 (i) 每次射击的目标曲线 id vs 教师标签 [需探针侧新增 per-curve 日志 — 现有输出无此字段]; (ii) 推理时先验平移对照 (verb logits − log[1,3.7,8.3]) 同 seeds [需探针侧小改动] | 1494001000, 1494001001 (2 臂 × 2 = 4 集) | CPU 4 集 ≤ 4000 ticks ≈ 3-6 min; 无训练 | swap 0.30 → ~0.14 且存活延长 → E; 学生打非临界链而另一链爬升 → D; 首决策起 aim 误差均匀高且无上述模式 → B | 诊断 + 复核 |
| P-city7 | city7 | B vs C | 教师影子回放 (同上, teacher_drive_prob 0.0), 记录逐决策一致率 / aim 误差与死亡时链长 (对照 full49 city7 走廊最大链长) | 1494001000-1494001002 (3 集) | CPU 3 集 ≤ 17000 ticks ≈ 5-10 min | 前 ~300 决策一致率接近离线水平后分歧上升 → C; 从头高分歧 → B | 诊断 + 复核 |
| P-city10 | city10 | E vs D vs B | 同 P-city4 (per-curve 日志 + logit 先验平移); 另作只读代码检查: 特征提取器是否受 fog (effect1) 影响 | 1494001000, 1494001001 (4 集) | CPU 4 集 ≤ 6600 ticks ≈ 4-8 min | 同 P-city4 | 诊断 + 复核 |
| P-Coast1 | Coast1 | checkpoint 不稳定子命题 | 零采集预检: 用 full49 现有 40 集 npz (manifest s99081698, level_filter Coast1) 重训 bc49 配方并保存每 epoch (checkpoint_interval_epochs=1); 取 holdout fire_ratio 最接近 1.2 的 epoch 与 epoch-20 对照, probe_alphazuma_55_macro_native_eval_v1 hold 8 | 1494001000-1494001003 (2 臂 × 4 = 8 集) | GPU ≈ 3-4 min (探针级重训, 无新采集) + CPU 8 集 ≈ 6 min | 仅回答 "出厂 epoch 是否近因"; B-vs-C 决定性判别 = 240-ep BC + 面板 (= 施工首步, 不列入) | 诊断 (复核: 读数须作判别而非 "B 确认") |
| P-Coast2 | Coast2 | B vs C (coast 区代表) | 教师影子回放 (teacher_drive_prob 0.0), 记录逐决策一致率 + 死亡时分数 vs 目标 4650 [综合者建议, 与 grotto / volcano 探针同构; 原诊断只给出 240-ep 判别] | 1494001000-1494001003 (4 集) | CPU 4 集 ≤ 13000 ticks ≈ 5-10 min | 首 50 决策即高分歧 → B; 分歧集中于 >400 决策或跨越 4650 之后 → C; 决定性判别 = 240-ep Coast2 BC (coast 最短关, 2.2-3.9 h 采集, = 施工首步, 暂缓) | 综合者 |
| P-Coast4 | Coast4 | checkpoint 选择子解法 | 同 P-Coast1 (per-epoch 重训, ratio≈1.2 epoch vs epoch-20) | 1494001000-1494001003 (8 集) | GPU ≈ 4-5 min + CPU 8 集 ≈ 5 min | ≥2/8 或死亡时程显著延长 → checkpoint 选择为近因 (配方侧 B); 无变化 → 数据量 (B/C 待 240-ep) | 诊断 + 复核 (子解法无收据) |
| P-Coast6 | Coast6 | B vs C | 教师影子回放 (teacher_drive_prob 0.0), 逐 1000-tick 窗口 fire 分数与 aim 误差 [综合者建议] | 1494001000, 1494001001 (2 集) | CPU 2 集 ≤ 7000 ticks ≈ 2-4 min | 首窗口即近随机 aim 误差 → B; 前期一致后期分歧 → C | 综合者 |
| P-Coast7 | Coast7 | F 上界 | 教师对照: probe_alphazuma_55_macro_teacher_v1 --levels Coast7 --seed-base 1494001000..1494001015 (16 次调用, 每次 1 集; 该探针一次仅跑 seed_base+index), hold 8, max_ticks 30000 | 1494001000-1494001015 (16 集) | CPU ≈ 1-3 min (CPU 教师; teacher55 55 集 241.6 s) | ≤12/16 教师胜 → F 升为次因并封顶学生胜率; 手工重加权臂 [1.0,2.5,6.0] 标未验证, 不列入 | 诊断 (复核保留) |
| P-grotto4 | grotto4 | B vs X | beta=0.5 混合回放 (dagger collector teacher_drive_prob 0.5, bc49/grotto4/final_model.zip, hold 8), 记录首 50 决策 aim 分歧 | 1494001000, 1494001001 (2 集) | CPU 2 集 ≤ 8000 ticks ≈ 2-4 min | >40% 学生射击与教师差 >3 bin → B; 分歧低而链仍失控 → X (表征 / 几何) | 诊断 + 复核 |
| P-grotto5 | grotto5 | B (首 50 决策) | 同 P-grotto4; 可选离线伴随: 剔除 5 负集重训比较 holdout aim_w3_fire / precision (GPU ≈ 3 min, 复核预期变化很小) | 1494001000, 1494001001 (2 集) | CPU 2 集 ≈ 2-4 min (+ 可选 GPU 3 min) | 首 50 决策 aim 分歧高 → B 确认 | 诊断 + 复核 |
| P-grotto9 | grotto9 | B vs X | 同 P-grotto4 (首 50 决策 aim 分歧); X 升级判据 (240-ep holdout aim_w3_fire < 0.65) 需 240-ep BC, 暂缓 | 1494001000, 1494001001 (2 集) | CPU 2 集 ≈ 2-4 min | 同 P-grotto4 | 诊断 + 复核 |
| P-grotto10 | grotto10 | D vs E vs B | 学生单独回放 (probe_alphazuma_55_macro_native_eval_v1, hold 8) 4 集, 新增 per-curve 链长 / 到达骷髅的曲线 / swap 时间线日志 [需探针侧只读仪表] | 1494001000-1494001003 (4 集) | CPU 4 集 ≤ 8000 ticks ≈ 3-5 min + 仪表改动 | 失败曲线恒为未被瞄准的那条 → D; 两链从头同时失控 → B; swap 突发先于崩溃 → E | 诊断 + 复核 |
| P-volcano1 | volcano1 | B vs C | 教师影子回放 (dagger collector teacher_drive_prob 0.0, bc49/volcano1/final_model.zip), 由 sidecar label_agreement 算逐决策一致率 | 1494001000-1494001003 (4 集) | CPU 4 集 ≤ 5000 ticks ≈ 3-6 min | 首 30 决策一致率 <0.6 → B; ≥0.8 后崩 → C (决定性: 240-ep, 暂缓) | 诊断 + 复核 |
| P-volcano3 | volcano3 | A vs B 轨道 | 教师单独宏探针: probe_alphazuma_55_macro_teacher_v1 --levels volcano3 --seed-base 1494001000..1494001015 (16 次调用), hold 8, max_ticks 30000; 记录胜负与 hop fallback | 1494001000-1494001015 (16 集) | CPU ≈ 1-2 min | ≥12/16 → 宏可达 (G → R1 轨道, 需 prereg 修订); ≤8/16 且负局多 hop fallback → A | 诊断 + 复核 |
| P-volcano5 | volcano5 | B vs C | 教师影子回放, 分别统计 verb 一致率与 aim-within-3 一致率随决策序号 | 1494001000-1494001003 (4 集) | CPU ≈ 3-6 min | verb 一致而 aim <0.6 从头 → B (aim 欠训练); 两者前高后崩 → C | 诊断 + 复核 |
| P-volcano6 | volcano6 | B vs C | 教师影子回放 6 seeds (该关按种子方差最大 5182-15567 ticks), 记录一致率与死亡阶段 (分数 vs 目标 5100) | 1494001000-1494001005 (6 集) | CPU 6 集 ≤ 16000 ticks ≈ 8-15 min | >500 决策后才崩 / 计量满后死亡 → C 已在起作用; 从头 <0.6 → B | 诊断 + 复核 |
| P-volcano7 | volcano7 | B vs C | 教师影子回放; 另记死亡是否早于 300 决策 (B 形) 或在与教师射速一致 >400 决策之后 (C 形) | 1494001000-1494001003 (4 集) | CPU 4 集 ≤ 6000 ticks ≈ 3-6 min | 同 P-volcano1 | 诊断 + 复核 |
| P-volcano8 | volcano8 | B vs C | 教师影子回放, 比较射击决策上学生执行 aim bin 与教师 aim 标签随决策序号 | 1494001000-1494001003 (4 集) | CPU 4 集 ≤ 5000 ticks ≈ 3-6 min | aim-within-3 一致率 <0.6 从首 30 决策 → B; 前高后崩 → C | 诊断 + 复核 |

合计: 约 113 集 (全部 seeds ∈ 1494001000-1494001015), CPU 约 1-2 h (单进程串行的上限估计), GPU 约 10-12 min (三个探针级重训: village6, Coast1, Coast4)。

执行政策 (全部未启动):
- 本清单为建议; 任何启动需主席另行批准; CPU 为 AlphaDiablo 战役保留, 在其运行期间不排程。
- 种子只取自 panel16 (1494001000-1494001015); 不消耗任何正式 seed 块 (formal_seed_consumption=false)。
- 三个探针 (P-village6, P-Coast1, P-Coast4) 含探针级 GPU 重训 (无新采集, 2-5 min); 三个探针 (P-city4, P-city10, P-grotto10) 需探针侧只读仪表改动 (per-curve 日志 / logit 先验平移), 不改训练与环境。
- 决定性 B-vs-C 判别 (240-ep 采集 + BC + 16 面板) 与已验证解法首步 R1 重合, 属施工, 不列入本清单; 若主席批准施工, coast 区以 Coast2 (最短, 579 决策/集, 采集 ≈ 240 × 7.9k = 1.9M ticks ≈ 2.2-3.9 h) 为区级代表, volcano 区以 volcano2 (最短, 645 决策/集, ≈ 2.2M ticks ≈ 2.5-4.5 h) 为代表。
- hold 约定: 探针建议 hold 8 (与 bc49 训练 stack [0,4,8] 及 eval8 一致); 丛林 timepolish 收据为 hold 4, 比较时需注明。
- 不建议的探针: 从 0 胜学生做 plain (beta=0 训练用) DAgger (jungle9-daggerbc 不开火毒性); 任何 allow_partial_level=True 的移动青蛙关运行 (代码自述 non-transferable)。
