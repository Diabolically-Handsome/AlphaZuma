# Gate repair audit — 2026-08-09

## 裁决

两道 Gate 的标准缺陷已修复，但没有通过降低物理一致性要求来强行开门，也没有
启动训练。

- Fidelity Gate v4：`CLOSED`；来源、分布、接口三条 lane 为 `PASS`，动力学
  lane 为 `MISSING`。
- Training Gate v3：`CLOSED`；环境合同、模型迁移、历史稳定性证据均通过，
  唯一关闭原因是所绑定的 Fidelity v4 尚未 `OPEN`。

## Fidelity Gate 修复

1. v3 要求 Jungle2 自然产生 slow powerup，但原版该关 CURV 对 slow 的权重为
   0；v4 以新 policy generation 移除这一不可达条件，旧策略保持不可变。
2. 原版长轨迹中 `ball_id` 会复用。轨迹加载器现以 `(ball_id, native address)`
   区分同帧实体，并拒绝无法消歧的重复身份；native address 不写入便携语义输出。
3. C122 601-tick 来源经过该规则重新包装和机制审计。它只认证真实观察到的
   `projectile_collision`、`rng_pending`、`shot_release`、`swap` 与长时来源，
   不把“接近终局”误报成 `natural_win`。

当前全量报告：

- suite：`D:\ZumaGolden\diagnostics\fidelity-suite-c122-candidate-v9.json`
  (`sha256:3d6aebae587a593496f7381d758f90a2a7804cbc525475dd8b6c3b20f38e93f3`)
- report：`D:\ZumaGolden\diagnostics\fidelity-gate-c122-candidate-v12-report.json`
  (`sha256:01ec52d02b63e9cf0f28694610d67bc25f5ec16501941eb39723b88faf37f673`)

已通过 15 项：

`back_insertion`, `front_insertion`, `input_cadence`, `match3`, `match4`,
`projectile_collision`, `rng_pending`, `shot_release`, `swap`,
`long_horizon_drift`, `startup_actor_distribution`, `actor_no_hidden_state`,
`fruit_asset_geometry`, `fruit_visual_oscillator`, `fruit_actor_observation`。

仍缺 16 项：

`gap_shot`, `natural_loss`, `natural_win`, `powerup_proximity_bomb`,
`powerup_reverse`, `powerup_spawn`, `rng_rejection`, `rollback_chain`,
`tunnel_collision`, `zuma_transition`, `fruit_scheduler_spawn`, `fruit_expiry`,
`fruit_projectile_collision`, `fruit_powerup_collision`,
`fruit_collection_score`, `fruit_collection_animation`。

这些是有效的证据缺口，不再是来源认证或 Gate 逻辑造成的伪失败。

## Training Gate 修复

1. 历史 v1 将 98,304-step 稳定性证据外推到 491,520 步；现保留为历史策略，
   不授权迁移。
2. v2 把整个 Python 包当作训练实现闭包，导致 PC 取证代码改动连带作废环境
   合同和模型迁移；现保留为历史策略，不授权迁移。
3. v3 只绑定模拟器与策略的显式运行时依赖闭包，仍绑定原版二进制、依赖版本、
   actor interface、环境指纹和确定性 KAT；授权上限固定为 98,304 步，外推倍数
   固定为 1.0。

当前内容寻址证据：

- environment contract v9：
  `sha256:c8923b265dffdd5fc471dc186ae11a0dee9105c54076d9a986e7f9e5864b6f24`
- model migration v4 (`PASS`)：
  `sha256:99de5c36607493d4c10865dd8b206a1aad6328255d71597fb2b086dab5ab559f`
- Training suite v6：
  `sha256:655cd677fcf2ae42cf10537267dfd685c11bf5ae1ec86d446451c9c5d498fed6`
- Training report v6 (`CLOSED`)：
  `sha256:8e76b3b8cd093e1b3ca70cba7e5d05d596ca5b755605d204b9c066d6d9192973`
- migrated actor model：
  `sha256:32e1e841962833b3c74ce743598df1c87e21c9c9c931092152e8bc2700081f64`

## 验证

- Windows 全量回归：1,343 tests collected，1,330 passed，13 skipped；
- WSL 完整训练依赖下的 Gate 聚焦回归：`PASS`；
- WSL Training Gate 全量实时重算：只有
  `bound Fidelity Gate is not OPEN on all v4 evidence lanes` 一项关闭原因；
- 本轮未创建训练 run，未占用 GPU 执行 PPO 更新。
