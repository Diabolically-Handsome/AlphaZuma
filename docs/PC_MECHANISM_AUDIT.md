# 原版 PC 机制审计

`zuma-audit-pc-mechanisms` 用来回答一个窄问题：某段已经通过 PC Golden
双回放验收的原版来源，是否真的在指定 update 窗口中展示了某个游戏机制。

它不是模拟器差分，也不替代 PC Golden。Fidelity Gate 只有在以下三者同时
成立时，才把机制计到 `pc_golden` 一侧：

1. PC Golden Manifest v4 在同一次 Gate 运行中通过完整验收；
2. 机制审计报告现场重算为 `PASS`；
3. 两者的 native-source 指纹完全相同。

## 计划不允许声明 feature

计划 schema 为 `zuma-rl.pc-mechanism-audit-plan` v1，每个案例只包含：

```json
{
  "id": "c2-shot-window",
  "manifest_path": "cases/<case>/manifest.json",
  "memory_probe_path": "diagnostics/<probe>/memory-probe.json",
  "window": {
    "start_update": 6980,
    "end_update": 7480
  }
}
```

出现 `features` 或其他未知字段会直接失败。授权 feature 只能由原始数据重算。

## 现场复核链

每个案例会重新检查：

- manifest、DMO、runtime 和 probe 的内容哈希；
- probe 的冻结状态与原始 Board／Ball／Bullet 字节；
- `diagnostic_mutation=null` 和完整 repaint guard；
- 声明分数或冻结后只读发现的 Board score、displayed score 与扫描值完全一致；
- schema v2 的逐 tick replay barrier、连续 update 和每个 tick artifact；
- probe 首帧与 PC Golden 对应 native tick 的原版录像 gameplay 区逐像素一致；
- 事件身份、先后顺序、颜色、精确爆炸数量和分数变化。

当前自动授权规则是：

- `shot_release`：同 tick 出现 shooter-current → fired、projectile spawn 和
  chamber advance，且 ball identity 一致；
- `projectile_collision`：同一已发射 projectile 后续进入原版 insertion
  staging list；
- `match3` / `match4`：同色、精确 3 / 4 个唯一 ball 同 tick 开始爆炸，
  与 insertion commit 或 rollback stop 身份耦合，并且原版 score 增加；
- `rollback_chain`：同一 ball 先开始吸回，后停止吸回并参与带 combo 的计分
  爆炸。

front/back insertion、gap/tunnel、powerup 物理效果、RNG rejection 和自然
胜负仍需要各自更强的机器规则；在规则实现前，工具不会因为计划或 suite 中
写了名称就授权。

## 执行

```bash
zuma-audit-pc-mechanisms \
  diagnostics/pc-mechanism-plan.json \
  --evidence-root /mnt/d/ZumaGolden \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
```

退出码 `0` 表示计划内所有来源绑定和至少一个机器推导机制通过；`1` 表示
证据有效性不足或没有可授权机制。报告仍需作为 `audit` 证据加入 suite，
其 suite feature 固定写为 `pc_mechanism_coverage`。
