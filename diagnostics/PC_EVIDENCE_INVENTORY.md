# 原版 PC 证据盘点

盘点日期：2026-07-30  
只读来源：`D:\ZumaGolden`

## 结论

- 当前仍只有 **1 个**符合 `original-transfer-jungle2-v1` 范围、且能通过
  Evidence-v4 验收的 PC Golden native source。
- 原始素材中存在 **3 个 DMO 随机种子**，但只有两个可进入认证路线。种子
  5,909,046 的 pilot 在 update 145 失活，确定性播放无法推进到后续激活命令，
  只能作为解析／导航诊断；仍需新录一个干净 DMO 才能满足第三个认证 seed。
- 已有最长的自然、完整 gameplay-memory 轨迹不足 500 tick。另有 501 和
  2048 tick 的 RNG 轨迹，但它们没有逐球 waypoint/位置，不能冒充完整物理
  漂移证据。
- 旧的双回放采集全部来自同一个 `e43c...` DMO。只有
  `pc-golden-v4-fixed-thread-crt-attach0-20260730-062733` 得到可用的连续
  逐像素确定性区间，最长为 78 tick。

## DMO 种子

| 种子 | 代表 DMO SHA-256 | 长度 | 当前状态 |
|---:|---|---:|---|
| 36,311,796 | `e43c645a...18b12` | 17,967 tick | 已有一个 Jungle2 PC Golden |
| 23,775,218 | `044fe875...bc10a` | 8,942 tick | 有 DMO/预存档，无完整双回放 |
| 5,909,046 | `8f7b6244...8b421` | 407,286 tick | focus-deadlocked pilot，仅 diagnostic，不得认证 |

其余 DMO hash 是对上述轨迹做过对齐、移除服务命令、点击注入或诊断修改的
派生文件，不能作为新的随机种子。

## 双回放采集

一共发现 30 个 `pc-golden-v4-*` 尝试，其中 6 个 collection 标记为
`complete`。三个保留了编码视频，三个只保留了帧表和 update map。

| Collection | 共同稳定 update | 精确像素匹配 | 最长连续匹配 |
|---|---:|---:|---:|
| `fixed-thread-crt-attach0` | 299 | 282 | 78 tick，7693–7770 |
| `phased-trace` | 296 | 0 | 0 |
| `sendinput-poll` | 26 | 8 | 3 tick |
| `fixed-all-rng` | 300 | 0 | 0 |
| `fixed-board-seed` | 287 | 0 | 0 |
| `fixed-crt-seed` | 298 | 0 | 0 |

现有四个 `cases/*/manifest.json` 都复用了同一 DMO、同一对视频和同一
7693–7770 窗口；三个 Jungle1 包只是历史打包版本，不是独立原版轨迹。
Fidelity Gate 已改为按原始 DMO、两次视频、DXGI 元数据和 update map
生成 native-source 指纹，重新打包不会增加案例数。

## 可复用长轨迹

- 自然完整 gameplay-memory：
  - 9910–10110：201 tick；
  - 11190–11360：171 tick；
  - 9425–9580：156 tick；
  - 6770–6923：154 tick。
- 自然 RNG/state-subset：
  - 6700–7200：501 tick；
  - 9000–11047：2048 tick。
- 9547–9900 的 354-tick full trajectory 和终局/道具触发轨迹含合成或注入
  状态，只能用于诊断，不能认证自然迁移。

## 最小补录矩阵

目标不是随便录七段录像，而是让八个独立 native sources 一次覆盖 Gate
所需机制、视觉恢复样本和三个种子。

| 案例 | 建议种子 | 必须覆盖 |
|---|---:|---|
| C1（已有） | 36,311,796 | input cadence、shot release、确定性回放 |
| C2 | 23,775,218 | projectile collision、front insertion、match3 |
| C3 | 新干净 seed | back insertion、match4、rollback，并满足第三个认证 seed |
| C4 | 三者之一 | gap shot、tunnel collision |
| C5 | 三者之一 | powerup spawn、reverse |
| C6 | 三者之一 | slow、proximity bomb |
| C7 | 三者之一 | zuma transition、自然 win |
| C8 | 三者之一 | 自然 loss |

至少一个案例必须连续 **500 tick**，同时保留双回放视频、逐 tick
gameplay-memory、RNG、输入、score、视觉帧历史和无干预溯源。所有案例都要
采集 actor 特征的视觉恢复标注；不能在 suite 文件中仅靠手写 feature 名称
宣称覆盖。

## 当前下一步

已完成：

- PC visual-derivability 生成器与 fail-closed verifier；当前实测 28/72
  特征可推导，44/72 仍缺，状态正确保持 `INCOMPARABLE`；
- 内容寻址、可重跑的 feature-claim 授权，suite 自报标签不再直接计数；
- CUDA PyTorch `2.12.1+cu130`、RTX 5090 矩阵烟测和 512-step PPO 烟测；
  RTX 5080 保持隔离；
- [`pc-capture-campaign-v1.json`](pc-capture-campaign-v1.json) 及其只读
  验证器：1 个现有案例、3 个已通过 DMO hash／seed／effect-free 窗口校验的
  待采任务、4 个明确需要新录 DMO 的任务。旧 5,909,046 pilot 被机器规则禁止
  用于 ready task。

等原版窗口空闲时，先执行三个 ready task；其余四个任务按 blocker 录制新
DMO，不再临时设计实验。
