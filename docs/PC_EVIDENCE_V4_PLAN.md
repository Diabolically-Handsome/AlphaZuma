# PC Evidence v4 设计与实现记录

> **状态：Manifest v4、生产器与只读 verifier 已实现，并有真实原版案例
> `PASS`。**
>
> v4 的 `save_transaction` 与 `replay_determinism` 已从原始 artifact 独立
> 重算，不再是固定 `INCOMPARABLE`。v3 旧案例仍会保持
> `INCOMPARABLE`。本文件和单个案例的 `PASS` 都不能被当作 Fidelity Gate
> 已打开。

目标是让验收器从原始 artifact 独立重算结果，而不是接受
`restored: true`、`deterministic: true` 或 producer 自报的摘要。

## 当前实证

首个封装案例 `jungle1_dmo_e43c645a18d7_u7693_7770` 使用同一 DMO、同一
pre-state 的两个独立原版进程，覆盖 update 7693–7770（78 个 native tick）。
案例共有 61 个内容寻址 artifact；验收器已经从原始数据独立通过：

- 完整双回放 FFV1 解码、逐 tick 全 viewport RGB24 一致性、DMO 输入和
  tick-map 绑定；
- pre／run-start／run-end／post／restored 状态根、hash-chain journal、
  Windows ProcessStart／ProcessStop 与正常退出；
- update 7738 与 7750 的冻结状态、Board／Shooter／Curve／Ball／Bullet
  原始内存字节和探针 BMP；
- 原始链从 7738 到 7750 保持 94 个球的身份与颜色，每个 curve distance
  增加 1.5；弹丸 ID 104 离膛、速度约 8 px/tick，枪膛 current／next 正确
  轮换；
- 两张探针 BMP 到主视频 native tick 45／57 的逐像素绑定。update 7738 只在
  被明确排除的最底 1 行存在 4 个窗口边缘像素差异，游戏 viewport 内为零；
  update 7750 全帧一致。

这是一条“开火但尚未命中”的窄案例。它证明 v4 证据链能够产生可信的退出码
0，不证明碰撞、插入、消除、gap rollback、powerup、终局或整套模拟器已经与
原版一致。

## 哈希分层

为避免 trace 内嵌 capture-contract 指纹造成循环，v4 使用两层哈希：

1. `capture_contract_fingerprint` 绑定场景、环境、DMO、时钟、坐标、coverage、
   容差、save/replay 语义对象及全部 ArtifactSpec；只排除字节内容必须内嵌该
   指纹的 artifact。排除列表本身必须是闭集并参与哈希。
2. `evidence_set_fingerprint` 绑定 capture-contract 指纹和包含最终 trace 在内
   的全部 ArtifactSpec。任何 artifact 都不得内嵌该值。

最终 Manifest SHA-256 仍需在案例目录以外固化。自包含哈希只能发现内部
不一致，不能认证可同时重写 Manifest 与全部 artifact 的管理员级攻击者。

## `save_transaction`

最小对象引用以下原始 artifact：

- canonical 状态快照：完整 `users` regular-file tree，以及两个固定注册表键的
  value name、type 与 raw bytes；
- append-only canonical NDJSON journal，带 session nonce、sequence、
  previous-record SHA-256、phase、process instance、单调时间与 exit code；
- Windows 原始进程时序证据，由 verifier 支持的 parser 独立读取，不能只提交
  collector 生成的 JSON 摘要。

固定 phase 至少包括：

```text
pre
r1-start -> r1-end
r2-start -> r2-end
restored
```

每个 run start 的复合状态根必须等于 `pre`；两次 run end 状态根必须相等；
`restored` 必须等于 `pre`。状态 bundle 必须排序、禁止绝对路径、`..`、
重复成员、symlink/reparse、加密成员和未声明注册表范围。

## `replay_determinism`

至少引用两个不同游戏进程的完整回放：

```text
同一 EXE／环境／DMO／pre-state
  ├─ r1: video + tick-map + trace + start/end snapshot
  └─ r2: video + tick-map + trace + start/end snapshot
```

verifier 必须自行：

- 完整解码两段 lossless 视频；
- 通过各自 tick-map 归一到 native tick；
- 逐 tick 比较完整 viewport 的规范 RGB24 哈希；
- 去除 run-specific frame index/PTS 后，精确比较 DMO binding、事件顺序、
  measurement status/value 和终局状态；
- 确认是两个独立 ProcessStart/Stop，均正常退出；
- 确认两次 end-state root 相同。

不能只挑关键截图，也不能信任 artifact 中缓存的 `match: true`。派生摘要若被
保存，只能在 verifier 从原始数据重算后作为缓存使用。

## 三态规则

- `PASS`：所有原始证据存在且可重算；phase、进程、恢复、双回放像素／trace／
  end-state 全部满足契约。
- `FAIL`：已经声明的 artifact 缺失、篡改、格式不安全，或重算后出现任何
  phase、状态根、命令、事件 tick、像素或终局差异。
- `INCOMPARABLE`：旧版 Manifest、少于两次独立回放、缺原始进程证据／快照／
  视频／trace，或所需 decoder/parser 不可用。

## 正式采集事务

1. 关闭游戏、Steam 与云同步，重新取得本 session 的 `pre`；现有 preflight
   备份只作为灾难恢复副本。
2. 启动 hash-chain journal 和原始进程时序采集。
3. 每次 run 前恢复 `pre`，立即取得 `rN-start` 并核对状态根；以新进程播放
   同一 DMO，lossless 录制，正常退出后取得 `rN-end`。
4. 至少完成两个 run；run 之间再次恢复并核对 `pre`。
5. 最后在游戏和 Steam 均关闭时保存实验后状态，恢复 `pre`，取得
   `restored`；只有状态根完全一致才能结束事务。
6. 任何中途错误都进入 finally 恢复路径，不删除唯一备份。Steam 在用户确认
   前保持关闭；若之后启用云同步，需再次复核状态根。

该设计仍不能对抗拥有管理员权限、可伪造全部本机原始证据的攻击者；这一信任
边界必须随案例公开。
