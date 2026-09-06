# 共享 Board 的多曲线实现

原版多曲线关卡不是两局祖玛并排运行，也不能把两条 CURV 首尾拼接。当前核心
按一个 Board 建模：射手、瞄准、当前／下一球、分数、游戏时钟、全局 RNG、
自由弹丸和最终 outcome 共享；每条曲线分别保存球链、pending、merge 弹丸、
运动／回退状态、gap latch、连消状态和 powerup 运行计时。

## 原版静态证据

本机零售运行载荷的只读反汇编显示：

- `Board+0x9C` 持有 `CurveManager`；
- manager 的曲线指针数组位于 `+0x16C`，曲线数量位于 `+0x360`；
- manager 更新函数按指针数组顺序遍历所有曲线，先调用每曲线更新，再调用
  每曲线后处理；
- Board 的原生时间计数位于 `Board+0xEC8`，不是每条曲线各有一份；
- 枪、分数和自由弹丸也属于共享 Board 层级。

原版 79 个关卡中有 10 个双曲线关卡。部分关卡还包含移动蛙、多个固定蛙位置、
Boss 或 Iron Frog，这些额外状态机仍会 fail closed；普通固定蛙双曲线关卡
可由 `RevengeSimulator.from_installed()` 作为一个共享 Board 加载。

## 当前实现契约

- `curve_states` 保存每曲线运行状态；兼容接口 `sim.balls` 等在外部仍指向
  curve 0，跨曲线工具使用 `iter_curve_balls()` 等显式接口。
- 每个 10 ms native tick 按曲线数组 `0..N-1` 更新，且只推进一次 Board
  时钟、射手和自由弹丸。
- 自由弹丸按数组顺序测试曲线，命中后记录 `curve_index` 并进入该曲线的
  merge 队列。
- Zuma 条属于 Board；Reverse／Slow 的持续时间按每条曲线自己的 DAT 参数
  写入。
- 胜利等待所有曲线、pending 和弹丸清空；失败选择数组顺序中首条致命曲线，
  并进入 Board-wide 终局。
- actor observation 展平所有曲线，并给球和弹丸附加曲线 one-hot 身份。
- 回放环境指纹绑定全部曲线几何、tunnel、参数和顺序，不能把单曲线回放误放
  到双曲线环境。

上述行为均有确定性和合成回归，实际 Jungle9 双曲线数据也已完成无渲染 smoke
test。它们仍不是 PC 运行 Oracle。

## 尚待原版裁决

在双曲线范围获得可迁移认证前，至少需要逐 tick 原版案例确认：

- 两条曲线的精确更新／后处理顺序及 RNG 调用顺序；
- 一发弹丸同时接近两条曲线时的碰撞归属；
- 两条曲线参数不同时的 Zuma Reverse／Slow 传播；
- Board 总分目标是否确为每曲线 `zuma_score` 求和；
- 一条曲线清空、另一条仍存活时的胜利相位；
- 任一曲线进洞后的 Board-wide 失败与吸入顺序。

因此，双曲线现在是“结构实现完成、Gate Oracle 未完成”，不会被窄范围
Jungle2 policy 的未来 `OPEN` 自动认证。
