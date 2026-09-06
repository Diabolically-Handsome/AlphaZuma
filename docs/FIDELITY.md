# 原版一致性门槛（Fidelity Gate）

本项目的目标不是做一个“玩法相似”的祖玛，而是先建立一个能够把策略迁移到
Steam PC 版《Zuma's Revenge》的训练环境。因此，环境吞吐量、奖励曲线和简化
环境胜率都不能代替原版一致性验证。任何写着“禁止正式训练”的项目未通过前，
只能进行单元测试、性能测试和算法管线冒烟测试，不能开始会被当作最终策略的
大规模强化学习训练。

当前总状态：**门槛未通过，禁止正式训练。**

门槛现在按训练范围分别认证，而不是用一个布尔值暗示“整款游戏已经完成”。
当前可授权策略是 `original-transfer-jungle2-v4`：只覆盖普通难度 Jungle2、
`tutorials_completed` 存档契约、单曲线和状态 actor view。v1/v2 仅保留历史
复现资格；v3 因包含 Jungle2 不可达的 slow-powerup 要求也已由 v4 取代。v4
通过后仍必须与独立 Training Gate v3 同时通过，且
只允许其固定的有界状态策略阶段；双曲线、其他关卡、视觉输入、Boss、Iron
Frog 和新档教程仍各自需要扩展策略与证据。机器可执行格式见
[FIDELITY_SUITE.md](FIDELITY_SUITE.md) 和 [TRAINING_GATE.md](TRAINING_GATE.md)。

当前静态证据来自本机 Steam 安装，复现实验时应同时记录版本指纹：

- `ZumasRevenge.exe` SHA-256：
  `db85b891ba662a463d23251a376081386b47437c29248b21e2c10335a6f3eb50`
- `main.pak` SHA-256：
  `02c7038e531c6b9d57ba114fd0e72a475ea51c7c62b1054678af38efd24a2e7d`
- 外层 EXE 中 byte offset 1,925,538 的签名内嵌运行载荷
  `popcapgame1.exe`，由 certificate directory end 推导为 6,657,328
  bytes，SHA-256：
  `2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20`

该内嵌身份由 `tools/embedded_pe_payload.py` 只读复算；正式 DXGI schema v2
同时绑定外层 EXE 哈希、内嵌 offset/extent/hash 与实时目标文件大小。一次运行
期清理窗口中的直接文件读取也得到同一 SHA-256，证明内嵌载荷身份与实际临时
运行文件一致。

## 证据等级与验收原则

本文使用三种证据等级：

- **C（Confirmed）**：由本机 Steam PC 版资源、PC 可执行文件静态证据，或
  可重复的 PC 运行测量确认。
- **S（Strong）**：由 WP7/XNA 可读移植源码强支持，并且其数据结构与 PC 资源
  能对应；仍需 PC 录像或逐帧测量校准。
- **U（Unknown）**：证据冲突、尚未测量，或模拟器尚未实现。

验收遵守以下规则：

1. 离散事件必须逐 tick 一致，包括命中、开始合并、插入、开始爆炸、删除球、
   吸回接触、停止出球和胜负触发；平均值接近不能算通过。
2. 所有影响临界 tick 的计算使用与原版相同的 `float32` 逐次运算和比较顺序，
   不得用公式一次算完，也不得悄然改成 `float64`。
3. 训练观察只能包含玩家在原版画面中可获得的信息。调试用真值状态必须与
   agent 观察隔离。
4. Steam 安装目录和存档只读使用；项目不复制或分发原版美术、音频和关卡资源。
5. 简化环境 `ZumaSimple-v0` 仅是算法与接口参考，不构成一致性验收依据。
6. 每条回放必须声明存档／教学状态。当前只支持
   `profile_mode="tutorials_completed"`；新档的教学脚本不得静默退化成普通
   开局。

## 已由 Steam 原版数据／PC 二进制确认（C）

### 固定时钟

- 游戏逻辑时钟为 **100 Hz**，每个逻辑 tick 为 **10 ms**。所有速度、计时器、
  冷却和动画都必须先以 tick 表示，不能依赖渲染帧率或墙钟时间。
- 例如 XML 中 `GauntletSessionLength=18000` 对应 180 秒，这与 100 Hz 口径
  一致。

### 原版 DMO 录制／回放入口

- Revenge 零售 EXE 中直接存在 DMO magic `0x42BEEF78`、v2 写入路径、
  `-record`、`-play`、`-demofile`、marker 和 PNG screenshot 字符串；同框架
  可读源码给出了最低有效位优先的完整文件结构。PNG 字符串只能证明通用导出
  代码存在，不能证明它绑定了热键。该结论是“框架源码推导并经 Revenge
  二进制交叉验证”，不是 Revenge 官方源码证明。
- DMO header 保存产品版本、framework RNG seed 和总 update 数；命令流保存
  framework update delta 及同 update 内的输入顺序。当前只读解析器覆盖 v1/v2
  和已知命令 0–23、31，对未知命令、截断和非法尾部 fail closed。
- DMO update 表示输入分发边界，不能未经 PC 可见事件校准就等同于 simulator
  native tick。重放中的 `P` 暂停和 `N` 单步 update 可用于建立内存相位证据；
  锁定零售载荷的 `F11` 实际通往受调试标志保护的 `_dump\\imagelist.html`
  图像资源导出器，而不是 `_screenshots\\%d.png` 通用画面导出函数，因此不能
  用作临界状态截图或 PC Golden 证据。
- 零售框架的 `mUpdateCount` 在 `DoUpdateFrames` 内部先增加，随后仍可能继续
  执行本 update 的逻辑与随机调用；因此“计数已到目标”不是安全取样屏障。
  内存轨迹 v2 必须在 `N` 单步后继续等待 `mFastForwardStep=false`，并核对
  `mFastForwardToUpdateNum` 恰好等于目标 update。旧 v1 轨迹只保留为诊断，
  严格加载器不会把它当作正式证据。

### 关卡与 CURV/DAT 曲线

- PC 原版关卡清单来自 `main.pak` 中的 `levels/levels.xml`；曲线来自原版
  `levels/**/*.dat`。当前解析器能够读取全部 **79 个关卡、89 条被引用曲线**。
- CURV 版本 12–15 的字段顺序、标志位和路径点记录已经按原版数据验证。需要
  保留的字段至少包括：
  `start_distance_percent`、`num_balls`、`ball_repeat_chance`、
  `max_single`、`colors`、`speed`、`slow_distance`、
  `acceleration_rate`、`max_speed`、`zuma_score`、
  `skull_rotation_degrees`、`zuma_back_distance`、
  `zuma_slow_duration`、`slow_factor`、`max_clump_size`、
  各类 powerup 记录和 `powerup_chance`。
- CURV 中的相对坐标增量必须以原版游戏字段的 `float32` 精度逐条乘、加并
  写回；“先用 `float64` 累加完整条曲线、最后再转成 `float32`”会在长曲线上
  产生可测漂移，当前解析器已用 4096 次小增量回归锁定这一点。
- 路径点的 tunnel、absolute-anchor 和 priority 信息是游戏语义，不是绘图附属
  数据，模拟时不得丢弃。
- 球的位置标量必须使用 **路径采样点下标形式的 waypoint**；几何累计弧长只可
  用于审计和可视化，不能替代原版 waypoint。也就是说，`waypoint=120.5`
  表示第 120 与 121 个采样点之间，而不是沿曲线走了 120.5 像素。
- 曲线的 `draw_tunnels`、`destroy_all`、`draw_pit`、`die_at_end` 等终局／绘制
  标志必须逐关卡保留，不能用一个全局胜负规则覆盖。

### PC 原生规则与关键常量

- PC 原生可执行文件确认正常弹丸速度为 **8.0 像素/tick**，而 WP7/XNA 可读
  源码返回 **9.6 像素/tick**。PC 环境必须采用 8.0；不得因为移植源码更易读
  而采用 9.6。Accuracy 为 19，慢蛙状态乘 0.25，特定 zone 3 cannon 为 28。
- 正常发射的弹丸在 firing 内部阶段已经更新，并在释放 tick 进入自由弹丸更新；
  从 fire point 计，该释放 tick 累计前进 **24 像素**。因此不能把“第 6 tick
  释放”实现成从静止 muzzle 位置开始、等到下一 tick 才移动。
- 合并过程必须以 `float32` 逐 tick 累加。PC 基准中关键插入临界落在命中后的
  **第 21 tick**：每个游戏 tick 更新两次、每次加 `float32(0.025)`，40 次
  累加只有约 `0.99999958`，第 41 次才夹到 1。实现必须复现这个临界，不能用
  连续时间近似或直接写成数学上的 20 tick。
- PC 原生代码确认正常 D3D 分支半径为 **18**、软件分支为 **17**，圆碰撞采用
  严格 `<`。当前 Steam 正常 D3D 环境应使用 18；仍需运行时探针确认用户实际
  渲染分支没有被切换。
- PC 原生代码确认 firing 每 tick 加 `float32(0.15)`，第 6 tick 释放；reload
  每 tick 加 `0.07` 且严格大于 1 才完成，共 15 tick。正常最短 click-to-click
  约 21 tick。
- 本机 Level 1 无操作运行时探针在 3840×2160、164 Hz 输出上取得 42 个连续
  present、零漏帧，其中只有 26 个不同像素状态；连续相同像素帧 run histogram
  为 `{1: 10, 2: 16}`。present 间隔平均 **6.058 ms**，像素变化间隔平均
  **9.695 ms**。这是 0.25 秒短窗的场景证据，不把 9.695 ms 当成新的全局常量；
  但它已经确认显示 present 与约 10 ms 的游戏更新不可混为一个 tick，验收器
  必须保留重复 present 并另行映射逻辑 tick。
- PC 原生计分按每一波消除独立计算：
  `10 × 本波球数 + 100 × combo_count + gap_bonus + streak_bonus`。例如直接
  消 3 颗、第一次连锁再消 3 颗，总分是 `30 + (30 + 100) = 160`，不能把两波
  合并后只算一次 combo。chain bonus 读取本次成功前的旧计数，因此第 **6**
  个连续有消除的射击首次加 100，第 7 次加 110；“第 5 次”口径不适用于 PC
  实际执行代码。
- gap bonus 使用两侧球心 waypoint 差 `D`：
  `adjusted=max(0,D-64)`，再计算
  `base × (300-adjusted) // 300`（普通模式 `base=500`，Endless 为 250），
  向下取整到 10 的倍数且最少 10；一次消除跨越多个不同 gap 时按 gap 数相乘。
- Zuma 分数目标与真正的 `zuma_reached` 是两个状态。UI 条当前值每 tick 最多
  增加 1，满宽为 330；达到目标后最多仍可延迟约 330 tick，条满才清空 pending
  并停止出球。初始 pending 队列为 10；`num_balls=0` 表示 Zuma 触发前无限补球，
  不是“本关没有球”。
- Jungle1 曲线末端 waypoint 为 3564，危险点为 `3565-600=2965`，提前
  `slow_distance=200` 从 2765 开始减速，危险区速度为 `0.5/4=0.125`。
  `zuma_back_distance=300` 因同 tick 先递减而实际移动 299 tick；随后至少
  保持 20 tick，再以每 tick `+0.005` 恢复推进速度。
- Jungle2 的严格连续点击轨迹确认：按每 3 update 注入左键时，只有 update
  11209／11230／11251 被接受，释放分别落在 11215／11236／11257，接受间隔
  为 21 tick；每颗自由弹丸保留 10 tick。右键换球另有原版输入边沿轨迹。
- Jungle2 的已校准 powerup 子集已有原版完整对象轨迹：生成屏障、颜色唯一性、
  生命周期以及 Proximity Bomb、Reverse、Slow 的触发状态和相关 RNG 消耗均由
  严格验收器覆盖。未实现的其他 powerup 类型不能由这组证据外推。
- `die_at_end` 败局已有受控 PC 诊断：u11191 重置 Board 原生时钟并重建
  3／3 射手球，u11192 锁输入、移除终点球并把剩余 98 球的 `suck_count`
  设为 1；随后每 tick 前进 `previous_suck_count >> 2`，普通
  `Ball.update_count` 冻结，u11367 清空。模拟器从 u11190 原版中态逐 tick
  重放 177 个更新，99 颗球的身份、waypoint、计数和删除批次完全一致，
  最大 waypoint 误差为 0.0。该案例通过内存注入制造终点前置条件，分类仍是
  diagnostic，不替代自然实战 evidence v4 Golden。
- 空链胜利已有一份不含 powerup 的受控 PC 诊断：u9548 的 active、pending
  和 inserting 列表同时归零，Board 运行标志仍为真但该帧左键输入已被拒绝；
  u9549 运行标志才翻转为正式胜利。模拟器以 `win_pending` 复现这一个 tick
  的输入锁，并让 Gym episode 在不可逆的 u9548 边界终止。强制同帧移除
  101 个对象产生的 63 次、每 5 tick 一次的 100 分结算队列属于诊断副作用，
  已被验收器标记为不可迁移到 reward model；该案例同样不替代自然通关 Golden。

## 由 WP7/XNA 可读移植源码强支持、但待 PC 录像校准（S）

以下机制适合作为独立 Python 实现的行为规格，但不能被描述为已经与 PC 版逐帧
一致。

### waypoint 与曲线采样

- waypoint 的整数部分选中路径样本，小数部分在线性相邻样本间插值。
- 若相邻点任一坐标轴跳变超过 5，视为曲线断点并吸附到当前样本，而不是跨断点
  插值。
- 负 waypoint 以“向零截断”的整数索引和第一段曲线做分数外推，例如
  `waypoint=-0.5` 会从第 0 个样本沿第 0→1 段反向外推半个采样间距；超过末端
  的显示坐标才钳制到最后一点。逻辑 waypoint 本身仍可越界，用于入场与进洞
  判断。
- waypoint 先向零截断为整数索引；截断后小于 0 的索引被视为 tunnel（因此
  `-0.5` 的索引仍是 0），正常范围内使用 DAT 的 tunnel 位，超过末端则为
  非 tunnel。切线、法线和 priority 查询必须沿用相同的索引边界规则。

### 球链拓扑与推进

- 球列表从入口侧／尾部到洞口侧／头部排列。正常推进只直接推进入口侧球，
  接触关系再把位移向洞口方向传递。
- 没有接触连接的前方球组保持静止，不能把整条链当成刚体整体平移。
- 匹配只能跨越接触边；几何上靠近但拓扑未连接的同色球不能直接组成三连。
- 强证据指向的单 tick 更新顺序为：补充球／待发球处理、球和合并弹更新、gap
  吸回、正常链推进、显式后退、爆炸与末端删除、powerup 更新。PC 录像必须验证
  同 tick 内多个事件相遇时的最终先后顺序。
- 当前核心按已确认值预生成 10 个 pending balls，并把 `num_balls=0` 解释为
  Zuma 停止出球前不设固定总数。重复色、`max_single` 和 `max_clump_size`
  共同限制颜色序列。Jungle2 的两个独立 pending 生成事件已逐调用确认顺序为：
  背景 MT、repeat `%100`、颜色 `%N`、Ball 视觉帧、背景 MT；颜色结果与完整
  MT 终态均和模拟器一致。其他关卡参数、重试／拒绝分支仍需扩充案例。

### 弹丸、碰撞与 tunnel

- 弹丸是每 tick 移动并做圆形物理碰撞的实体，不是瞬时射线。
- 当前 PC 核心使用已确认的正常弹速 8.0 和 Accuracy 速度 19；移植源码中的
  9.6 只保留为证据冲突记录，不能作为配置默认值。
- 命中插入侧由“弹丸相对球心的向量”和曲线法线的叉积决定。tunnel 状态会按
  命中侧查询 waypoint 邻域，遮挡中的球不能像普通可见球一样被击中。
- 碰撞半径存在 **17 与 18 像素**两种原生路径（软件/D3D 默认值），边界已
  由原生代码确认为严格 `<`。当前环境按正常 D3D 分支使用 18；仍需运行时
  探针或截图确认目标安装没有切到软件分支。
- 原版速度低于球直径，源码表现为离散逐 tick 碰撞；除非 PC 录像证明存在扫掠
  碰撞，否则不能为了“更稳健”自行改用连续碰撞检测。

### merge（插入动画）

- 命中后弹丸不是立即变成链球，而是保存命中球、前后侧、命中位置与合并进度，
  沿曲线寻找空闲 waypoint，并在动画过程中推动相邻球。
- 推动量包含合并进度的平方项；合并过程中还可能把命中归属改到前一颗球。
- 进度、目的位置、推动与碰撞重算的先后顺序会改变最终插入 tick 和链拓扑。
- 验收基准必须复现 PC 的 `float32` 第 21 tick 临界。不能仅让动画时长“约等于
  0.2 秒”，也不能把移植源码 `0.025f` 直接解释成数学上的固定 40 tick。

### 匹配、爆炸、吸回与连锁

- 3 颗或更多同色且通过接触边相连的球形成匹配。
- 移植源码中爆炸帧每 2 tick 增加一次，在第 20 帧删除，因此候选寿命约为
  **40 tick**。PC 的开始计数和删除 tick 仍需逐帧确认。
- 爆炸移除形成 gap 后，如果两侧边界同色，洞口侧球组进入吸回。源码使用
  `suck_count`，后退步长含 `(suck_count >> 3)`；接触后触发链式匹配。
- 接触后的边界球组还会得到约 30 tick 的 rollback，候选速度为
  `max(0.5, combo_count * 1.5)`，并沿接触关系传播。开始 tick、传播方向和
  插入弹打断吸回的规则必须以 PC 录像确认。
- 连续清除奖励已经由 PC 原生函数及其调用顺序裁决：第 6 个连续有消除的射击
  首次加 100，第 7 次加 110。录像仍应用于端到端计分回放，但该常量不再猜测。

### 发射器与颜色

- 发射、弹丸在蛙内的预更新、释放、进入 reload 和 reload 完成是不同事件。
  正常 firing/reload 的 6+15 tick 节拍及释放 tick 的弹丸位移已经由 PC 原生
  代码确认并进入回归；右键输入边沿、Hot Frog、boss slow 与实际 UI 接受点击
  的相位仍需运行校准。Accuracy 只改变弹速，不改变普通 firing/reload 步长。
- 当前球／下一球的颜色只从链中仍存在且未爆炸的颜色集合产生，并带动态权重；
  不能简单使用固定颜色集合上的 IID 均匀采样。
- Jungle1 新档还会在特定链长使用脚本颜色并受首次射击提示控制。当前环境明确
  假定相关提示均已看过；捕获 PC 金样本时必须使用相同 profile 状态。

### Zuma 条、停止出球与终局

- `score >= zuma_score` 只表示已经达到分数目标，不等于同一 tick 立即停止
  出球。当前核心已经按 PC 原生顺序实现 330 宽 UI 条每 tick `+1`、条满后
  `ZumaAchieved`、清空 pending 和停止生成，并用 330 tick 回归锁定相位。
- 触发 Zuma 后还存在按关卡 DAT 配置的整体后退、减速和最终清链阶段，例如
  `zuma_back_distance`、`zuma_slow_duration`、`slow_factor`；这些不是“达到
  分数后立即获胜”的同义词。
- 清空、进入洞口、越过末端、`destroy_all`、`die_at_end`、多曲线是否仍有存活
  球等条件必须组合判断。简化环境的“空链立即胜利”不能直接移植。

### powerups 与计分

- XML 默认值强支持：初始 powerup 延迟约 1500 tick、全局生成间隔约 700 tick、
  单类冷却约 1000 tick；具体关卡还受 DAT 频率、最大数量和总 chance 影响。
- powerup 颜色唯一性、寿命、命中触发、同 tick 爆炸传播和 Zuma 后处理都必须
  保留。不能把 powerup 简化成一个无状态奖励。
- 基础球分、combo、chain、gap、果实、额外生命等计分会影响 Zuma 条和策略，
  因此计分并非“只影响 UI”。基础球分、combo、chain 和 gap 公式已由 PC
  原生代码裁决；果实、额外生命、powerup 相关分数以及完整端到端累加顺序仍需
  PC 重放对照。

### RNG

- 移植源码显示颜色生成不是简单 IID：重复概率、clump 限制、single 限制、
  当前链颜色集合和近期颜色间隔都会参与。
- PC 全局流已确认为 PopCap MT19937，采用零售版正 31 位输出；DMO framework
  seed 可恢复完整 624-word 状态与 cursor。发射器另有 Board 内 QRand，并使用
  主线程 MSVC CRT rand；三段独立、不做 shooter 状态同步的原版轨迹已通过。
- 因为 DMO 头会恢复 gameplay MTRand，同一 DMO 即使在多个进程中得到不同的
  natural CRT startup seed，也不会形成独立的 gameplay 起始分布样本。此前
  “同 DMO、32 个自然进程”的 distribution-v2 设计已在正式采样前作废。
  distribution-v3 改为代码固定、互不重叠的 PC32/simulator256 gameplay seed；
  每个 PC DMO 只能改变头部偏移 8–11，并由逐字节 provenance 和零 RNG 写入的
  strict replay 共同约束。
- 排除于两组正式 seed 之外的原版机制探针已一次通过：在 update 3429 得到
  `current_color_id=1`、`next_color_id=1`，且全局 MTRand 状态与旧基准 DMO
  明确不同；strict broker 的 RNG 内存写入和外部输入均为 0。这只证明 seed
  传输机制有效，不计入正式 32 个样本，也不单独打开 distribution lane。
- 相位安全的 Jungle2 501-tick 轨迹精确连接 500 个相邻完整 MT 状态，共重建
  1,025 次调用。10-tick 动态 INT3 轨迹的 22 次调用与每个相邻状态的调用数、
  调用前状态哈希、输出和 index 推进逐项一致；两次 pending 生成又验证了
  5-call 语义窗口与颜色结果。
- 尚未完成的是所有额外视觉／特效调用者的语义分类、颜色候选被限制拒绝时的
  消耗、未支持的 powerup、教程／多曲线分支和跨关卡种子矩阵。Gate 因这些
  剩余分支仍保持关闭。

## 实现与校准阻断清单（U）

`[x]` 只表示当前独立实现和合成回归已经覆盖该项，不表示整套环境已通过 PC
录像验收。以下任一 `[ ]` 未通过，都保持 Fidelity Gate 为关闭状态：

- [x] 高保真核心以 100 Hz 固定 tick 运行，并通过与渲染帧率无关的确定性测试。
- [x] Gym 保真默认每个 action 推进 1 个原生 tick；更大的 `frame_skip` 被明确
      视为诊断控制抽象，避免把只能落在 5-tick 边界的发射相位冒充原版输入。
- [x] CURV 坐标解码以及球 waypoint、弹丸、枪状态、merge、链速与回退等关键
      动态字段均按逐运算 `float32` 写回；4096 tick 小增量和临界相切碰撞已有
      位级敏感回归。PC 的 `sqrt`／三角函数及 x87／SSE 最末位仍由金样本裁决。
- [x] 球运动使用原版 waypoint 索引、向零截断、分数外推、断点和 tunnel
      越界语义，不再使用归一化弧长代替；priority 与 absolute-anchor 也已由
      解析器保留并具备查询回归。
- [ ] priority、absolute-anchor、`draw_tunnels`、`destroy_all`、`draw_pit`
      和 `die_at_end` 对运行时碰撞、绘制与终局的影响已逐关卡接入并校准。
- [x] 球链基础核心采用接触拓扑、分段推进和 gap 静止语义，并已有断链／重接触
      的逐 tick 回归；复杂多 gap 回放仍列在后续金样本中。
- [x] PC 正常弹速 8.0、Accuracy 19 与移植版 9.6 的冲突已由 PC 原生代码裁决，
      并已进入回归测试。
- [x] D3D 18／软件 17、严格 `<`、离散逐 tick 碰撞、插入侧和 tunnel 邻域
      遮挡已进入合成回归。
- [ ] 已用目标 PC 运行探针确认实际渲染分支，并以相切、前后插入和 tunnel
      金样本验证完整碰撞路径。
- [x] merge 使用逐次 `float32` 累加，并稳定复现命中后第 21 tick 插入临界。
- [ ] merge 的平方推动曲线、改挂前一命中球、两侧最终拓扑和同 tick 事件顺序
      已通过 PC 金样本。
- [x] 接触匹配、约 40 tick 爆炸寿命、同色 gap 吸回、接触连锁和 rollback
      已有合成逐 tick 回归；吸回时 attached merge 重定位与异色新弹打断吸回
      也已按可读源码覆盖；全局 Zuma 回退 300 配置对应实际移动 299 tick。
- [ ] 爆炸开始／删除精确 tick、同 tick 匹配顺序、复杂吸回、rollback 传播与
      被新弹打断的行为已通过 PC 录像回放。
- [x] 普通 firing 第 6 tick 释放、15 tick reload 和约 21 tick 最短节拍已按
      原生 `float32` 顺序实现并回归。
- [x] Jungle2 右键换球输入边沿和连续点击接受／释放相位已有严格 PC 内存轨迹；
      连续点击验收锁定 21 tick 接受间隔、6 tick 释放和 10 tick 自由弹丸寿命。
- [ ] Hot Frog 和 boss slow 的输入与发射节拍已用 PC 运行证据确认。
- [x] chain bonus 第 5/6 次冲突已由 PC 原生代码裁决为第 6 次，环境按每一波
      独立计算基础分、combo 与 streak。
- [x] 单曲线 gap 穿越记录、去重、插入归属及基础 bonus 公式已有合成回归。
- [ ] gap 的完整 PC 逐波计分矩阵，以及果实、额外生命和 powerup 计分已接入
      并通过金样本验证。
- [x] Zuma 条 330 tick 增长、停止出球延迟、10 个 pending 的清理和全局回退
      299 tick 已有合成相位回归。
- [ ] Zuma 后减速、速度恢复、入口回收以及所有单曲线终局条件已逐 tick 对齐
      PC 金样本。
- [x] `die_at_end` 的一帧入口过渡、时钟／射手重置、输入锁、终点球删除和
      全链吸入已从原版中态逐 tick 重放到清空；训练接口在败局不可逆时终止，
      不把后续 175 tick 强制动画计入 agent 决策。
- [x] 受控空链诊断已锁定 u9548 的 `win_pending` 输入锁与 u9549 正式胜利；
      训练接口在不可逆清空时终止，注入造成的延迟结算分数不进入奖励标定。
- [x] Jungle2 已校准的 Proximity Bomb／Reverse／Slow 子集，其生成、颜色、
      生命周期、触发和 RNG 消耗已实现并通过严格原版轨迹。
- [ ] 其余 powerup 类型的命中、叠加、清除和计分语义已实现并校准。
- [x] 高保真核心的 PopCap MT19937、MSVC CRT rand 与 shooter QRand 在同
      seed／同操作下可确定性重放，并有完整状态导入、签名和逐调用回归。
- [x] v3 回放指纹同时绑定静态曲线／tunnel、枪位、物理配置、profile、完整
      动态状态与实际执行 tick 数；错误环境工厂和篡改 compact tick 数均有
      负向回归。
- [x] DMO v1/v2 只读解析器已覆盖 LSB-first 位序、4-bit timing、short
      mouse 命令、全部已知长命令和尾部 padding；敏感同步载荷只输出长度与
      SHA-256，并有全部命令类别的合成二进制回归。
- [x] PC Golden Manifest v4、canonical NDJSON Trace v2、坐标／时钟契约、
      artifact 内容寻址和 partial-observation 状态已实现；v4 的
      capture-contract／evidence-set 两层指纹会绑定场景、最终 artifact 集、
      DMO 相位、视频、坐标、coverage、save/replay 契约与比较阈值。v3 只作
      旧案例兼容读取。未知字段、重复键、NaN／Inf、路径逃逸和 hash 篡改均被
      拒绝。
- [x] Golden 只读验收器已串联 manifest、artifact、trace、tick-map、至少
      9 点控制点标定、DMO header／身份、显式
      `dmo_sequence`／`dmo_update` 输入绑定、coverage、完整视频解码和可选
      原版场景解析；必需测量只能来自受支持的直接 PC 证据源，未绑定的非空
      输入为 `INCOMPARABLE`，不会静默通过。
- [x] artifact 内容读取前先做纯声明资源预算；超大／超多案例直接
      `INCOMPARABLE`，不会先完整哈希恶意超大视频或 sidecar。
- [x] 视频语义验收会完整解码唯一视频流，核对流与逐帧 time base、PTS、
      codec、pixel format、CFR、丢帧／重复帧和规范 RGB24 像素哈希，并把
      trace/tick-map 的 frame index→PTS 绑定回实际解码序列；解码器缺失为
      `INCOMPARABLE`，内容与 Manifest 不符为 `FAIL`。
- [x] 首个原版开火案例的 `pc_memory_probe` provenance 已形成机器可复算的
      measurement 交叉验证：验收器从 24 个 Board／Shooter／Curve／Ball／
      Bullet 原始内存 artifact 重建语义，核对精确冻结 update，并把 update
      7738／7750 的原生 BMP 绑定到主视频 native tick 45／57。除已声明的
      1 行窗口边缘预算外，游戏 viewport 像素精确一致；伪造 JSON、篡改字节、
      未声明依赖和 viewport 内像素差异均有负向回归。该勾选只覆盖这个案例，
      不代表所有未来 `video_tracker`／`pc_memory_probe` 声明自动可信。
- [x] Manifest v4 已包含原始状态快照、hash-chain journal 和 Windows 进程
      时序的 `save_transaction` 契约；验收器会独立核对 pre、每次 run start、
      两次 run end、post 与 restored 状态根。首个真实案例已通过最终
      users／注册表恢复核验。
- [x] Manifest v4 已包含同一 DMO 两次独立进程回放的
      `replay_determinism` 契约；验收器会完整解码两段 lossless 视频，经各自
      tick-map 对齐后逐 native tick 比较全 viewport RGB24，并核对 trace、
      输入、进程正常退出和 end-state。首个真实 78 tick 案例已通过，因而
      verifier 的 `PASS`／退出码 0 已可达；这只是证据协议通过，不等于
      Fidelity Gate 通过。
- [x] Golden 差分器的公开入口会先验收 Manifest，再按 native tick 比较离散
      状态、有序球心、waypoint 与事件顺序；未知必需通道、不可见必需值和模拟器
      轨迹断 tick 均 fail closed，并只对稳定拓扑的完整 100 tick 窗口计算净
      漂移。弹丸尚无稳定跨 tracker 身份，声明为必需通道会
      `INCOMPARABLE`。当前只有合成回归，不能替代真实 PC 案例。
- [ ] 已在目标 PC 上完成至少一次短 DMO record→normal exit→parse→play，
      并证明 Steam 参数透传、DMO 可确定重放及 update/tick 相位。
      首个 pilot 的 update 145 至 12,551 失焦段会令播放自锁，只保留为负向
      诊断。替代 DMO `clean-level1-001` 已完成 record→normal exit→parse，
      不含失焦命令，且两次 `-play` 均创建窗口并走到进程退出。后续原版窗口化
      DMO 又完成两次 lossless 实战采集及在线命令序列校验；开火可见事件、
      内存 update 和视频 native tick 现已机器交叉绑定。剩余未满足部分是把
      原版 native tick 与 `RevengeSimulator.tick_count` 的同一力学状态逐 tick
      比较，所以本项仍不勾选。
- [ ] PC RNG 全部分支已复现或证明等价。全局 MT、CRT、QRand、三段 shooter
      轨迹、501-tick 状态轨迹和两个 pending 生成窗口已经审计通过；仍需覆盖
      限制拒绝／重抽、未支持的 powerup、教程、多曲线及其失败分支消耗。
- [x] 环境与版本化回放显式记录 `tutorials_completed` profile 契约，其他
      profile 默认拒绝而非静默替代。
- [ ] 新档首次射击／换球／果实等教学脚本、脚本颜色、暂停推进和输入门控已
      实现并通过独立 PC 金样本；在此之前只能用“提示均已看过”的存档采集。
- [x] 普通固定蛙多曲线关卡已实现共享 Board／射手／分数／时钟／自由弹丸，
      以及每曲线独立球链、pending、merge、gap、移动状态和 powerup 计时；
      更新按原版 CurveManager 数组顺序执行，观察与渲染槽保留曲线身份，不会
      把多曲线拼成一条路径。
- [ ] 已用原版双曲线逐 tick Oracle 裁决更新顺序、自由弹丸跨曲线碰撞归属、
      总分目标、RNG 消耗、Zuma 传播和跨曲线胜负。当前 `score_target` 的跨
      曲线求和与 Board-wide 败局吸入仍是有静态证据支持但待运行裁决的实现。
- [ ] Iron Frog 的会话、计时、计分、生命与终局状态机已实现；相关关卡当前
      与 Boss、移动蛙和多个蛙位置关卡一样默认拒绝构造。
- [ ] boss 关卡的 boss 状态机、攻击／防御、目标选择、关卡脚本和终局已实现；
      boss 关卡不得用普通 Adventure 规则冒充。
- [x] 默认 actor view 的无隐藏状态边界已有可重跑审计：tunnel 球身份／颜色、
      pending 颜色、移动计时器和 powerup manager cooldown 的联合扰动不会
      改变 actor 输出，而同一扰动会改变 `privileged_debug` 输出；所有 72 个
      特征均有可见性类别，可见 powerup 类型也已进入 one-hot。
- [ ] 已用 PC 视频和短帧历史验证 actor view 的视觉可恢复性，特别是精确
      waypoint、接触拓扑、弹丸速度、merge 进度、枪动画进度和 Zuma 状态；
      无隐藏状态泄漏不等于视觉编码器一定能稳定预测这些字段。
- [ ] 至少建立一组固定 PC 录像“金样本”，覆盖普通命中、前后插入、tunnel、
      三连、四连、gap shot、吸回连锁、Zuma 延迟和进洞失败。
      进洞失败的严格内存诊断及模拟器零误差差分已经完成，但因前置状态由受控
      内存注入制造，尚未冒充这里要求的自然实战录像 Golden。

在以上清单全部完成前，允许进行的工作只有：

- 解析原版数据、编写高保真核心和单元／性质测试；
- 运行随机策略、脚本策略和极短训练来验证 API、吞吐量、显存和 checkpoint；
- 训练明确标记为“不可迁移基线”的诊断模型。

训练入口会把默认请求限制为 10,000 timesteps，并按 PPO 完整 rollout 后的实际
采样量实施 100,000 timesteps 硬上限；普通 Gate 确认参数不能绕过该上限。

禁止把简化环境的百万／千万步训练结果当作未来视觉 agent 的预训练成果，也
禁止基于尚未裁决的常量长时间训练。

## PC 录像校准协议

完整安全事务、目录结构与验收命令见
[PC_GOLDEN.md](PC_GOLDEN.md)。每个金样本应同时保存：关卡名、游戏模式、
原版 DMO、分辨率、原始视频、逐 tick 事件表、可见球心轨迹、分数变化、
坐标／时钟标定和环境版本。

输入主证据使用原版 `-record/-play` DMO；临界 update 的内存采样可以在回放中
使用 `P + N`，但不能再使用已被静态控制流和实机负探测共同否定的 `F11`
截图路径。内存采样不能只等 `mUpdateCount`，还必须通过
`mFastForwardStep=false` 的 post-update 屏障。外部视频以 120 fps 或更高采集，但仍必须检测实际
独立帧、重复帧和丢帧，并映射回 10 ms 逻辑 tick。标称帧率本身不构成验收
证据。

建议按以下顺序校准：

1. 空旷直线段测量普通弹丸和 Accuracy 弹丸连续 10 tick 的位移，复核原生代码
   已确认的 8.0/19，并校验画面坐标映射。
2. 固定单球命中，分别从法线两侧射击，记录碰撞、merge 各临界和插入 tick。
3. 做恰好相切与相差 1 像素的射击，裁决半径 17/18 和严格／非严格比较。
4. 连续点击并插入右键换球，测量 firing、reload 和输入边沿的最短节拍。
5. 构造 3 连爆炸与同色 gap，记录删除、吸回、接触和二次爆炸 tick。
6. 连续制造清除并逐次记分，复核第 6 次首次 chain bonus 及每波 combo 计分。
7. 在 Zuma 条接近满值时制造不同分值爆炸，记录达到目标、条满、停止出球、
   后退、减速和真正通关的独立 tick。

离散事件期望为 **0 tick 偏差**。球心位置的初始验收容差可设为 0.5 像素，
waypoint 容差可设为 0.05；任何误差若随时间持续累积，即使暂时小于容差也判定
失败。每次修改物理、计时、RNG 或事件顺序后，必须自动重跑全部金样本。

## 开门条件

任何训练范围只有在对应的固定 Gate policy 全部满足、内容寻址的金样本矩阵
通过，并且高保真环境在固定 seed 下可确定性重放时，才能变为 `OPEN`。首个
Jungle2 policy 的通过不代表双曲线或整款游戏通过；完整原版范围仍要求上面的
阻断清单全部勾选。5090 只能提高正确环境产生数据的速度，不能降低此门槛。
