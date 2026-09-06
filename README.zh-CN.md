# Zuma RL(中文完整版)

> English overview: [README.md](README.md)

面向 Steam PC 版《Zuma's Revenge》的高保真强化学习研究环境。项目当前的主
环境是 `ZumaRevenge-v0`：它只读加载用户本机的原版关卡数据，以原版 100 Hz
逻辑 tick 和 waypoint 坐标推进游戏状态，并提供不依赖渲染的 Gymnasium 接口。

> **Fidelity Gate v4 与 Training Gate v3 当前均为 `OPEN`，已允许启动一次有界的原版迁移状态策略训练。**
>
> Fidelity Gate 已拆为真实原版来源、逐机制动力学、随机分布和 actor interface
> 四条独立证据链。c218 对冻结的 97 项证据完成实时复算：四条 lane 全部
> `PASS`，31 项固定要求无缺口，且没有 rejected evidence。
>
> Training Gate v3 已修正旧版的五倍步数外推和包级源码哈希问题：只授权精确
> 98,304 effective steps，并分别绑定模拟器与策略的真实运行时源码闭包。当前
> v9 已绑定环境合同 v13、fruit actor 模型迁移 v7 和 c218 Fidelity 报告，完整
> WSL 复算为 `OPEN`。这不授权视觉训练或直接部署到原版。

完整证据、已确认常量、未知项和开门标准见
[docs/FIDELITY.md](docs/FIDELITY.md)。原版运行证据的采集、存档回滚、DMO
重放和验收格式见 [docs/PC_GOLDEN.md](docs/PC_GOLDEN.md)。两道正式门的固定
规则见 [docs/FIDELITY_SUITE.md](docs/FIDELITY_SUITE.md) 和
[docs/TRAINING_GATE.md](docs/TRAINING_GATE.md)。

## 两个环境的定位

| 环境 | 用途 | 是否可作为原版迁移训练依据 |
| --- | --- | --- |
| `ZumaRevenge-v0` | 高保真主线；读取原版 CURV/关卡数据并以 100 Hz tick 模拟 | 当前仅可按 Training Gate v3 的固定配方训练 |
| `ZumaSimple-v0` | 早期简化原型，仅用于接口、算法和工具链参考 | 不可以 |

不要把 `ZumaSimple-v0` 的胜率或训练步数理解为原版预训练成果。正式训练必须
使用 `ZumaRevenge-v0`、两个当前 OPEN suite 和 Training Gate 固定的完整配方。

## 当前高保真实现

- 从本机 Steam 安装中只读解析 `main.pak` 的关卡 XML 和 `levels/**/*.dat`
  曲线，不复制或分发原版美术、音频与关卡文件。
- CURV v12–15 解析器；本机目录审计覆盖 226 个曲线文件，原版关卡清单中的
  79 个关卡和 89 条唯一引用曲线可全部加载。
- 原版 waypoint 索引、断点、tunnel、priority、越界查询和入口隐藏段数据；
  CURV 相对坐标按每条记录逐次 `float32` 累加，不在末尾才统一降精度。
- 100 Hz 固定逻辑 tick、接触拓扑、分段球链推进、gap、物理弹丸、tunnel
  命中侧遮挡、插入 merge、匹配、爆炸、吸回与 rollback 的第一版核心。
- 球 waypoint、弹丸位置／速度、枪状态、链速、merge 和临界碰撞等游戏动态
  字段按 `float32` 写回；长时间小步累加和相切边界已有专门回归。
- PC 常量的独立回归，包括正常弹速 8 px/tick、D3D 半径 18、严格 `<` 碰撞、
  第 21 tick 插入、6 tick 发射、15 tick reload、计分和 330 tick Zuma 条。
- 固定 seed 的确定性状态签名、版本化逐 tick JSON 回放与 Gymnasium 五元组
  API。高保真核心已实现零售版全局 MT19937 正 31 位输出、MSVC CRT rand
  状态转移和发射器 QRand；三段不做 shooter 同步的原版轨迹已逐 tick 通过。
- PopCap DMO v1/v2 只读解码器，可恢复 framework seed、update 时序及有序
  鼠标／键盘输入；注册表、文件、网络等嵌入载荷只暴露长度与 SHA-256。marker
  名称始终只输出字节数与 SHA-256；键盘值默认脱敏，只有显式传入
  `--include-sensitive-input` 才会输出原始键码／字符。
- 内容寻址的 PC Golden Manifest v4 与 canonical NDJSON Trace v2，显式区分
  observed、inferred、occluded 和未采集状态。capture-contract 指纹同时绑定
  场景、除需内嵌该指纹的 trace 外的 artifact、DMO 相位、视频、时钟、坐标、
  coverage、存档事务、双回放确定性和比较阈值；evidence-set 指纹再绑定最终
  trace，不能在保留旧 trace 的同时悄悄更换关卡、证据或放宽容差。v3 仅作为
  旧案例兼容格式读取。
- 只读验收器对 artifact、DMO、trace、输入绑定、逐 tick PTS、evidence
  coverage、控制点标定和完整视频解码采用 fail-closed 规则；视频流会逐帧
  核对元数据、PTS、CFR／丢帧，反向绑定 trace/tick-map 的 frame index→PTS，
  并生成规范 RGB24 像素哈希。原版安装目录可选用于复核 EXE、main.pak、
  levels.xml、关卡、曲线和枪位身份。
- Windows 短时原始 DXGI 采集底座会绑定目标 PID／HWND／EXE、显示输出和锁定
  依赖，跳过 pointer-only 更新，拒绝任何真实 present 丢失，并在采集结束后
  才把内存中的 BGRA 帧、逐帧 QPC 和哈希事务性写盘。只读转换器会把它无损
  编为 FFV1、完整解码复核像素，并在 MKV 中保留 QPC timeline 和采集来源。
  Manifest v4 的 `save_transaction` 与 `replay_determinism` 已进入生产器和
  只读验收器：原版窗口化 DMO 的两次独立无损回放、进程时序、采集前后状态根、
  恢复状态根及逐 native tick 全画面 RGB24 均可从原始 artifact 独立复算。
- 原版内存证据验收会从 Board、Shooter、Curve、intrusive list、Ball／Bullet
  原始字节重新解码，不信任采集器写出的语义 JSON；冻结状态同时绑定精确
  framework update。首个开火案例把 update 7738/7750 的探针 BMP 绑定到主
  视频的 native tick 45/57，游戏区域逐像素一致，并证明 94 个链球身份／颜色
  不变、每球前进 1.5、弹丸 ID 104 以约 8 px/tick 离膛且枪膛球正确轮换。
- 内存轨迹 v2 不再把 `mUpdateCount` 刚增加误当作整帧完成；每次 `N` 单步都
  必须等待零售框架的 `mFastForwardStep` 后更新屏障。501 tick 的 Jungle2
  轨迹已在 501/501 个样本上通过该契约并精确重建 1,025 次全局 MT 推进；
  另一个 10 tick 动态断点窗口把 22 次调用逐项绑定到调用前状态哈希、输出和
  调用者。
- 在读取 artifact 内容前先执行纯声明资源预算；超预算案例为
  `INCOMPARABLE`，不会先对恶意超大视频做完整磁盘哈希。
- 默认观察为 actor view：tunnel 内球不会进入 agent 的球槽；内部计时器只有
  显式启用 `privileged_debug` 时才暴露；可见球的道具图标以零售 powerup ID
  one-hot 暴露。Jungle2 的无隐藏状态审计已证明 tunnel 球身份、pending
  颜色、移动计时器和 manager cooldown 不会改变 actor 输出；精确 waypoint、
  merge 进度等字段能否从原版画面／短历史稳定恢复仍需视觉证据审计。
- 普通固定蛙关卡现可在同一个共享 Board 中加载 1–2 条曲线：射手、分数、
  时钟与自由弹丸全局共享，球链、merge、移动状态、gap 和 powerup 计时按
  曲线独立保存，并严格按原版曲线数组顺序更新。多个蛙位置、移动蛙、Boss
  和 Iron Frog 仍默认拒绝，不会静默选取第一项冒充完整关卡。
- 双曲线实现目前属于结构与回归测试通过；在原版双曲线逐 tick 案例裁决更新
  顺序、跨曲线弹丸命中、计分目标、RNG 消耗和胜负条件前，不会被 Fidelity
  Gate 认证。
- Iron Frog 专用关卡同样默认拒绝。当前唯一受支持的存档契约是
  `profile_mode="tutorials_completed"`：所有教学提示已看过；新档 Jungle1 的
  脚本颜色、暂停推进与提示输入状态机尚未实现，不会被默认为普通随机开局。

这些项目表示“已经实现并有单元测试”，不等于“已经由 PC 录像逐 tick 验收”。

## 安装

建议使用 Python 3.11 或 3.12 和独立虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

只做基础环境时不强制安装视频解码器；执行 PC 金样本视频验收时安装：

```bash
python -m pip install -e '.[pc-video]'
```

该可选组同时安装 PyAV 和 Pillow；前者负责完整视频解码，后者负责原版冻结
BMP、内存探针与视频帧的逐像素绑定。缺少所需依赖时，对应阶段会明确返回
`INCOMPARABLE`，不会静默跳过。

环境需要用户合法安装的《Zuma's Revenge》。程序会尝试从 Steam library
配置和常见目录自动定位；也可以显式传入安装目录，或设置
`ZUMA_REVENGE_ROOT`。目录中应包含 `ZumasRevenge.exe`、`main.pak` 和
`levels/`：

```bash
export ZUMA_REVENGE_ROOT="/path/to/SteamLibrary/steamapps/common/Zuma's Revenge"
zuma-audit-original --level Jungle1
```

训练依赖用于算法管线诊断和获授权后的正式阶段；安装它本身不构成 Gate 授权：

```bash
python -m pip install -e '.[train,test]'
```

训练入口在没有明确选择正式或诊断模式时仍会拒绝执行。只有做极短的算法管线
诊断时，才使用 `--acknowledge-fidelity-gate` 进入非迁移模式。默认请求 10,000
timesteps；该模式还会按 PPO 的完整 rollout 计算实际采样量，并将其硬限制在
100,000 timesteps 以内，不能用确认参数绕过：

```bash
zuma-train \
  --acknowledge-fidelity-gate \
  --total-steps 1000 \
  --num-envs 1 \
  --single-process \
  --batch-size 256 \
  --run-dir runs/diagnostic-only
```

生成的 `config.json` 会固定写入 `fidelity_gate: "closed"` 和
`diagnostic_non_transferable: true`，并记录
`profile_mode: "tutorials_completed"`、实际计划采样量和关闭期间的硬上限。
简化原型还需要额外的
`--environment simple --allow-reference-smoke`，以避免误启动。

正式入口不是另一个可绕过上限的确认开关。只有 Fidelity v4 和 Training Gate v3
都实际返回 `OPEN`，且命令与固定环境、模型、seed、PPO 配方和 98,304-step
阶段逐字段相同时，`--transfer-training` 才会继续：

```bash
zuma-train \
  --transfer-training \
  --fidelity-suite /path/to/ZumaGolden/diagnostics/fidelity-suite-c200-candidate-v24.json \
  --fidelity-suite-root /path/to/ZumaGolden \
  --training-suite /path/to/ZumaGolden/training/suites/training-gate-v3-candidate-v9.json \
  --training-suite-root /path/to/ZumaGolden \
  --original-root "/path/to/Zuma's Revenge" \
  --level Jungle2 \
  --aim-bins 180 \
  --mask-rejected-actions \
  --max-ticks 12000 --max-balls 768 \
  --initial-model /path/to/ZumaGolden/training/models/fruit-actor-migrated-v1.zip \
  --reset-optimizer-state --teacher-kl-coef 1000 \
  --total-steps 98304 --num-envs 4 --seed 20260950 \
  --device cuda --rollout-steps 512 --batch-size 512 --ppo-epochs 1 \
  --learning-rate 0.000005 --reward-scale 0.01 --entropy-coef 0 \
  --checkpoint-every 16384 --eval-every 1000000 --eval-episodes 8 \
  --boundary-eval-episodes 32 --eval-num-envs 8 \
  --also-evaluate-stochastic-boundaries \
  --run-dir /new/nonexistent/run-directory
```

当前两个 suite 均为 `OPEN`，上述命令在 `--run-dir` 指向全新且不存在的目录时
可以继续。`config.json` 会写入两个 suite 的 SHA-256、完整报告和 recipe hash。
该阶段仍只授权 Jungle2 状态策略校准，不授权视觉训练或直接部署到原版。

## 审计原版数据

先确认解析器读取的是预期安装和关卡：

```bash
zuma-audit-original --level Jungle1
zuma-audit-original --level Jungle1 --json
zuma-audit-original --root "/path/to/Zuma's Revenge" --level Jungle1
```

当前本机 `Jungle1` 审计基准为：

- 枪位置 `(420, 290)`；
- 3,565 个 waypoint 样本，末端 waypoint 为 3,564；
- 几何审计长度约 3,582.88 px（模拟推进不使用该弧长）；
- 105 个 tunnel 样本、4 种颜色、基础速度 0.5；
- Zuma 目标分数 1,250，回退 300，减速时长 1,100。

## PC DMO 与金样本验收

原版运行校准不再依赖“看起来相似”的普通录像。先审计原版录制产生的 DMO：

```bash
zuma-audit-dmo path/to/input.dmo
zuma-audit-dmo path/to/input.dmo --inputs-only --json
```

DMO 的 framework update 与环境 tick 不能未经测量直接等同；每个案例都要用
第一发射击等可见锚点建立相位，并保留原始视频、tick-map、坐标标定和逐 tick
trace。案例完成后执行：

```bash
zuma-verify-pc-golden \
  path/to/case/manifest.json \
  --original-root "/path/to/Zuma's Revenge"
```

验收结果严格分为 `PASS`、`FAIL` 和 `INCOMPARABLE`，对应退出码 0、1、2。
只要必需通道缺失、视频未被可用解码器完整核验、视频丢帧、时钟仍有不确定度、
标定不能从控制点重算、原版场景身份未解析，或非空输入没有显式绑定到 DMO，
就不能得到 `PASS`。Manifest v4 还要求从原始快照／journal／进程时序重算
`save_transaction`，并从两次独立回放的完整视频、trace 和 tick-map 重算
`replay_determinism`；带内存契约的案例还会执行 `measurement_provenance`。

当前真实案例 `jungle2_dmo_e43c645a18d7_u7693_7770` 的 62 个 artifact 已通过
全部检查，证明这套证据管线可以产生退出码 0。这个 `PASS` 只认证该 78 tick
开火片段及其声明的测量，不认证整个模拟器，也不打开 Fidelity Gate。详细目录
格式和安全采集流程见 [docs/PC_GOLDEN.md](docs/PC_GOLDEN.md)；v4 的设计与
已实现状态见 [docs/PC_EVIDENCE_V4_PLAN.md](docs/PC_EVIDENCE_V4_PLAN.md)。

单案例 `PASS` 之外，正式训练还必须通过不可删减要求的多案例总门槛：

```bash
zuma-verify-fidelity-suite \
  /path/to/ZumaGolden/diagnostics/fidelity-suite-c200-candidate-v24.json \
  --suite-root /path/to/ZumaGolden \
  --original-root "/path/to/Zuma's Revenge"
```

总门槛会重新验收 PC Golden 与单进程 PC source、核对每份报告的 SHA-256，
并把原版来源、模拟器差分、随机分布、actor interface 和仅供诊断的报告分 lane
计数。使用 shooter 同步纠正状态的差分、内存注入制造的终局以及 diagnostic
`PASS` 都不能替代自然原版证据。当前固定策略是
`original-transfer-jungle2-v4`，当前状态为 `OPEN`：冻结的 97 项证据重新验收
通过，四条 lane 均为 `PASS`，31 项固定要求没有缺口。
Gate 从报告正文推导机制，不信任 suite 手写标签。格式和硬编码条件见
[docs/FIDELITY_SUITE.md](docs/FIDELITY_SUITE.md)。

视觉可恢复性可用以下命令逐案例重算；它属于后续视觉阶段，不会借状态环境的
Fidelity v4 `OPEN` 自动获得授权：

```bash
zuma-audit-actor-visual \
  /path/to/ZumaGolden/cases/<case>/manifest.json \
  --evidence-root /path/to/ZumaGolden \
  --original-root "/path/to/Zuma's Revenge"
```

当前 actor 视觉审计对 72 个 actor 特征证明了 28 个；其余 44 个仍缺道具
图标、爆炸态、连续弹道／合并历史、射手与 HUD 等视觉证据，因此工具严格返回
`INCOMPARABLE`，不会用整帧录像存在这一事实冒充逐特征可恢复。
补采 DMO 的内容寻址任务、三个已准备会话和安全执行方式见
[docs/PC_CAPTURE_CAMPAIGN.md](docs/PC_CAPTURE_CAMPAIGN.md)。

原版机制使用独立的机器审计，计划只能选择证据窗口，不能手写机制结论：

```bash
zuma-audit-pc-mechanisms \
  diagnostics/pc-mechanism-plan.json \
  --evidence-root /path/to/ZumaGolden \
  --original-root "/path/to/Zuma's Revenge"
```

它要求无注入的 v2 replay-barrier 逐 tick 轨迹，并把 probe 首帧逐像素绑定到
同一 native-source 的 PC Golden 录像，再自动推导发射、碰撞、精确三／四连
和 rollback。现有旧 v1 轨迹不会被重新包装成认证材料。完整契约见
[docs/PC_MECHANISM_AUDIT.md](docs/PC_MECHANISM_AUDIT.md)。
对应 C2/C4/C5 的第三次 v2 memory follow-up 已写入
[`diagnostics/pc-memory-followup-v1.json`](diagnostics/pc-memory-followup-v1.json)；
它会在双回放完成后解锁，自动只读发现稳定分数，并拒绝任何 feature 标签。

案例达到 `PASS` 后，只能用 `zuma_rl.compare_pc_golden_case()` 从
Manifest 路径进入差分；旧的内存对象入口会 fail closed。`SimulatorTick`
必须携带实际
`RevengeSimulator.tick_count` 和 `post_update_presented` 采样相位。比较器支持
离散标量、有序链球中心、waypoint 和事件顺序；只有稳定拓扑连续 100 tick
窗口才计算净漂移。弹丸尚无跨 PC tracker 的稳定身份契约，因此把
`projectile_centers` 声明为必需通道会得到 `INCOMPARABLE`，不会用不稳定列表
顺序冒充匹配。

## Gymnasium 冒烟

默认 `aim_bins=180`，动作空间为 `MultiDiscrete([3, 180])`：第一个分量是
verb，第二个分量是瞄准 bin。

- verb 0：只瞄准并等待；
- verb 1：请求发射；
- verb 2：请求交换当前球与下一球。

因子化只改变策略分布的参数化，不改变任何游戏输入或物理。它避免把“等待”和
“换球”各复制成 180 个等价类别；旧的 `3 × 180=540` 扁平 Discrete 编码仍可
通过 `RevengeEnvConfig(action_mode="flat")` 或训练参数 `--flat-actions`
显式启用，用于旧回放和受控对照。

保真默认下每个 action 只推进 **1 个原版 tick**，使策略能在每个 10 ms 输入
边界重新瞄准、发射或换球。显式设置 `frame_skip>1` 会降低控制时间分辨率，
只应用于吞吐／算法诊断，不能无条件当作等价原版输入。下面只验证 API 和数据
流，不属于正式训练：

```python
import gymnasium as gym
import zuma_rl  # 注册 ZumaRevenge-v0

env = gym.make("ZumaRevenge-v0", level_id="Jungle1")
observation, info = env.reset(seed=42)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

env.close()
```

需要构造具体动作时，可以使用：

```python
fire_at_bin_90 = env.unwrapped.encode_action(verb=1, aim_bin=90)
```

调试 RGB 渲染是独立绘制的状态可视化，不含原版素材，也不能作为原版视觉输入
的替代品。

## 训练管线验收状态

本机 WSL2／Threadripper 7970X 上，32 个异步 `ZumaRevenge-v0` 环境、
`frame_skip=1` 的随机动作基准约为 3,244 个原生 tick/s；98,304-step PPO
诊断训练段约 60 秒，约 1,600 steps/s。项目环境现已安装
PyTorch `2.12.1+cu130`；RTX 5090 上的矩阵运算和 4 环境／512-step 真实
Stable-Baselines3 PPO CUDA 烟测均通过，RTX 5080 保持隔离。环境步进仍是
CPU 工作，小型 MLP PPO 通常也以 CPU 更高效；5090 的主要收益会出现在后续
视觉编码器/CNN 和更大 batch。

训练入口默认使用因子化动作，并把原生奖励整体乘以 0.01：score、step、
win、loss 同比缩放，不改变 reward-optimal policy，只改善 value 网络数值。
每个 `config.json` 会记录 Python、Gymnasium、NumPy、SB3、PyTorch、
TensorBoard、CUDA 可用性、线程数、动作模式、缩放和熵系数。

当前短基线证明了完整的“多进程采样→PPO 更新→checkpoint/final model→重新
加载→固定 seed 确定性回放”链路，但尚未证明已经学会通关。98,304-step
因子化模型在 10 个未见 seed、每个 200 tick 的小样本中，随机采样策略均分
87、均匀随机均分 76；确定性 argmax 仅 18。另一组统一奖励缩放实验数值更稳定，
但小样本得分没有胜过随机。样本量很小，不能据此宣称策略优越；下一阶段需要
动作 mask、课程场景和正式成组评估，而不是继续盲调 PPO 或修改原版动力学。

## 测试

```bash
python -m pytest -q
```

没有检测到原版安装时，依赖安装数据的审计测试会跳过；合成曲线上的核心、
Gym API、DMO、Golden 格式、验收器和确定性测试仍会运行。修改物理、计时、
RNG 或事件顺序后都应重跑完整测试。真实 PC 金样本建立后也必须进入同一回归
流程。

当前本机验收为 **635 tests passed**；另有一条 Gymnasium 关于未注册
`render_modes` 的既有 warning，不影响测试结果。

`zuma_rl.replay` 提供 `ReplayPlan`、`ReplayAction`、`run_replay` 和
`verify_replay`，可把原生 tick 输入、事件与 SHA-256 状态轨迹写成版本化
JSON；回放文件显式记录 profile mode，避免把新档教学脚本与已完成教学的正常
开局混为同一条轨迹。v3 指纹同时绑定曲线／tunnel 数据、枪位、物理配置和实际
执行 tick 数，不会只因动态球列表暂时相同就接受错误环境。它是未来 PC 金样本
差分的基础设施，本身不表示已经完成 PC 校准。

## 高保真环境吞吐量

只测球链推进开销：

```bash
zuma-benchmark-revenge \
  --level Jungle1 \
  --seconds 15 \
  --num-envs 1 \
  --frame-skip 5 \
  --policy wait
```

同时压测弹丸并使用多进程环境：

```bash
zuma-benchmark-revenge \
  --level Jungle1 \
  --seconds 30 \
  --num-envs 8 \
  --frame-skip 5 \
  --vector-mode async \
  --policy random
```

下面是严格逐运算 `float32` 收紧之前的一次历史短时快照，条件均为
`Jungle1`、`frame_skip=5`、无渲染。它只用于说明“完整局数”为什么不是稳定
吞吐单位，**不能作为当前版本的性能承诺**；收紧后的空载、目标策略网络基准
仍需重新测量。

| 配置 | 决策 transitions/s | 原版 100 Hz ticks/s | 短局 episodes/day 外推 |
| --- | ---: | ---: | ---: |
| 1 env，wait | 990 | 4,945 | 约 57,600 |
| 1 env，random | 439 | 2,191 | 约 57,600 |
| 4 env，async random | 994 | 4,956 | 约 172,700 |
| 8 env，async random | 1,744 | 8,703 | 约 224,600 |

该历史快照中 8 个异步环境相当于约 **1.51 亿次决策 transition/天**。但表中的
`episodes/day` 来自随机或不发射策略很快失败的短局；策略变强、单局变长后，
完整对局数会显著下降。因此规划训练预算应优先使用
`decisions_per_sec` / `native_ticks_per_sec`，不要直接采用“每天 22 万局”。

本轮精度收紧后，单环境核心的初步短测已经显示吞吐下降；测量时开发机另有
高占用 CPU 训练任务，结果不具备可比性，因此没有把它冒充为新的正式数字。
待机器空载后，应使用上面的命令重新建立 1／4／8／16 环境基准。

高保真游戏逻辑目前在 CPU 上推进；RTX 5090 主要加速策略网络的训练和推理，
不会把 Python 游戏物理自动搬到 GPU。最终吞吐量还取决于 CPU、并行环境数、
策略网络、IPC 和单局长度，应在目标训练配置上重新实测。

## 两道 Gate 开门后的边界

1. 只执行 Training Gate v3 锁定的 98,304-step Jungle2 状态策略阶段，不更换
   seed、模型、环境、并行数、PPO 参数或评估协议。
2. 阶段完成后使用新预注册、未消费 holdout 和独立 seed 决定是否放大，不自动
   跳到千万步。
3. 视觉模块、双曲线、更多 Adventure 关卡和原版部署各自建立新 Gate，不沿用
   Jungle2 状态策略的窄范围许可。
4. 每次正式运行都使用全新且不存在的 run 目录，并保留两个 suite、报告和模型
   的内容寻址身份。

“环境先验收、再正式训练”是项目约束。5090 可以更快地产生训练数据，但无法
弥补错误的规则或事件顺序。
