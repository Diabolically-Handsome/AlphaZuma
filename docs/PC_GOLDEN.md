# PC 金样本采集与逐 tick 校准协议

本协议用于从合法安装的 Steam PC 版《Zuma's Revenge》建立不可变、可复核的
运行证据。它不是普通“录屏”：每个案例必须同时绑定原版输入流、随机种子、
原始画面、100 Hz 时钟映射、坐标标定、环境指纹和逐 tick 标注。

当前状态：Manifest v4、Trace v2、DMO 解码、逐 tick PTS、控制点标定、
完整视频语义解码、存档恢复事务、双回放确定性、原始内存／画面测量溯源和
只读验收均已建立。首个原版 Jungle2 开火案例已经得到 `PASS`／退出码 0。
该案例只覆盖 78 个 native tick 且弹丸尚未命中；Fidelity Gate 因完整力学
校准矩阵仍未完成而继续关闭。

## 为什么以 DMO 为主、视频为辅

零售版 EXE 包含 PopCap SexyAppFramework 的 `-record`、`-play` 和
`-demofile` 路径。DMO v2 文件保存：

- framework RNG seed；
- framework update 计数；
- 同一 update 内保持顺序的鼠标、按键和滚轮命令；
- 产品版本和 marker；
- 回放所需的同步命令。

项目的 `zuma_rl.popcap_dmo` 按最低有效位优先的原始位流读取 DMO v1/v2，
并完整跳读已知命令 0–23、31。注册表、文件、网络和同步载荷不会写入报告，只
保留字节数与 SHA-256。marker 名称始终只输出字节数与 SHA-256；
`key_down`／`key_up` 键码和 `key_char` 字符默认脱敏，只有明确使用
`--include-sensitive-input` 才会输出原始键盘值。

DMO 的 `update=k` 表示 framework update 计数已经到达 k，输入将在下一次更新
前分发；计数会在该 update 的其余逻辑结束前先增加，因此不能单凭它宣称获得
了完整 post-update 状态。它也不是未经验证即可直接命名为 simulator tick。显式绑定时，
`native_tick = dmo_update + input_timeline.native_tick_offset`，trace input
payload 必须且只能包含 `dmo_sequence` 和 `dmo_update`，并与 DMO 中的
sequence、update、kind 一一对应。完全未绑定的非空输入为 `INCOMPARABLE`，
部分绑定或错误绑定为 `FAIL`。每个案例仍须用第一发射击等可见事件确定 DMO
update、视频 PTS 和原生 100 Hz tick 的相位。

## 不可变案例目录

推荐每次实验使用新的 ASCII 路径，绝不覆盖旧案例：

```text
case-root/
  manifest.json
  input.dmo
  video/
    r1.mkv
    r2.mkv
  tick-map/
    r1.csv
    r2.csv
  calibration.json
  trace/
    r1.ndjson
    r2.ndjson
  evidence/
    state/
    process/
    memory/
```

`manifest.json` 使用 `zuma-rl.pc-golden-manifest` v4；trace 使用
`zuma-rl.pc-golden-trace` v2。Manifest 中每个 artifact 都必须声明规范化相对
路径、精确字节数和 SHA-256。主 trace 必须从 tick 0 的无输入 baseline
开始，连续覆盖到 `clock.tick_end`，不能跳 tick。v3 仅作为历史兼容格式读取；
它不能声明 v4 的存档／双回放契约，因此正式新案例必须使用 v4。

Trace header 还必须携带 `capture_contract_fingerprint`。该指纹覆盖 case、
scenario、PC 环境指纹、除必须内嵌该指纹的 trace 外的 artifact 身份、DMO
timeline、视频、时钟、坐标、coverage、save/replay 契约和比较阈值；
`evidence_set_fingerprint` 再绑定最终 trace。修改 `level_id`／`hard`、DMO
相位、坐标矩阵、coverage、证据或容差后若不重新产生绑定 trace，验收会
`FAIL`。发布案例时还应另外公布最终 `manifest.json` 的 SHA-256；内容寻址
只能发现不一致，不能替代发布者身份认证。

视频只能证明可见状态，不能凭空补全遮挡或未采集状态。每个 measurement 必须
显式使用：

- `observed`
- `inferred`
- `occluded`
- `not_captured`
- `not_applicable`

必需通道若不完整，验收结果必须是 `INCOMPARABLE`，不能用 0、上一帧或模拟器
值填空。

v3 的 measurement 和 event 名称是闭集；计数会与球心／waypoint 列表长度交叉
核对。必需的 complete measurement 只能使用直接 `observed` 值，且 provenance
必须来自允许的 PC 证据源，例如 `pc_memory_probe`、`video_ocr`、
`video_tracker`、`frame_annotation` 或 `internal_screenshot`。把
`source="simulator"` 写进 PC trace 不构成原版证据。

## 存档安全事务

原版会写入 `C:\ProgramData\Steam\ZumasRevenge\users`，并使用以下注册表键：

```text
HKCU\Software\SteamPopCap\ZumasRevenge
HKCU\Software\PopCap\ZumasRevenge
```

每次正式采集都必须执行完整事务：

1. 正常退出游戏和 Steam，确认 `ZumasRevenge.exe`、临时
   `popcapgame1.exe`、`steam.exe` 均未运行。
2. 新建本次 session 的备份目录，复制整个 `users` 目录并导出两个注册表键。
3. 为每个备份文件记录 SHA-256，并立即进行一次反向校验。
4. 录制和回放均使用同一份已完成教学提示的存档；Manifest 写入
   `profile_mode="tutorials_completed"` 和采集前存档指纹。
5. 实验结束后再次退出游戏和 Steam。保留实验后的状态副本，再恢复采集前
   users／注册表。
6. 恢复后重新计算全部 SHA-256；任何不一致都视为事务失败，停止后续采集。

本地 `pc_captures/` 已被 `.gitignore` 排除。该目录可能包含个人存档、注册表
状态和受版权保护的原版画面，禁止提交或分发。

## 录制与确定性重放

首次只做很短的隔离试录。使用唯一的绝对路径，不使用会轮换旧文件的
`-recnum`：

```text
-record -demofile=D:\ZumaGolden\run001\input.dmo
```

DMO 只在正常退出时完成写入。退出后先运行：

```bash
zuma-audit-dmo input.dmo --inputs-only --json
```

确认 magic、DMO v2、产品版本、seed、长度和命令流均可读后，再以同一文件
重放：

```text
-play -demofile=D:\ZumaGolden\run001\input.dmo
```

回放路径支持 `P` 暂停和 `N` 推进一个 framework update。对本项目锁定的零售
运行载荷，实机探测与静态控制流验证已经否定“`F11` 保存游戏画面 PNG”：
`VK_F11` 分支受运行时调试标志保护，并直接调用写入 `_dump\\imagelist.html`
的图像资源导出器；它没有调用二进制中另存于 `0x00487DA0`、使用
`_screenshots\\%d.png` 的通用 PNG 导出函数。后者存在不等于存在可用热键；
扫描该载荷的可执行节也未发现指向它的直接 `call/jmp rel32` 或绝对 VA 引用。
因此正式 PC Golden 只能使用完整视口的外部无损采集，不能再把 `F11` 当作
内部截图途径。静态报告由 `tools/derive_pc_internal_screenshot_capability.py`
生成；本机冻结报告为
`D:\\ZumaGolden\\diagnostics\\pc-internal-screenshot-capability-v1.json`
（SHA-256
`a107a17f1b27e3200f8b9e9d75407cbdb4aafa8a0b35f5ba9a832e7871f81b0c`）。
相关 F11 工具仅保留用于复现实机负结果，不能授权 Golden。

外部采集仍应尽可能达到 120 fps 且无丢帧，用于完整时序、菜单状态和音效证据。
不能仅凭“视频标称 120 fps”判定有效，必须检测实际独立帧、重复帧和丢帧。

至少重复回放两次并比较关键截图、事件 tick 与 DMO 命令序列。若同一 DMO 无法
确定性重放，该案例不能成为金样本。

## 屏障安全的逐 tick 内存轨迹

`mUpdateCount == target` 不是安全冻结条件：动态断点已证明计数增加后，同一
update 仍可能继续消费全局 RNG。轨迹 v2 对每个样本使用以下契约：

- 初始样本由跨过 0.1 倍速阈值的窗口消息充当 post-update 消息屏障；
- 后续样本发送一次 `N`，先核对 update 没有越过目标，再等待
  `mFastForwardStep=false`；
- 每个 stepped 样本还必须满足 `mFastForwardToUpdateNum == target` 且
  `mFastForwardToMarker=false`；
- 两次连续稳定读取后才采 Board、Shooter、Curve、QRand 和全局 MT 状态。

严格加载器只接受 `frozen_post_replay_update_barrier` v2；旧的
`frozen_post_framework_update` v1 标签没有足够证明，只能用于诊断。当前
Jungle2 证据包括一条 501-tick／1,025-MT-call 轨迹，以及一条把 10 tick 内
22 次动态 wrapper 命中逐项绑定到完整 MT 状态的独立校验。它们证明采样器和
已覆盖 RNG 分支，不会替代尚未完成的多机制 Golden 矩阵。

## Windows 原始 DXGI 采集底座

`tools/capture_dxgi.py` 用于正式视频编码之前的短时无损采集。它不按请求值
抽帧，而是尝试取得目标窗口所在输出上的每一个新 desktop present；命令行的
`--frame-budget-fps` 只定义内存／磁盘资源预算，不能被解释为实测帧率。
实测平均 present 率会由 DXGI 的 QPC tick 独立写入 metadata。

采集器只接受命令行指定名称所对应的唯一进程，以及它唯一可见、未最小化的
顶层窗口。本机 Steam 版本中，`ZumasRevenge.exe` 是启动器，实际可见窗口属于
`C:\ProgramData\PopCap Games\ZumasRevenge\popcapgame1.exe`，因此实机采集必须
显式传入 `--process popcapgame1.exe`。采集器绑定 PID、进程创建时间、
EXE SHA-256、HWND、客户区、DXGI adapter、输出矩形和依赖版本。正式采集期间
要求同一窗口持续位于前台、客户区不移动、duplicator 不发生恢复。初始
desktop frame 只用于预热边界，不写入结果；`AccumulatedFrames == 0` 的
pointer-only 更新会跳过，任何 `AccumulatedFrames != 1` 的真实 present 都会
立即失败，因此丢帧案例不会生成完成 metadata。

为避免原始磁盘写入干扰采集，最多 2 GiB 的 BGRA 帧先保存在内存，采集和窗口
复核全部成功后才写入。在 3840×2160、164 Hz 的本机输出上，逐帧临时分配／
复制会偶发落后一个 present；正式路径因此在边界前一次性分配并预触页所有
owned BGRA 目标，通过 DXcam 的 pinned `_grab_into` 逐帧填充。初始化耗时不计入
采集区间，且这项优化不改变 `AccumulatedFrames == 1` 的严格要求：

```text
frames.bgra.raw
frames.csv
metadata.json
```

前两个文件先以 `.part` 独占创建，逐帧和整体 SHA-256、offset、字节数、QPC
频率全部交叉核对后，`metadata.json` 最后发布，作为本次 acquisition 的逻辑
commit marker。`status="acquisition_complete"` 只表示这三个原始文件完整，
不表示案例已成为 PC Golden，更不表示 Fidelity Gate 已打开。

Windows 采集环境锁定为 CPython 3.12、DXcam 0.3.0、comtypes 1.4.16 和
NumPy 2.5.1。依赖文件带 wheel SHA-256，安装时必须使用：

```powershell
py -3.12 -m venv D:\ZumaGolden\tools\dxcam-venv
D:\ZumaGolden\tools\dxcam-venv\Scripts\python.exe -m pip install `
  --no-index --find-links D:\ZumaGolden\tools\wheelhouse `
  --require-hashes -r tools\windows-capture-requirements.txt
```

输出目录必须预先创建、为空、不是 symlink／junction，并使用 ASCII 绝对路径。
首次只录 0.25 秒；游戏进入预期画面、关闭通知并保持无遮挡前台后执行：

```powershell
D:\ZumaGolden\tools\dxcam-venv\Scripts\python.exe `
  tools\capture_dxgi.py `
  --output-dir D:\ZumaGolden\session-id\record `
  --duration-seconds 0.25 `
  --frame-budget-fps 240 `
  --process popcapgame1.exe `
  --runtime-source-executable `
    "D:\SteamLibrary\steamapps\common\Zuma's Revenge\ZumasRevenge.exe"
```

采集器使用 Desktop Duplication，目标窗口上方的通知或 overlay 仍可能进入原始
像素。因此正式运行必须使用专用、无遮挡的游戏显示状态并关闭通知；NVIDIA
统计浮窗也必须在采集前关闭，采集后还要人工检查首帧、中间帧、末帧和完整
视频。原始 acquisition 之后仍需无损编码、完整解码、tick-map、trace、
双进程重放和存档事务，不能绕过 v4 packager／verifier 把原始文件冒充正式
视频 artifact。

本机实测在关闭 `Full Screen`、保持 `Hi-Res (1920x1200)` 关闭时，客户区恰为
800×600，与逻辑画布一一对应；该模式不是从 4K 下采样。当前 Windows 175% DPI
缩放会让旧版 DPI-unaware 窗口在 Computer Use 中只暴露 459×374 的左上裁片，
但 DXGI／Win32 客户区测量仍为完整 800×600。自动化点击问题解决前可用全屏
完成 UI 导航；不能把 459×374 误记为游戏渲染分辨率。

本机运行中的 `popcapgame1.exe` 会被 Windows 锁定，正常退出后又会被删除，
因此无法依赖“退出后再哈希”。持久 Steam `ZumasRevenge.exe` 内含一个签名
PE；其节表和 PE security/certificate directory 可自行推导出精确
6,657,328-byte 文件边界，恰好等于实测临时运行文件。只读审计命令为：

```powershell
python tools\embedded_pe_payload.py `
  "D:\SteamLibrary\steamapps\common\Zuma's Revenge\ZumasRevenge.exe" `
  --expected-payload-bytes 6657328
```

当前安装得到外层 SHA-256
`db85b891ba662a463d23251a376081386b47437c29248b21e2c10335a6f3eb50`，
内嵌运行载荷 SHA-256
`2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20`。
正式采集使用 schema v2 和 `--runtime-source-executable`；metadata 保留外层
文件哈希、载荷 offset/size/hash、certificate extent，并要求载荷大小与实时
目标文件一致。若实时文件恰好可直接读取，两个哈希还必须相等。

`--defer-locked-executable-hash` 仍只允许该进程、恰好 0.25 秒、非 dry-run
的诊断探针；它把 EXE 哈希标记为
`deferred_locked_runtime_payload_probe_only`，转换器会按设计拒绝这种
metadata。该选项不能用于正式 Golden。

DXGI 的真实 present 间隔也可能不是精确 CFR。派生 FFV1 必须保留相对 QPC
时序并明确标记兼容性，不能把不规则间隔重写成均匀 PTS 后再宣称“原始无丢帧
CFR”。如果 v3 的 CFR 契约无法表达该原始时序，案例应继续
`INCOMPARABLE`，直到 v4 verifier 能把 raw acquisition sidecar 纳入机器
验收，或另有显式、可复算的采样契约。

`tools/encode_dxgi_capture.py` 是后续的只读、fail-closed FFV1 转换器。它要求
输入目录恰好包含完成的 raw／CSV／metadata 三件套，重新核对闭集 schema、
PID／HWND／EXE／GPU 身份、锁定依赖、所有大小、offset、逐帧／整体哈希、
QPC/host 时钟和零丢帧契约；输入目录不会被修改。输出使用新的 `.mkv.part`
编码，随后完整解码每帧并与原始 BGRA SHA-256 比较，全部通过后才无覆盖发布：

```bash
python tools/encode_dxgi_capture.py \
  --input-dir /path/to/raw-acquisition \
  --output /path/to/new-empty-output
```

Matroska/FFmpeg 会把流时基规范为 1 ms，因此转换器用纯整数、半值向上的最近
毫秒映射生成 VFR PTS；不抽帧、不补帧。原始 QPC ticks、映射误差、
metadata/raw/CSV 三个 SHA-256、无遮挡能力限制和
`pc_golden_v3_compatible=false` 会内嵌到 MKV timeline metadata，并在 CLI
报告中重复给出。1 ms PTS 是容器派生时间；原始 QPC sidecar 仍是权威时序。

本机当前会话的安全状态和 Windows-native 恢复入口记录在
[WINDOWS_CAPTURE_RESUME.md](WINDOWS_CAPTURE_RESUME.md)。

## 时钟与坐标标定

逻辑画布固定为 800×600，但采集帧可能被整数缩放、拉伸或加黑边。
`calibration.json` 必须使用
`zuma-rl.pc-coordinate-calibration` v1 canonical JSON，并用至少 9 个按
canonical 顺序排列、raw 与 logical 的 x/y 各至少包含 3 个不同值且分布在
画面上的控制点拟合 `logical_from_raw`。当前 sidecar 验收只接受
`axis_aligned_affine`。验收器会完全忽略文件自报的“已拟合”结论，使用 NumPy
最小二乘从控制点独立重算矩阵、RMS 和最大误差，再与 sidecar、Manifest 和
artifact 身份逐项核对。RMS 上限为 0.25 px，最大误差上限为 0.5 px；只写一张
3×3 矩阵、点集退化或误差被放宽都会失败。

`tick-map.csv` 必须为每个 native tick 给出对应 presentation-order frame
与 PTS。初始要求：

- 100 Hz 逻辑时钟；
- `sample_phase="post_update_presented"`；
- 离散事件 tick 不确定度为 0；
- 原始视频无丢帧、无重复帧；
- DMO update 到 tick 的相位由可见锚点证明。

对于 CFR 视频，`time_base × nominal_fps` 必须能表示精确整数 PTS 步长，
`first_pts`、`last_pts` 和 `frame_count` 必须自洽。不能把 100 ms 的 PTS 间隔
标成 100 fps。

## 验收

完成案例后运行只读验收器：

```bash
zuma-verify-pc-golden \
  case-root/manifest.json \
  --original-root "/path/to/Zuma's Revenge"
```

退出码契约：

- `0`：`PASS`
- `1`：`FAIL`，任何已声明证据的格式、身份、绑定或语义不一致
- `2`：`INCOMPARABLE`，结构有效但必需证据不足

验收器依次运行 `manifest`、`artifact_budget`、`artifacts`、`trace`、`tick_map`、
`calibration`、`dmo`、`input_sequence`、`coverage`、`trace_coverage`、
`video_semantics`、可选的 `measurement_provenance`、`scenario_semantics`、
`save_transaction` 和 `replay_determinism`。`calibration` 会从 sidecar
控制点独立重算。视频阶段使用 PyAV/FFmpeg 打开每个声明的视频流，完整解码
每一帧，核对宽高、codec、pixel format、time base、nominal fps、每帧 PTS、
CFR、丢帧／重复帧，把每个 trace FrameRef 的 frame index→PTS 反向绑定到
实际解码序列，并生成规范 RGB24 解码像素哈希；解码器缺失为
`INCOMPARABLE`，已声明视频与实际内容不一致为 `FAIL`。可用以下命令安装
视频与 BMP 像素绑定依赖：

```bash
python -m pip install -e '.[pc-video]'
```

`artifact_budget` 在读取任何案例 artifact 内容前，仅根据已验证 Manifest
声明执行本机资源预检：最多 4,096 个 artifact；非视频单文件最多 512 MiB、
合计最多 4 GiB；视频 artifact 最多 64 GiB、单边最多 16,384 px、每帧最多
33,554,432 像素、最多 100,000 帧、规范 RGB24 解码量最多 32 GiB。超预算
代表当前 verifier 无法安全处理，返回 `INCOMPARABLE` 且不读取 artifact
内容，不把本机预算限制误报成证据矛盾。

提供 `--original-root` 时，`scenario_semantics` 还会只读核对
EXE、main.pak、levels.xml、关卡、难度、曲线和枪位；没有该证据时场景项为
`INCOMPARABLE`。这个目录是调用方明确选择的信任边界：当前只证明 Manifest
与该目录自洽，不提供零售 build 的官方签名认证。任何已声明证据的格式、
身份、绑定或语义不一致都为 `FAIL`。结构化报告不会复制外部异常正文中的
本机路径，也不会包含 DMO 嵌入载荷或控制点内容；Manifest 自带的 case ID 等
调用方标识符会正常回显。

视频解码本身仍只证明容器完整性、解码结果和时钟绑定，不会凭“能解码”断言
画面一定来自祖玛，也不会信任 trace 中自报的
`video_tracker`／`pc_memory_probe` 字符串。案例声明
`memory_transition_contract` 时，`measurement_provenance` 会从 Manifest
列出的 Board／Shooter／Curve／Ball／Bullet 原始字节重建语义，核对冻结
update、像素报告、状态转移和主视频 PTS；任何伪造摘要、未声明依赖或
gameplay viewport 内像素差异都会 `FAIL`。没有该契约的历史案例不会凭来源
字符串获得这项认证。

Manifest v4 的 `save_transaction` 会从原始状态快照、hash-chain journal 和
Windows 进程时序重算恢复事务；`replay_determinism` 会完整比较两个独立进程
回放的逐 native tick viewport RGB24、trace、输入、终局状态和正常退出。
首个真实案例已经让两项及 `measurement_provenance` 同时通过。v4 设计和该
案例的精确范围见 [PC_EVIDENCE_V4_PLAN.md](PC_EVIDENCE_V4_PLAN.md)。

案例达到 `PASS` 后才允许进入 simulator 差分。差分必须使用 Manifest
中冻结的容差；离散事件一对一匹配，球链按入口到洞口顺序配对，位置误差和
waypoint 误差分别统计，任何持续累积的漂移都判定失败。代码入口为：

```python
from zuma_rl import SimulatorTick, compare_pc_golden_case

result = compare_pc_golden_case(
    "case-root/manifest.json",
    simulator_ticks,
    original_root="/path/to/Zuma's Revenge",
)
```

`simulator_ticks` 必须从 0 连续覆盖 `clock.tick_end`，每项还要满足
`source_tick == tick`，并绑定实际 `RevengeSimulator.tick_count` 和
`sample_phase="post_update_presented"`。已知
`GoldenEvent.sequence` 按严格顺序比较，未知顺序事件才按 multiset 比较。
链长变化或插入／删除等拓扑事件会结束身份分段；漂移定义为稳定身份恰好
100 tick 窗口的净误差增长，不用单 tick 尖峰乘 100。弹丸中心目前没有稳定的
跨 tracker 身份，因此不能作为必需通道。

## 首批案例矩阵

按以下顺序建立最小、可诊断案例，不要直接录完整随机一局：

1. 无射击 rollout 与普通直线弹丸；
2. 前侧／后侧单球命中；
3. tunnel 边界与严格相切；
4. firing、reload、连续左键和右键换球；
5. 三连、四连与爆炸删除；
6. gap 吸回和二次连锁；
7. 每波 combo、chain、gap 计分；
8. 达到 Zuma 分数、UI 条延迟、停止出球、后退和终局。

`die_at_end` 败局已有一份受控内存诊断先行案例：模拟器从 u11190 原版中态
重放至 u11367，177 个更新的球身份、waypoint、吸入计数和删除批次全部一致，
最大 waypoint 误差为 0.0。由于该前置状态通过只改一颗终点球 waypoint 的
受控注入制造，它用于裁决败局状态机，但仍不能替代本矩阵要求的自然实战
evidence v4 Golden。

空链胜利也有一份受控内存诊断：无 powerup 的 u9547 中态在经过字节审计的
`should_remove` 注入后，于 u9548 清空并进入输入锁，u9549 才翻转 Board
运行标志。另一个只增加 u9548 左键边沿的 DMO 与基线在可迁移 gameplay
projection 上零差异且没有生成弹丸；模拟器的两拍状态机也通过。强制同帧删除
101 个对象造成的后续 63 次 100 分队列被报告明确标为诊断污染，不能用于通关
奖励或自然计分校准。验收报告 SHA-256 为
`7e2741fcf69acae7f2be3d6bfe15f97d148e2bcd8081fe9932b77dfb4d8e61e7`。

每个案例只裁决少量机制。某个案例失败时，应能定位到一个时钟、坐标、输入或
规则分支，而不是得到“整局看起来不太像”的模糊结论。
