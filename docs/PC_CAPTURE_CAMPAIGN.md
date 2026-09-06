# PC Golden 补采任务

当前任务清单：
[`diagnostics/pc-capture-campaign-v1.json`](../diagnostics/pc-capture-campaign-v1.json)

只读验证：

```bash
zuma-verify-pc-capture-campaign \
  diagnostics/pc-capture-campaign-v1.json \
  --evidence-root /mnt/d/ZumaGolden
```

2026-07-30 实机结果为 `PASS`：

- 1 个已经验收的 C1 案例；
- 3 个 `ready_for_collection` 任务；
- 4 个 `requires_recording` 任务；
- 1 个被明确隔离的 diagnostic-only DMO。

这里的 `PASS` 只证明 DMO hash／seed／长度、原版 runtime、prestate 和
effect-free repaint/capture 窗口完全匹配，不证明候选机制已经出现在画面中。
机制标签必须在采集后由 PC memory／video 与模拟器差分重新推导。

采集后的 PC 机制认证使用 `zuma-audit-pc-mechanisms`。它不读取
`candidate_features` 作为结论；计划中只能指定证据路径和 update 窗口。原版
memory trajectory 必须是 schema v2、`frozen_post_replay_update_barrier`，
首帧必须逐像素绑定到同一 native-source 的 PC Golden 录像。旧 v1 轨迹即使
事件看起来正确，也只能用于导航和诊断。

## 已准备会话

以下目录已经执行完 `prepare`；这一步没有启动祖玛：

| 任务 | 会话目录 | seed | 窗口 |
|---|---|---:|---:|
| C2 | 需要重新录制 | 新 seed | 原 seed 23,775,218 的未修改 DMO 在 bit 221 确定性失步，只保留为诊断 |
| C4 | `D:\ZumaGolden\campaign-c4-two-run-u230-v43` | 36,311,796 | 双回放已完成，wait 12795，5.5 秒，180 fps 资源预算 |
| C5 | `D:\ZumaGolden\campaign-c5-existing-reverse-v5-u230-2s` | 36,311,796 | 双回放已完成，wait 9460，2.0 秒，240 fps 资源预算 |

C2 原始录制器在首个 D3D sync 前写入了两条位级相同的 `registry_read`。
未修改回放在当前宿主与录制时 `ScreenMode=1` 宿主上都确定性停在 bit 221；
删除任一重复命令后的版本只能用于诊断，不能计入 PC Golden。500-tick 长时域
证据必须由一份新录、未修改且可双回放的 DMO 重新取得。

## 执行采集

`collect` 会启动两次可见原版回放、使用 DXGI 设备 0（本机 RTX 5080 的
`DISPLAY1`），并在 `finally` 中恢复祖玛用户状态。只能在用户让出桌面和
RTX 5080 后执行：

```powershell
& 'D:\ZumaGolden\tools\dxcam-venv\Scripts\python.exe' `
  'C:\Users\Laure\Documents\祖玛\tools\run_pc_capture_campaign_task.py' `
  collect `
  --session-root 'D:\ZumaGolden\campaign-c2-clean-long-shot-tail-v1'
```

C4、C5 只需替换 `--session-root`。已准备目录不可覆盖、不可重新 prepare；
失败会原样保留用于诊断，也不得伪装成完成案例。

## 为什么仍需新录 DMO

- seed 5,909,046 的 pilot 在 update 145 失活，播放无法推进到后续激活命令；
  它只能作为 parser/navigation diagnostic，验证器禁止 ready task 使用它。
- C3 需要新干净 DMO，以提供第三个认证 seed，并覆盖 back insertion、
  match4、rollback 和 RNG rejection。
- C6 需要自然 slow／proximity-bomb 物理效果。
- C7 需要自然 Zuma transition／win。
- C8 需要自然 loss；已有 loss 是内存注入诊断，不能认证。

录制新 DMO 时应同时保留完整双回放、逐 tick gameplay-memory、RNG、输入、
score 和视觉帧历史，以避免再次得到“视频有了但无法认证机制”的孤立素材。
双回放结束后还要用同一 DMO、runtime 和 protocol prestate 单独执行一次
无注入 v2 memory trajectory 采集；该第三次回放用于机制审计，不与正在编码
的视频进程混用，以免冻结／单步探针扰动录像时序。

三个第三次回放已经由
[`diagnostics/pc-memory-followup-v1.json`](../diagnostics/pc-memory-followup-v1.json)
内容寻址。只读检查：

```bash
zuma-verify-pc-memory-followup \
  diagnostics/pc-memory-followup-v1.json \
  --evidence-root /mnt/d/ZumaGolden
```

当前计划只保留已完成的 C4 与 C5；两项 memory trajectory 均已采集完成并使用
冻结后只读 score discovery。原 C2 follow-up 已从可执行计划移除，避免将诊断
DMO 误当作待认证证据；新 C2 录制通过双回放后再以新的内容哈希加入。

双回放完成后可先无窗口查看即将执行的精确参数：

```powershell
& 'D:\ZumaGolden\tools\dxcam-venv\Scripts\python.exe' `
  'C:\Users\Laure\Documents\祖玛\tools\run_pc_memory_followup_task.py' `
  show `
  'C:\Users\Laure\Documents\祖玛\diagnostics\pc-memory-followup-v1.json' `
  --evidence-root 'D:\ZumaGolden' `
  --task 'c2-clean-long-shot-tail-memory'
```

确认对应 PC Golden 已完成且桌面仍空闲后，把 `show` 改为 `collect`。C4/C5
替换 task ID。执行器会再跑一次完整计划验证；输出目录已存在、session plan
被改动、双回放未完成或时间窗口越过下一条 effectful command 时都会失败关闭。
