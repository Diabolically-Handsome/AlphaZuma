# Windows PC capture handoff

Active session: `20260728-203153`.

## Current safety and runtime state

- Verified pre-capture backup:
  `pc_captures/state_backups/session-20260728-203153`
- Backup `SHA256SUMS` SHA-256:
  `5b693c76ec30858e55df84c22a7f61c22ff2b2eec0a9b2195a4e1b2725906676`
- Immutable ASCII session root:
  `D:\ZumaGolden\session-20260728-203153`
- Steam launched the game with:
  `-record -demofile=D:\ZumaGolden\session-20260728-203153\record\input.dmo`
- The Steam launcher process is `ZumasRevenge.exe`; the visible game window is
  actually owned by the single runtime payload
  `C:\ProgramData\PopCap Games\ZumasRevenge\popcapgame1.exe`.
- The game was exited through its own menus. The committed DMO was copied to
  `pilot-level1-navigation\input.dmo`; it is 67,941 bytes with SHA-256
  `8f7b62446cb3aad11ff21f0bc8556ede02117d8398f758baccb4826c8e08b421`.
- The profile was deliberately restarted from Zone 1 with the user's
  authorization; the previous score was reset. The game is currently stopped,
  `InProgress=0`, and `LastShutdownOK=1`.
- NVIDIA's statistics overlay was removed by temporarily stopping
  `NvContainerLocalSystem`. Final cleanup restored it to `Running` with
  `Automatic` start mode.
- `FvSvc` remained running. Do not assume its state controls the visible
  statistics overlay.
- Failed probe directories `probe-4k-attempt-01` through
  `probe-4k-attempt-05`, plus `.part` files in `record`, are diagnostic debris
  from the preallocation investigation. Never reuse any of those directories
  as completed evidence.

## Prepared Windows runtime

```text
D:\ZumaGolden\tools\dxcam-venv
D:\ZumaGolden\tools\wheelhouse
```

Locked packages:

```text
CPython 3.12.10
dxcam 0.3.0
comtypes 1.4.16
numpy 2.5.1
```

The wheelhouse `SHA256SUMS` passes. A `pip --dry-run --no-index
--require-hashes` install resolves exactly those three packages.

DXGI device `0`, output `0` is:

```text
NVIDIA GeForce RTX 5080
\\.\DISPLAY1
3840x2160, rotation 0, 164 Hz
```

CUDA GPU `0` is the RTX 5090 and remains available for later training. DXGI
and CUDA index numbers are separate namespaces.

## Findings from the first live attempts

The ordinary `DXCamera.grab(copy=True)` path occasionally returned
`AccumulatedFrames=2` at 3840×2160/164 Hz and therefore correctly failed with
`missed_presentations_detected`. The standard was not relaxed.

`tools/capture_dxgi.py` now allocates and pre-touches an owned frame pool before
the capture boundary and fills each destination through DXcam's pinned
`_grab_into` path. The standalone cadence diagnostic completed a 0.25 second
run with every post-warm-up `AccumulatedFrames` value equal to one. The full
test suite passes.

Two production-path probes subsequently completed with no missed
presentations and no `.part` files:

```text
probe-4k-prealloc-01
  title screen, 25 frames, 829440000 raw bytes
  observed mean present rate: 100.06971523494701 Hz
  raw SHA-256:
    sha256:3cfb5321decff659aa75224e0b8adacb79fb41f87625e11570eb1e389f5687c1

probe-level1-noop-01
  active Level 1, no shot input, 42 frames, 1393459200 raw bytes
  observed mean present rate: 165.0607463805399 Hz
  raw SHA-256:
    sha256:78ff5bb8b42b0c3000bd41a02c8fef8eb713ac7ae2bee1621d718e30b3fb79d1
```

Independent verification recomputed every frame hash, raw hash, CSV hash, and
aggregate pixel hash; checked exact offsets and sizes; and confirmed strictly
increasing QPC and host timestamps with `AccumulatedFrames == 1`. First,
middle, and last frames are unobscured and contain no NVIDIA overlay.

The Level 1 probe had 42 real presents but 26 distinct pixel hashes. Consecutive
identical-pixel runs had histogram `{1: 10, 2: 16}`; present spacing averaged
6.058 ms while visible pixel changes averaged 9.695 ms. The 164 Hz presentation
clock and roughly 10 ms game update clock must remain separate in calibration.

Turning off `Full Screen` while `Hi-Res (1920x1200)` remained disabled produced
an exact 800×600 Win32 client region at `[1520,591,2320,1191]`. That is the
native logical canvas, not a downsample. The old DPI-unaware game exposes only
a 459×374 upper-left crop to Computer Use at the current 175% Windows scaling,
so the run was returned to full screen for reliable UI control. A later
Windows-native compatibility pass should make the 800×600 window fully
controllable without changing its client pixels.

Windows locks the running `popcapgame1.exe` payload and deletes it on normal
exit, so post-exit hashing is impossible. The persistent Steam
`ZumasRevenge.exe` contains one signed embedded PE at byte offset 1,925,538.
Its section table ends at 6,651,904 bytes and its certificate directory adds
5,424 bytes, deriving the exact 6,657,328-byte runtime file observed live.
`tools/embedded_pe_payload.py` independently computes:

```text
persistent launcher SHA-256:
  sha256:db85b891ba662a463d23251a376081386b47437c29248b21e2c10335a6f3eb50
embedded runtime SHA-256:
  sha256:2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20
```

Capture schema v2 accepts this binding through
`--runtime-source-executable`; the target PID, creation time, HWND, live file
size, persistent launcher hash, embedded offset/extent, certificate extent,
and runtime payload hash are all retained in metadata. Direct runtime hashing,
when available, must agree. A live cleanup-window read directly hashed the
temporary runtime as
`2181ce2bfbfcb4678bf69a1474e08d3db941311aa768176a88453cc69692af20`,
exactly matching the embedded payload. `--defer-locked-executable-hash`
remains probe-only and the encoder still rejects it.

The pilot DMO parses as v2, product 1.0.4.9496, random seed 5,909,046,
407,286 framework updates, 28,340 commands, and 1,111 input commands. It
contains menu navigation, tutorial dismissal, display-mode changes, and exit
actions; it is not a clean action Golden. In particular, it deactivates the
application at update 145 and does not reactivate it until update 12,551.
Playback started both launcher and runtime processes but never created a
window during more than 140 seconds of observation because inactive framework
updates could not reach the later activation command. The exact processes were
terminated, the game was relaunched normally, and it was exited through its
own menu; the final registry state is `InProgress=0`, `LastShutdownOK=1`.
This pilot is retained only as a parser/navigation diagnostic and must never be
used as deterministic replay evidence.

The replacement recording is:

```text
D:\ZumaGolden\session-20260728-203153\clean-level1-001\input.dmo
bytes: 62036
sha256:044fe875b5c77e2a8b5a0f2193a7e9f90c4cb719481a1704d19651bb50ebc10a
```

It parses as DMO v2 for product 1.0.4.9496 with seed 23,775,218, 8,942
framework updates, 7,002 commands, 6,648 input commands, 34 complete left
clicks, and one `loading_complete` at update 439. It contains no
`activate_app`, `mouse_exit`, or incomplete button command, so it has none of
the pilot's focus deadlock.

Two `-play` smoke runs from a clean `InProgress=0` pre-state both created the
expected 1.0.4.9496 window and reached process exit without a Windows crash
event, WER report, residual process, or residual runtime payload. Their wall
times differed (about 16.1 s and 10.5 s), confirming that replay advances as
fast as the host allows; wall time is not a determinism metric. Replay leaves
the real registry at `InProgress=1`, consistent with recorded registry writes
being replay-isolated, so each run was followed by a normal menu launch/exit.
Final state was rechecked as `InProgress=0`, `LastShutdownOK=1`.

These are process-level smoke checks only. They do not yet prove identical
pixels, visible event updates, or simulator-tick phase and must not be entered
as `replay_determinism` evidence.

## Native windowed replay scheduler and lossless evidence

The later native-windowed recording is retained unchanged at:

```text
D:\ZumaGolden\session-20260728-203153\native-windowed-record-001\input.dmo
bytes: 21976
sha256: aa807c1686c07277aaa85c835c48ede5ebb2f6c24c86d7cc372140786bdaafde
```

Playback uses the provenance-bound startup-aligned diagnostic derivative:

```text
D:\ZumaGolden\session-20260728-203153\native-windowed-record-001-aligned-diagnostic\input.dmo
bytes: 21962
commands: 1357
sha256: e43c645a18d74c5320a0687446f38cd9c46fff4eb8c0ff1a2b9c265b78a18b12
```

The only transformation removes the bit-identical duplicate startup
`registry_read` observed between `-record` and `-play`. The raw recording is
preserved and the aligned derivative remains diagnostic; neither may be
presented as an unmodified retail Golden.

Every independent replay restored the same reconstructed prestate:

```text
users.dat         99a6996f6105eccc3e1bb2f95b5d2ec2a1d261e6d63aa22cc213e5c88f3e26f6
user2.dat         8bfbb6b7321169c19539f9c0e580bf0de7436f60bd1e0f7f63433e07aa0cd672
adv_in_game2.sav  0009b6b3309aaa1bd6689e8093d66f36fea43648ebc3ad66ec911ddf1ee23539
```

Static caller identification showed that retail registry/file commands are
required. The divergence was instead a scheduler race: the modern host could
let the main `PrepareDemoCommand` caller skip a service block before the
original loading/service thread consumed it. `tools/trace_popcap_demo_commands.py`
now suspends only that racing main thread, lets the original service code
consume the exact contiguous same-update offline block, checks the resulting
bit boundary, resumes the main thread, single-steps the restored original
instruction, and re-arms the breakpoint. Every live entry is also checked for
offline start bit, command order, number, short form, and update.

An intentionally retained first structured run failed closed at update 388.
Its altered logging cadence exposed the second arrival ordering where the
service command was already prepared. The broker now handles both
`needs_command` states; a 501-update probe passed 62 entries with zero
failures. Subsequent independent runs produced:

```text
run-002: update 9006, 1250 hits, 4 brokered blocks, 0 failures
run-003: update 9006, 1234 hits, 15 brokered blocks, 0 failures
run-004: update 17967, 2484 hits, 15 brokered blocks, 0 failures
run-005: exit code 0 after command 1355 at bit 175397, 0 boundary failures
```

The DMO has one final successful `registry_write` result at bits
175397..175408 which the playback process does not request after normal
process exit. This is explicitly classified as a one-command terminal
bookkeeping tail, not as a consumed command or a gameplay mismatch.

Two independent raw acquisitions contain exactly three committed source
files each, 495 800x600 BGRA frames, 950,400,000 raw bytes, and zero missed
presentations:

```text
capture-001
  metadata sha256: 64ee23ed65bb41bfc52a140d7c24324eaac6ecea707ea6307f01ca7515aaf1a4
  raw sha256:      58db9f271fd13cb070a2a31be0c5e51d5f1d0dcfb923affac1ffd9d8e6382629
  pixel aggregate: a8fd191ba5e545e97b8280b24954ee1c70e7ea2b39977b3913df2aa73d67211b
  FFV1 bytes:      396580939
  FFV1 sha256:     18202bb1440aca9c4bae1e4a29ad8fc16a3d5ae465ce74a4411e69922886da8c

capture-002
  metadata sha256: 219b341120b72f740c69bc0dcd117e84bdb50b10c7d74b591b8f7760af021493
  raw sha256:      8a06475b60dec109a1a8b40f6e669ea5d5c8a3fe57c02597b1fa22a0f06035d0
  pixel aggregate: 3fa3c8018c2201122c1a363f6419c81a803d5af76c27713e7ca9ea680fe1c648
  FFV1 bytes:      394866413
  FFV1 sha256:     c37d3fbdbfb713dda4cb5216f6a2f073fc4b83da9ddd818c784c0b35476ffd6f
```

For `capture-002`, result schema v2 proves the target PID is 54544 in both
artifacts and the complete capture interval
`41925958801800..41928959017000` ns is inside the strict trace interval
`41771078259200..41948744473600` ns. Both validated FFV1 files were fully
decoded; every PTS and every BGRA frame hash matched the immutable raw input.
Representative middle frames show unobscured active Level 2 gameplay.

The original failed encoder output
`native-windowed-strict-broker-capture-001.ffv1.mkv.part` is diagnostic debris
from an overly exact Matroska advisory-rate check. It must not be published.
The two `.validated.ffv1.mkv` files are the successfully verified artifacts.

Final handoff restoration copied all eight files from
`D:\ZumaGolden\state-backups\pre-reconstructed-native-windowed-replay-001\users`
back to the live users directory. Relative paths, sizes, and SHA-256 hashes
match the backup inventory exactly. No Zuma launcher/runtime process remains;
the registry is `InProgress=0`, `LastShutdownOK=1`, `ScreenMode=0`, and
`HiRes=0`.

## Next steps

1. Add the replay-result/capture binding, terminal bookkeeping tail, and save
   transaction to evidence v4 and the fail-closed verifier.
2. Compare both decoded captures at visible anchors and establish DMO-update
   to native-tick phase from the first shot.
3. Capture one-shot, insertion, match, rollback, Zuma, failure, and level-end
   boundaries separately.
4. Diff those PC traces against `ZumaRevenge-v0`; keep the Fidelity Gate closed
   until all required mechanics and observation channels pass.

Do not encode either deferred-hash probe, call it a PC Golden, or open the
Fidelity Gate.
