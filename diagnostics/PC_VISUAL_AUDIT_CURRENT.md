# PC visual-derivability audit — 2026-07-30

Scope: `original-transfer-jungle2-v1`, normal Jungle2,
`profile_mode="tutorials_completed"`, state actor observation.

Live result:

- status: `INCOMPARABLE`
- live actor contract: `PASS`
- referenced PC Golden verification: `PASS`
- feature derivation coverage: `INCOMPARABLE`
- actor features: 72
- currently derivable: 28
- missing: 44
- independent PC native sources: 1

The verified source is
`cases/jungle2-shot-u7693-7770-v4-memory-curve-bound/manifest.json`,
native-source fingerprint
`sha256:5e6e648c7fe1d0e3684900f0acca444b43c8ca1b790727135051cb05314f8b19`.

The current capture proves:

- active-ball presence, positions, and colors 0–3 against retail pixels;
- one fired projectile's presence, position, and color 0;
- original Jungle2 curve geometry binding;
- full-viewport, per-tick deterministic replay history;
- actor-side constant-zero hidden channels, tunnel filtering, agent clock,
  and terminal API channels.

Missing proof capabilities:

- visible power-up icons for actor channels;
- exploding-ball state;
- projectile colors 1–3, motion history, and merge history;
- shooter orientation and current/next chambers;
- score HUD and Zuma progress bar;
- gun animation state/history;
- board phase history (`zuma_reached` / `stop_adding`).

The report intentionally remains non-certifying. Run the generator after new
PC Golden captures are available:

```bash
zuma-audit-actor-visual \
  /mnt/d/ZumaGolden/cases/<case>/manifest.json \
  --evidence-root /mnt/d/ZumaGolden \
  --original-root "/mnt/d/SteamLibrary/steamapps/common/Zuma's Revenge"
```
