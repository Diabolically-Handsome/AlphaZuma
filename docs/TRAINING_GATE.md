# Training Gate v3

Training Gate 约束“怎么训练”，Fidelity Gate 约束“环境是否足够可信”。
`zuma-train --transfer-training` 只有在两者都为 `OPEN` 时才继续，并在创建正式
训练器、并行环境或执行训练步前完成验证。环境合同与模型迁移的独立复验本身会
加载 SB3/Torch 验证依赖，但不会执行训练步。

## 本次标准修复

历史策略均保留为可读取、不可授权的新证据：

- `jungle2-teacher-anchor-v1` 把 98,304-step 稳定性证据外推为 491,520 步，
  外推倍数为 5，因此不再具有 transfer authority；
- `jungle2-transfer-bootstrap-v2` 虽已把上限修正为 98,304 步，但环境合同错误地
  哈希了 `src/zuma_rl` 下所有 Python 文件。只修改 PC 取证解析器也会伪造
  “训练环境漂移”，因此 v2 也转为历史策略；
- 当前策略是 `jungle2-transfer-bootstrap-v3`，stage 为
  `state-policy-bootstrap-98304-v3`。

v3 环境合同只绑定两个显式运行时源码闭包：

- environment runtime：`original_data.py`、`revenge_core.py`、`revenge_env.py`；
- policy runtime：`revenge_features.py`、`teacher_anchor.py`、`train.py`。

Fidelity/PC 取证代码仍由 Fidelity Gate 自己实时复算，不再混入训练环境身份。
合同继续绑定原版 EXE/main.pak、精确依赖版本、actor interface、环境静态指纹和
三个固定 seed 的确定性 KAT。

## 当前复算结果

当前内容寻址 suite：

`D:\ZumaGolden\training\suites\training-gate-v3-candidate-v9.json`

完整 WSL 复算结果为 `OPEN`，`reasons=[]`。当前冻结身份为：

- suite：`sha256:bedff8521857d36414689fce5682049d8dd9dca10ab010fcc64a86a76b0f8e17`；
- OPEN 报告：`sha256:137f831422e6f40ee51425a184554d767f489ff4265cc3a1b773e9ba537b1999`；
- 环境合同 v13：`sha256:bcbeb5e6b53f977aa6884baa7c670a660ac4d26800965301a4a294dff07c6974`；
- fruit actor 模型迁移 v7：`PASS`，报告
  `sha256:80dbe41aca16bb129fe337378a0f4480eff99b0268dd7767a23d12d4e9edd0b6`；
- 迁移后模型：`sha256:32e1e841962833b3c74ce743598df1c87e21c9c9c931092152e8bc2700081f64`；
- Fidelity suite：`sha256:0bfb00323d7544626b1d19452e64e55c88bacc496ecee0ba2303c7cb0ea29712`；
- Fidelity OPEN 报告：`sha256:a08d64bfd13bdbfdcb74c65163aa9e9b6a004e9d8c8fc5ff35376d252655c7d1`；
- 既有 98,304-step 稳定性实验及其 18 项 artifact 全部重新哈希通过；
- scale extrapolation factor 固定为 `1.0`。

真实 `zuma-train` 入口还完成了安全预检：两个 Gate 与完整 recipe 均被接受，
随后只因故意指定的已存在 run 目录而在创建环境、载入训练器和执行训练步之前
拒绝。这证明换成新的不存在目录即可启动固定阶段。

## 唯一授权配方

| 项目 | 固定值 |
| --- | --- |
| 环境 | 普通 `Jungle2`、教程已完成、单曲线、actor state |
| 动作 | factorized、180 aim bins、actor-observable rejection mask |
| 时序/容量 | frame skip 1、max ticks 12,000、max balls 768 |
| 初始化 | 已验证的 fruit actor migrated model |
| 算法 | TeacherAnchoredMaskablePPO |
| anchor | forward KL coefficient 1000，重置优化器，不冻结 aim path |
| 并行 | 4 env，rollout 512，batch 512，1 epoch |
| 优化 | learning rate `5e-6`、reward scale `0.01`、entropy `0` |
| 阶段步数 | 精确 98,304 effective steps |
| seed | `20260950` |
| 边界评估 | deterministic + stochastic，各 32 局，8 eval env |
| checkpoint | 每 16,384 effective steps |

任何模型、seed、环境、评估协议或超参数漂移都会失败。run 目录必须事先不存在，
以防覆盖或混合旧 artifact。

## 授权边界

当前 v3 为 `OPEN`，但它只授权一次有界 Jungle2 状态策略 bootstrap：

- `state_policy_training_only=true`；
- `visual_policy_training_authorized=false`；
- `original_game_deployment_authorized=false`；
- `required_fidelity_policy=original-transfer-jungle2-v4`。

98,304-step 阶段完成后，是否继续放大必须由新的预注册、未消费 holdout 和独立
seed 决定，不能沿用本 Gate 自动外推。

实现入口为 `src/zuma_rl/training_gate.py`；本次修复审计见
`diagnostics/GATE_REPAIR_2026-08-09.md`。
