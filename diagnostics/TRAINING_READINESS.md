# 训练就绪状态（2026-07-30）

## Fidelity Gate

- policy：`original-transfer-jungle2-v1`
- 状态：`CLOSED`
- PC Golden：1/8
- 独立 simulator differential 来源：7/8
- PC DMO seed：1/3
- 已完整通过：原版双回放确定性、actor 无隐藏状态边界
- 仍缺：自然 PC 机制矩阵、500-tick 长时无漂移窗口、完整 PC 视觉可恢复性审计

功能标签现由报告正文机器推导，不再直接信任 suite 手写声明。两份旧反向道具
报告只证明生命周期、换球报告只证明输入接收，因此对应机制不会虚增计数。
当前 Gate 有 27 条关闭原因。

视觉审计已实现并现场复核：原版 PC Golden 与 actor 契约本身均为 `PASS`，
72 个 actor 特征中 28 个已有像素／几何／agent API 推导证明，44 个仍缺，
所以视觉总状态保持 `INCOMPARABLE`。

PC mechanism audit 已实现并接入 Gate 的 native-source 交叉授权。计划文件
不能声明 feature；工具只接受无注入、首帧与原版录像逐像素绑定的 v2
`frozen_post_replay_update_barrier` 轨迹，并自动推导发射、碰撞、精确
三／四连消和 rollback chain。现有 u7738／u9910 物理轨迹是旧 v1 协议，
仍作为诊断保留但不能认证；C2/C4/C5 新采集必须产出 v2 轨迹。

C2/C4/C5 的第三次原版 memory follow-up 也已内容寻址并通过静态验证：
C2/C4 各 501 tick，C5 为 201 tick；全部使用冻结后只读分数发现。当前三项
均安全停在 `waiting_for_pc_golden_collection`，执行器会在相应双回放完成前
拒绝启动。

`zuma-train --transfer-training` 已和 Gate 硬连接。当前 suite 请求一百万步时会
在导入 Stable-Baselines3／占用 GPU 前退出，不能用确认参数绕过。

## 回归

- 本轮完整测试套件 553/553 PASS；其中新增机制审计、Gate 交叉授权、
  read-only score discovery、PC Golden、原版验收、内存证据和视觉审计均已
  纳入
- 唯一 warning 是 Gymnasium 对直接构造环境、没有 registry spec 时无法枚举
  其他 render mode；不影响环境契约。
- actor observation 现包含可见 powerup ID one-hot，共 72 个逐球／弹丸／全局
  特征类别。

## GPU 与运行时

- GPU 0：NVIDIA GeForce RTX 5090，32,607 MiB，compute capability 12.0
- GPU 1：NVIDIA GeForce RTX 5080，16,303 MiB，保留给用户
- NVIDIA driver：610.88
- 项目 PyTorch：`2.12.1+cu130`
- CUDA runtime：13.0；cuDNN：92000
- `CUDA_VISIBLE_DEVICES=0` 下设备数：1，设备名 RTX 5090
- 4096×4096 CUDA matmul：通过，结果有限，约 0.224 秒，峰值约 336 MiB
- 真实 Stable-Baselines3 PPO CUDA 烟测：4 环境、512 steps、2 iterations
  通过，输出位于 `diagnostics/cuda-smoke-20260730-v1`

烟测退出后 5090 显存已释放；5080 未被训练进程使用。对于当前小型 MLP PPO，
Stable-Baselines3 也提示 CPU 往往更快；5090 的主要价值会在后续视觉 CNN
编码器和大 batch 上体现。

## 短时 CPU 吞吐

在用户同时使用电脑时，以 8 个 async 环境、1-tick action、随机策略测得：

- Jungle2：约 1,175 decisions/native ticks/s，约 1.015 亿 transitions/day；
- Jungle9 双曲线：约 882 decisions/native ticks/s，约 0.762 亿
  transitions/day。

这只是 5 秒后台烟测，且没有完整 episode 结束，不能据此宣称每天能通关多少
局。正式预算应在 Gate `OPEN` 和目标视觉 PPO 配置确定后重新测量。
