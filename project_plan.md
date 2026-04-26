# 自改进机器人训练闭环 — 项目规划

## 核心思路

在仿真中实现全自动 DAgger 式的自改进闭环：
**训练 policy → 评估 → 收集失败 trajectory → VLM 自动标注 stage → 检测失败 stage → 针对性补充 expert 数据 → 重新训练 → 再评估**

整个流程全自动，无需人工遥操作、人工标注或人工定义 stage。基于 SAGE 的 MCP + Agent Loop 框架扩展。

**核心 contribution**：现有 DAgger 变体（HG-DAgger, ThriftyDAgger, Fleet-DAgger）都需要人类 expert 介入。我们用 scripted policy 作为 expert + VLM 自动发现 stage + progress model 自动检测失败，实现 **fully automated DAgger in simulation**。无已发表工作实现完全相同的全自动闭环。

---

## 1. Task（任务选择）

### 1.1 第一优先：SAGE 原生任务（PoC）
- **Pick-and-Place**（当前成功率 54.3%）：Franka 机械臂，在 SAGE 生成的室内场景中抓取并放置物体
- **Mobile Manipulation**（当前成功率 52.4%）：移动底盘 + Franka 臂，多步骤操作
- 优势：SAGE 已有完整的场景生成、数据采集（IsaacLab scripted policy）、policy 训练（Robomimic BC）流程
- 作为 **概念验证 (PoC)**，目标是在这些任务上显著提升成功率

### 1.2 第二优先：GarmentLab（可变形物体）
- Isaac Sim 原生的衣物操作基准，20 个任务
- 包含折叠、展开、悬挂等长时间序列操作
- 一旦 rigid task 的闭环跑通，迁移到这里

### 1.3 备选：FurnitureBench
- 家具组装任务，多步骤 rigid 操作
- 如果 deformable 已经搞定，rigid 的 FurnitureBench 优先级反而低
- 可作为论文中的额外实验补充

---

## 2. Scene（场景构建）

### 2.1 SAGE 原生任务的场景
- SAGE 自带场景生成 pipeline：LLM agent 调用 `generate_room_layout` → `place_objects_in_room` 等 tool
- 场景由 MCP server (`layout_wo_robot.py`) 管理，输出 USD 格式供 Isaac Sim 使用
- 物理检测：`room_physics_critic()` 做碰撞检查，`room_semantic_critic()` 做语义合理性评估
- **无需额外搭建**，直接复用

### 2.2 GarmentLab 的场景
- 衣物物理模拟的难点：柔性体 (deformable) 的物理引擎支持
- GarmentLab 本身基于 Isaac Sim 的 Particle-based cloth simulation
- 需要搞清楚：衣物初始状态如何设定、如何 reset 到特定构型
- 场景复杂度更高，可能需要自定义场景 setup

### 2.3 SceneSmith 替代方案（待评估）
- SceneSmith 的物理处理更强：Drake IK 优化 + 15秒物理 settling + 倾倒检测移除
- 5阶段 pipeline（floor_plan → furniture → wall → ceiling → manipuland），每阶段 planner/designer/critic 三个 agent
- 输出格式：Drake DMD YAML → MuJoCo MJCF → USD (Isaac Sim)
- **目前建议**：PoC 阶段先用 SAGE 原生场景，后续视需要引入 SceneSmith

---

## 3. 方法论：Automated DAgger + Failure-Aware Recovery

### 3.1 DAgger 核心思想

DAgger（Dataset Aggregation, Ross et al., AISTATS 2011）解决 BC 的 covariate shift 问题：
- BC 只在 expert 访问过的 state 上训练，policy 一旦犯错就进入没见过的 state，error 以 O(T²) compound
- DAgger 让 policy 在自己实际访问的 state 上也获得 expert 数据，error 降到 O(T)

**传统 DAgger 算法**：
```
1. 用 expert demo 训练初始 policy π₁
2. 循环 i = 1, 2, ..., N:
   a. 用 πᵢ rollout → 收集 πᵢ 实际访问的 states
   b. 在这些 states 上 query expert 得到 action labels
   c. D_i = D_{i-1} ∪ 新数据
   d. 训练 π_{i+1}
3. 返回最好的 policy
```

**我们的自动化 DAgger**：
- expert = SAGE 的 scripted policy（CuRobo 运动规划 + M2T2 抓取规划），完全不需要人
- stage 发现 = VLM 自动标注，完全不需要人定义
- 失败检测 = progress model 自动判断 stuck

### 3.2 迭代训练流程

```
Round 0:
  scripted policy 采集初始 demo → 训练 π₁

Round 1:
  π₁ rollout 大量 episode → 成功率 ~40%
  → 收集所有失败 trajectories（保存完整 state sequence）
  → VLM 自动标注每条 trajectory 的 stage 边界（全自动）
  → progress model 分析：失败集中在哪些 stage
  → replay 失败 trajectories，跳到失败 stage 前一帧 state
  → 从该 state 出发，用 scripted policy 采集 expert recovery demo
  → D_1 = D_0 ∪ recovery demo
  → 训练 π₂

Round 2:
  π₂ rollout → 成功率 ~60%
  → 重复上述流程...
```

### 3.3 Progress / Reward Model（全自动，无需人工标注）

这是闭环中最关键的技术组件 —— 自动判断 policy 在哪个 stage、进展如何、是否失败。

#### 已有方案的四种技术路线

| 路线 | 代表工作 | 原理 | 是否需要训练 | 适用性 |
|------|---------|------|-------------|--------|
| **A. 时序排序** | GVL (ICLR 2025), TOPReward (2026) | 给 VLM 看打乱的帧让它排序 / 问"任务完成了吗？"读 token 概率 | 不需要（zero-shot） | 粗粒度 progress，不能分 stage |
| **B. Stage + Progress 联合预测** | SARM (ICLR 2026) | 双头：stage 分类 + progress 回归，用语言 subtask 描述自动生成标签 | 需要训练 | 最适合我们，能知道卡在哪个 stage |
| **C. 反事实重标注** | RoboReward (2026) | 成功视频 + VLM 描述 → LLM 生成"如果目标是别的，这就是失败" → 自动标注 | 需要训练 | 通用 reward，不分 stage |
| **D. 轨迹对比** | Robometer (2026) | 帧级 progress loss + 轨迹间偏好排序，失败轨迹无需标注 | 需要训练 | 大规模泛化，不分 stage |

#### 我们的方案：路线 B 为主，路线 A 辅助

**VLM 自动 Stage 发现**（替代 SARM 中的人工 subtask 定义）：
1. 对 trajectory 的每帧图像，用 VLM 生成 caption（描述当前在干什么）
2. 计算相邻帧 caption 的 text embedding 相似度
3. 相似度突变 = stage 边界
4. 结合 proprioceptive 数据辅助（关节速度突变、gripper 开合状态变化）
5. 参考工作：Proprioception-Enhanced VLM Captioning（2025, arxiv 2512.20876）

**或者用 TOPReward 的 zero-shot 方式**（更简单）：
- 直接问 VLM "这个轨迹完成了任务吗？" → 读 token 概率作为 progress score
- TOPReward 用 Qwen3-VL 在 130+ 真实任务上达到 0.947 相关性
- 优势：不需要训练任何模型，zero-shot 即可
- 劣势：只给出总体 progress，不知道具体 stage

**推荐方案**：
1. PoC 阶段先用 TOPReward 式 zero-shot progress（最快验证闭环）
2. 进阶阶段训练 SARM 式 stage-aware progress model（需要 VLM 自动标注数据）

**失败检测逻辑**：
- 如果 progress 在连续 K 步内不增长 → 判定为 stuck
- 记录 stuck 时的 state → 作为 DAgger 中 "query expert" 的触发点

#### 关键参考工作

| 论文 | 年份/会议 | 核心贡献 |
|------|----------|---------|
| **GVL** | ICLR 2025 | VLM 做时序帧排序 = progress 估计，zero-shot，300+ 任务 |
| **SARM** | ICLR 2026 | Stage + progress 联合预测，语言 subtask 自动标签，83% 成功率 |
| **RoboReward** | 2026 | 反事实重标注自动生成 54K 训练样本，8B VLM reward model |
| **TOPReward** | 2026 | Token 概率 = reward，zero-shot，开源 VLM 也能用 |
| **Robometer** | 2026 | 1M+ 轨迹，帧级 + 轨迹级联合训练，大规模泛化 |
| **LRM** | 2026 (TRI) | 过程 reward + 完成 reward + 时序对比 reward，30 轮 RL 就提升 |
| **VLAC** | 2025 | VLA + Critic 统一架构，pairwise progress delta，30% → 90% |
| **IKER** | ICRA 2025 | VLM 生成 keypoint-based reward 代码，real-to-sim-to-real |

### 3.5 失败后的处理策略（按任务类型分类讨论）

**核心原则**：简单任务不需要 reset，直接从失败 state 继续；复杂任务才需要物理 reset。

#### 简单 Rigid 任务（Pick-and-Place）—— 不 reset，直接继续
- 物体掉到别的位置 → expert 直接从那个位置夹起来，继续完成任务
- 不需要还原到初始位置，因为 expert 有 ground truth state，无论物体在哪都能完成
- 好处：policy 学到的是"不管物体在哪，我都能完成"，更鲁棒
- 数据格式：独立的 recovery trajectory（从失败 state → 完成任务），与正常 trajectory 混合训练

#### 复杂 Deformable 任务（衣物折叠）—— 需要物理 reset
- 衣服揉成团了，直接折不可能 → 必须先恢复到可操作状态
- **物理 reset = learned recovery skill**：抓住衣角 → 抖开 → 平铺回桌面
- 这是一个 policy 自己学会的动作，不是 simulator 强制 reset
- 可迁移到真机（真机也需要这种恢复能力）
- recovery 数据采集：主动制造异常 state → scripted policy 演示恢复动作

#### 中等复杂 Rigid 任务（Mobile Manipulation / FurnitureBench）
- 视具体失败模式判断：物体掉了直接捡起来（不 reset），碰撞导致场景混乱则需要 reset
- 大多数情况下可以不 reset

**数据格式统一**：无论是否 reset，都用独立 trajectory
- 如果把 `正常执行 → 失败 → reset → 恢复 → 成功` 拼成一条，reset 瞬间有 state 不连续，BC 无法解释
- 拆成独立 trajectory，每条内部的 state 转换都是因果连续的

### 3.6 Eval 与 Data Collection 环境差异
- replay 失败 trajectory 的 state sequence：加载失败 stage 前一帧的仿真 state，瞬间回到该 stage 起点
- Isaac Sim 的 state get/set 接口支持这种操作，replay 代价极低

---

## 4. Policy Training（策略训练）

### 4.1 SAGE 当前用的方法
- **Robomimic BC**（Behavior Cloning）：MLP 网络 (1024, 1024)，直接回归 action
  - 观测：joint_pos, joint_vel, object_position, target_object_position
  - 可选 vision encoder：ResNet18 + SpatialSoftmax
  - 训练：200 epochs, batch_size=100, lr=1e-4
- **M2T2**：用于 grasp pose 生成（点云输入 → 抓取姿态）
- **CuRobo**：运动规划（避障 + 轨迹优化）
- 完整采集链：SAGE 场景 → Isaac Sim → scripted policy 演示 → HDF5 数据 → Robomimic 训练

### 4.2 候选升级方案
| 方法 | 特点 | 适合场景 |
|------|------|---------|
| **ACT** | Transformer + CVAE，chunk action prediction | 精细操作 |
| **Diffusion Policy** | DDPM 生成 action sequence，多模态分布 | 复杂操作 |
| **pi0 / pi0.5** | Flow matching，VLM backbone，language-conditioned | 通用型 |
| **Robomimic BC** | 最简单，SAGE 已有 | PoC 阶段 |

- **PoC 阶段建议**：沿用 Robomimic BC，专注验证闭环本身是否有效
- **正式实验**：考虑 Diffusion Policy 或 ACT

### 4.3 RA-BC 集成（可选增强）
- 用 progress model 对训练数据加权：进展好的帧权重高，stuck 的帧权重低
- SARM 的 rabc.py 已有完整实现：`r = φ(o_{t+Δ}) - φ(o_t)` → 归一化 → 加权 BC loss
- 可以直接复用到我们的 pipeline 中

### 4.4 DAgger 相关工作
| 方法 | 年份 | 特点 |
|------|------|------|
| **DAgger** (Ross et al.) | 2011 | 原始算法，需要 expert 在 learner 的 state 上标注 |
| **SafeDAgger** | 2017 | 学习安全 policy，只在偏离大时 query expert |
| **HG-DAgger** | 2019 | Human-gated，人决定何时介入 |
| **ThriftyDAgger** | 2022 | Budget-aware，用 novelty + risk 决定何时 query |
| **Fleet-DAgger** | 2023 | 多机器人共享人类 supervisor |
| **Diffusion Meets DAgger (DMD)** | RSS 2024 | 用 diffusion model 合成 OOD 数据，8 demo → 80% 成功率 |
| **Diff-DAgger** | 2024 | Diffusion policy 的 uncertainty 决定何时 query |
| **Ours** | 2026 | **Fully automated: scripted expert + VLM stage discovery + progress model** |

---

## 5. Eval 与 Dataset 管理

### 5.1 评估流程
- 在 Isaac Sim 中运行 trained policy
- 记录：成功率、完成 stage 数、失败 stage 分布、总步数
- 每轮 eval 后自动触发下一轮 DAgger 迭代

### 5.2 Dataset 管理
- 版本化：每轮数据采集后标记版本号（round_0, round_1, ...）
- 数据组成：
  - round_0：初始 scripted policy 采集的 demo
  - round_1：第一轮 eval 后针对失败 stage 补充的 recovery demo
  - round_2：...
- 存储格式：HDF5（与 Robomimic 兼容）
- 元数据：每条 trajectory 的来源（scripted / recovery / perturbation）、采集轮次、目标 stage

### 5.3 自动化闭环控制
- 全流程由 SAGE 的 MCP agent loop 控制
- agent 的每一步：
  1. 检查当前 eval 结果
  2. VLM 分析失败 trajectory 的 stage 分布
  3. 决定下一步采集策略（针对高失败率 stage）
  4. 调用 data collection tool（scripted expert 在失败 states 上采集）
  5. 调用 training tool
  6. 调用 eval tool
  7. 判断是否达到目标成功率，否则回到步骤 1

---

## 6. VLM Stage Annotation 具体方案

### 6.1 两轮式自适应 Stage 发现

**第一轮：粗粒度 stage 发现（VLM 调用 1 次）**
- 均匀采样 3-5 帧（首帧、中间帧、末帧 + 1-2 个中间点）
- 问 VLM："这几张图展示了一个机器人操作任务。请描述这个任务可以分成哪几个阶段，每个阶段的关键特征是什么。"
- VLM 返回 stage 定义（如："1.接近 2.抓取 3.搬运 4.放置"）
- agent 得到粗粒度 stage 数量

**第二轮：精细 stage 边界定位（VLM 调用 0-5 次 + proprioceptive 信号）**
- 用 proprioceptive 信号自动定位候选边界：
  - gripper width 突变 → 抓取/释放边界（最可靠的信号）
  - 关节速度突变 → 运动模式切换（approach 减速、lift 加速）
  - end-effector 速度接近零 → 停顿点
- 如果候选边界数量与第一轮 stage 数量吻合 → 直接用，不再调 VLM
- 如果不吻合（多了或少了）→ 在可疑帧附近再调 VLM 确认

**agent 自适应逻辑**：
- 简单任务（3-4 stages）：第一轮 + proprioceptive 信号就够，VLM 总共调 1 次
- 复杂任务（>5 stages）：agent 看到第一轮返回的 stage 多，自动决定做第二轮细化
- 不需要硬编码规则 — agent loop 天然支持迭代决策

### 6.2 最终输出格式

```json
{
  "task": "pick_and_place",
  "stages": [
    {"id": 0, "name": "approaching object", "start_frame": 0, "end_frame": 120, "key_signal": "ee_velocity > 0.1"},
    {"id": 1, "name": "grasping", "start_frame": 121, "end_frame": 145, "key_signal": "gripper_close"},
    {"id": 2, "name": "lifting and transporting", "start_frame": 146, "end_frame": 300, "key_signal": "ee_z > 0.3"},
    {"id": 3, "name": "placing", "start_frame": 301, "end_frame": 450, "key_signal": "ee_velocity decreasing"},
    {"id": 4, "name": "releasing", "start_frame": 451, "end_frame": 500, "key_signal": "gripper_open"}
  ]
}
```

---

## 7. 需要新增的 MCP Tools

到 Phase 1 Step 4（自动化封装）时需要实现：

| Tool | 功能 | 复杂度 | 依赖 |
|------|------|--------|------|
| `run_policy_eval` | 在 Isaac Sim 中运行 trained policy，录 trajectory + state | 高 | IsaacLab |
| `analyze_trajectory_progress` | VLM/progress model 分析 trajectory 进度，检测 stuck | 中 | Claude API |
| `replay_to_state` | 加载指定 trajectory 的指定帧 state 到 Isaac Sim | 中 | Isaac Sim state API |
| `collect_expert_demo` | 从指定 state 出发，用 scripted policy 采集 demo | 高 | M2T2 + CuRobo |
| `merge_dataset` | 新 demo 合并到 HDF5 dataset | 低 | h5py |
| `train_policy` | 用 Robomimic 训练 policy | 低 | 已有脚本 |
| `get_eval_summary` | 返回成功率、失败 stage 分布等统计 | 低 | 读 log |

每个 tool 先手动验证对应操作可行，再封装。不一次性全写。

---

## 8. 实现路线图

### 核心原则：先手动跑通，再逐步自动化

每个环节先手动验证可行性，确认有效后再封装成 MCP tool + 写 agent prompt。
遇到问题在手动阶段就能调整，避免浪费时间写不好用的 tool。

### Phase 1：手动验证闭环（SAGE Pick-and-Place）

**Step 1：复现 baseline**
- 跑通 SAGE 的 Pick-and-Place 数据采集 + 训练 + eval
- 确认能复现 ~54% 成功率
- 产出：可用的 eval 脚本、一批失败 trajectory

**Step 2：手动 VLM stage 标注**
- 取几条失败 trajectory，手动调 VLM API 做 stage 标注
- 验证标注质量：stage 边界是否合理、VLM 是否能看懂 Isaac Sim 渲染图
- 产出：标注方案确认、prompt 模板

**Step 3：手动 DAgger 一轮**
- 手动 replay 到失败 state → 手动跑 expert 采集 demo → 手动合并数据 → 重新训练
- 验证：成功率是否提升
- 产出：闭环有效的证据

**Step 4：封装 tool + 写 prompt，自动化**
- 把 Step 1-3 的手动操作逐个封装成 MCP tool
- 写 agent prompt 指导 LLM 按流程调用 tool
- 每个 tool 写好就测，不一次性全写

### Phase 2：多轮迭代 + Ablation
1. 自动跑多轮 DAgger，观察成功率曲线
2. Ablation：同样闭环下 MLP BC vs Diffusion Policy vs ACT
3. 扩展到 Mobile Manipulation 任务

### Phase 3：GarmentLab 扩展
1. 搭建 GarmentLab 环境
2. VLM 自动发现 deformable task 的 stage
3. 训练物理 recovery skill（揉成团 → 抖开 → 铺平）
4. 验证闭环在 deformable 任务上的效果

### 工程量估计
| 阶段 | 预计时间 | 说明 |
|------|---------|------|
| Phase 1 Step 1（复现 baseline） | 1-2 周 | 主要是环境搭建 |
| Phase 1 Step 2（VLM 标注验证） | 2-3 天 | 写 prompt + 调 API |
| Phase 1 Step 3（手动 DAgger） | 1 周 | 手动操作 + 观察结果 |
| Phase 1 Step 4（自动化封装） | 3-4 周 | 7 个 tool 逐个封装 |
| Phase 2 | 2-3 周 | 跑实验 + ablation |
| Phase 3 | 4-6 周 | 新环境 + 新 task |
| **总计** | **约 3-4 个月** | |

---

## 9. 硬件和软件需求

### API

| API | 用途 | 是否必须 | 备注 |
|-----|------|---------|------|
| **Anthropic API Key** | Claude Sonnet 4：房间结构生成、材质描述、物理/语义 critic、VLM 场景理解、替代所有本地 LLM/VLM | **必须** | SAGE 代码里已有 Claude 调用路径（`vlm.py`, `llm_client.py`） |
| ~~gpt-oss-120b~~ | 物体提议（`generate_scene_requirements`） | **需要替换为 Claude** | NVIDIA 内部模型，`layout_wo_robot.py` 中 5 处调用需改 |
| ~~TRELLIS Server~~ | 3D 物体生成 | PoC 阶段可跳过 | Pick-and-Place 用 SAGE 已有物体 |
| ~~FLUX Server~~ | 纹理生成 | PoC 阶段可跳过 | 同上 |

**结论：只需要一个 Anthropic API Key。**

### GPU 分配

**SAGE 原方案 vs 我们的方案**：

```
SAGE 原方案（NVIDIA 内部）：          我们的方案（Claude API 替代）：
Qwen3-VL-32B-Thinking → 8 GPU       Claude API        → 0 GPU
Qwen3-VL-30B-A3B      → 2 GPU       Claude API        → 0 GPU
gpt-oss-120b           → 4 GPU       Claude API        → 0 GPU
Isaac Sim              → 1 GPU       Isaac Sim         → 1 GPU
M2T2 grasp planning    → 1 GPU       M2T2              → 1 GPU
Policy 训练             → 1 GPU       Policy 训练        → 1 GPU
                                      Progress model    → 1 GPU（可选）
─────────────────────────────         ──────────────────────────────
总计：17 GPU（不可能）                 总计：3-4 GPU ✅
```

**用 Claude API 后 3 张 GPU 即可运行完整 pipeline，剩余 5 张可做并行 eval。**

### 需要改的代码（Phase 1 Step 1 的一部分）

| 文件 | 改什么 | 工作量 |
|------|--------|--------|
| `client/client_generation_room_desc.py` | Qwen3-VL OpenAI client → Anthropic client，tool calling 格式适配 | 中 |
| `server/layout_wo_robot.py` | 5 处 `gpt-oss-120b` → Claude 调用 | 小 |
| `server/vlm.py` | VLM 调用统一走 Claude（已有 `claude` 分支，改 config 即可） | 小 |
| `client/key.json` | 填入 Anthropic API key 和 URL | 极小 |
| `server/key.json` | 创建并填入 ANTHROPIC_API_KEY + MODEL_DICT | 极小 |

### 软件依赖

| 软件 | 状态 | 安装方式 | PoC 是否必须 |
|------|------|---------|-------------|
| **Isaac Sim 4.2.0** | 需下载 | NVIDIA 官网，约 10GB | **必须** |
| **IsaacLab** | 已有代码 `sage/IsaacLab/` | `pip install -e` | **必须** |
| **M2T2** | 已有代码 `sage/M2T2/` | 安装 pointnet2_ops | **必须**（expert grasp planning） |
| **Robomimic** | 已有代码 `sage/robomimic/` | `pip install -e` | **必须**（policy 训练） |
| **CuRobo** | 需安装 | pip install | **必须**（expert motion planning） |
| **Python env** | 需创建 | `client/environment.yml` | **必须** |
| Matfuse | 已有代码 `sage/matfuse-sd/` | pip install | PoC 可跳过 |
| objathor | 需下载数据 | `python -m objathor.dataset.download_*` | 场景生成需要，PoC 可跳过 |
| CLIP model | 需下载 | HuggingFace `openai/clip-vit-base-patch32` | progress model 训练需要 |

### 环境搭建顺序（建议）

```
1. 创建 conda env（client/environment.yml）
2. 下载安装 Isaac Sim 4.2.0
3. pip install -e IsaacLab
4. pip install -e M2T2 + pointnet2_ops
5. pip install -e robomimic
6. pip install curobo
7. 配置 key.json（填入 Anthropic API Key）
8. 改代码：gpt-oss-120b → Claude（5 处）
9. 改代码：Client LLM 从 Qwen → Claude
10. 测试：跑一次场景生成 → 数据采集 → 训练 → eval
```

---

## 10. 参考文献

### DAgger 系列
- Ross, Gordon, Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." AISTATS 2011.
- Zhang, Cho. "Query-Efficient Imitation Learning for End-to-End Simulated Driving." AAAI 2017. (SafeDAgger)
- Hoque et al. "ThriftyDAgger: Budget-Aware Novelty and Risk Gating for Interactive Imitation Learning." CoRL 2022.
- Hoque et al. "Fleet-DAgger: Interactive Robot Fleet Learning with Scalable Human Supervision." CoRL 2023.
- Zhang et al. "Diffusion Meets DAgger: Supercharging Eye-in-hand Imitation Learning." RSS 2024.

### Progress / Reward Model
- GVL: Vision Language Models are In-Context Value Learners. ICLR 2025 Spotlight. arxiv 2411.04549.
- SARM: Stage-Aware Reward Modeling. ICLR 2026. arxiv 2509.25358.
- RoboReward: General-Purpose Vision-Language Reward Models. 2026. arxiv 2601.00675.
- TOPReward: Token Probabilities as Hidden Zero-Shot Rewards. 2026. arxiv 2602.19313.
- Robometer: Scaling General-Purpose Robotic Reward Models. 2026. arxiv 2603.02115.
- LRM: Large Reward Models. TRI, 2026. arxiv 2603.16065.
- VLAC: Vision-Language-Action-Critic Model. 2025. arxiv 2509.15937.
- IKER: Iterative Keypoint Reward. ICRA 2025. arxiv 2502.08643.
- Proprioception-Enhanced VLM Captioning. 2025. arxiv 2512.20876.

### Scene Generation
- SAGE: Sim Agent for Generating Environments. NVIDIA 2025.
- SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes. MIT/TRI 2025.

### Benchmark
- GarmentLab: A Unified Simulation and Benchmark for Garment Manipulation.
- FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation.

## 11. 后续详情

请看autodagger/CLAUDE.md