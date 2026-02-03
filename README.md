# AGI Phase 2: Governed Continual Learning System

<p align="center">
  <strong>🧠 Dynamically Extend Static Transformer Models Without Retraining</strong>
</p>

<p align="center">
  <a href="#中文版本">中文</a> •
  <a href="#english-version">English</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a>
</p>

---

# English Version

## 🎯 Vision

**Build a governed continual learning system that dynamically extends static Transformer models without pre-training.**

Traditional AI systems face a fundamental dilemma: **capability vs. safety**. Powerful models are difficult to control, while safe models lack adaptability. AGI Phase 2 solves this through a **three-layer closed-loop architecture**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Service Layer (Transformer)    │  Powerful, but frozen params  │
├─────────────────────────────────┼───────────────────────────────┤
│  Self-Learning Layer (NL)       │  Continual learning, isolated │
├─────────────────────────────────┼───────────────────────────────┤
│  Governance Layer (NLGSM)       │  Human oversight, controls    │
└─────────────────────────────────────────────────────────────────┘

Core Principle: "Innovation Under Control"
- Allow AI to retain powerful learning capabilities
- But strictly limit its impact within safety boundaries
```

## 🌟 Key Features

### 1. Zero-Training Knowledge Injection

-   **AGA (Auxiliary Governed Attention)**: Hot-pluggable knowledge injection without gradient computation
-   **Runtime Dynamic**: Add/remove knowledge at runtime
-   **Instant Isolation**: Problematic knowledge can be immediately quarantined

### 2. Nested Learning Paradigm

-   **Multi-Frequency Optimization**: PARAMETER → MEMORY → OPTIMIZER → POLICY
-   **Continuum Memory System**: Isolated experimental memory for learning
-   **Context Flow**: Auditable learning process with full traceability

### 3. NLGSM Governance Framework

-   **8-State FSM**: LEARNING → VALIDATION → FROZEN → RELEASE → ROLLBACK → SAFE_HALT → DIAGNOSIS → RECOVERY
-   **Event-Decision-Action Pipeline**: Structured governance workflow
-   **Human-Centric**: Humans define rules, audit results, and approve production deployment

### 4. Production-Ready Backend

-   **Multi-dimensional Anomaly Detection**: Metric, Behavior, Drift, External detectors
-   **Transactional Rollback**: Atomic operations with snapshot recovery
-   **Comprehensive Observability**: Prometheus metrics, health checks, alerting

## 📐 Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Request                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🔷 Service Layer                                     │
│                         Transformer Model                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  Input → Transformer Backbone → Hidden States → Decision Head → Output │  │
│  │                                      │                                 │  │
│  │                                      ▼                                 │  │
│  │                          ┌─────────────────────┐                       │  │
│  │                          │   AGA Module        │                       │  │
│  │                          │   (Knowledge Slots) │                       │  │
│  │                          │   Hot-pluggable     │                       │  │
│  │                          └─────────────────────┘                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       🔶 Self-Learning Layer                                 │
│                       Nested Learning Paradigm                               │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │   │PARAMETER │  │ MEMORY   │  │OPTIMIZER │  │ POLICY   │             │  │
│  │   │ (Fast)   │  │ (Medium) │  │ (Slow)   │  │(Slowest) │             │  │
│  │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘             │  │
│  │        └─────────────┴─────────────┴─────────────┘                    │  │
│  │                              │                                        │  │
│  │                              ▼                                        │  │
│  │                    Learning Unit Builder                              │  │
│  │                    (Chainable, Concurrent)                            │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🔴 Governance Layer                                  │
│                         NLGSM Framework                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                     Finite State Machine                               │  │
│  │   LEARNING → VALIDATION → FROZEN → RELEASE                            │  │
│  │       ↑                                 ↓                              │  │
│  │   ROLLBACK ← SAFE_HALT ← DIAGNOSIS ← RECOVERY                         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   Event Layer → Decision Layer → Action Layer                         │  │
│  │   (Anomaly)      (Rules)          (Transitions)                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   👤 Define Rules → 👤 Audit → 👤 Manage Lifecycle → 👤 Approve       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

> **Note**: AGA (Auxiliary Governed Attention) has been separated into an independent project. See [AGA Repository](../AGA/README.md) for details.

> **Note**: The `bridge/` module has been **deprecated**. Knowledge transfer is now handled by `backend/app/services/knowledge_transfer_service.py` + AGA API Portal.

```
AIFuture/
├── self_learning/              # 🧠 Self-Learning System
│   ├── nl_core/               # Nested Learning Core
│   │   ├── kernel.py          # NL Kernel (LLM-based)
│   │   ├── memory.py          # Continuum Memory System
│   │   ├── types.py           # Core types & LearningScope
│   │   └── optimizer.py       # Multi-level optimizer
│   ├── explorer.py            # Autonomous exploration engine
│   ├── knowledge_generator.py # Knowledge generation
│   ├── knowledge_reader.py    # Production knowledge reader
│   ├── chainable_learning_builder.py  # Chain learning support
│   ├── nl_learning_unit_builder.py    # NL-based LU builder
│   ├── learning_unit_builder.py       # Base LU builder
│   ├── learning_unit_state.py         # LU state management
│   ├── concurrent_learner.py  # Multi-threaded learning
│   ├── async_learning_model.py # Non-blocking async model
│   ├── checkpoint.py          # Learning checkpoint
│   └── governance_interface.py # Governance integration
│
├── bridge/                     # ⚠️ DEPRECATED - Use knowledge_transfer_service
│   └── (legacy code, retained for reference)
│
├── backend/                    # 🏢 NLGSM Backend (Governance System)
│   └── app/
│       ├── api/               # REST API endpoints
│       ├── core/
│       │   ├── anomaly/       # Multi-dimensional anomaly detection
│       │   ├── eda/           # Event-Decision-Action pipeline
│       │   └── observability/ # Metrics, health, alerting
│       ├── services/
│       │   ├── knowledge_transfer_service.py  # ★ AGA Portal integration
│       │   ├── state_machine_service.py       # FSM implementation
│       │   ├── governance_service.py          # Governance operations
│       │   ├── learning_unit_service.py       # LU management
│       │   ├── learning_control_service.py    # Learning control
│       │   ├── approval_service.py            # Multi-sig approvals
│       │   ├── artifact_service.py            # Governed artifacts
│       │   ├── diagnosis_service.py           # Diagnosis & recovery
│       │   ├── anomaly_detection_service.py   # Anomaly detection
│       │   ├── observability_service.py       # Observability
│       │   └── ...                            # Auth, User, Notification, etc.
│       ├── models/            # Database models
│       ├── schemas/           # Pydantic schemas
│       ├── middleware/        # Auth, logging middleware
│       └── db/                # Database setup
│
├── llm/                        # 🤖 LLM Adapters
│   ├── adapters/              # DeepSeek, Ollama, vLLM, OpenAI
│   ├── client.py              # Unified LLM client
│   ├── prompts.py             # Prompt templates
│   └── risk_evaluator.py      # Risk evaluation
│
├── web/                        # 🌐 Frontend (Vue.js)
│   └── src/                   # Vue components & pages
│
└── examples/                   # 📚 Demo Scripts
    ├── chainable_learning_demo.py
    ├── concurrent_learning_demo.py
    ├── async_learning_demo.py
    ├── governance_intervention_demo.py
    └── llm_adapter_demo.py
```

### Architecture Change: Knowledge Transfer

The knowledge transfer flow has been redesigned:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  OLD (Deprecated)                                                        │
│  self_learning → bridge/ → AGA (embedded)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  NEW (Current)                                                           │
│  self_learning → knowledge_transfer_service.py → AGA Portal (HTTP API)  │
│                                                                          │
│  Key Changes:                                                            │
│  - AGA is now a standalone project with its own API Portal              │
│  - Governance system only passes semantic text (condition/decision)     │
│  - KV encoding is handled by AGA Portal internally                      │
│  - Supports multi-tenant, distributed deployment                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt

# Optional: Redis & PostgreSQL for production
docker-compose up -d redis postgres
```

### Basic Usage

```python
from self_learning import (
    ChainableLearningUnitBuilder,
    AsyncLearnerPool,
    LearningScope,
)
from self_learning.nl_core import LLMBasedNLKernel, ContinuumMemorySystem

# 1. Initialize NL Kernel (uses existing LLM knowledge as starting point)
kernel = LLMBasedNLKernel(
    llm_client=your_llm_client,
    cms=ContinuumMemorySystem(),
)

# 2. Create Learning Unit Builder
builder = ChainableLearningUnitBuilder(
    nl_kernel=kernel,
    production_knowledge_reader=reader,
    max_chain_depth=10,
)

# 3. Smart Learning (auto-selects starting point)
learning_unit = builder.smart_learn(
    goal="Learn to handle customer complaints about delivery delays",
    scope=LearningScope(max_level=NLLevel.MEMORY),
)

# 4. Submit to Governance for approval
governance.submit_for_review(learning_unit)
```

### Concurrent Learning

```python
from self_learning import AsyncLearnerPool, AsyncLearningCoordinator

# Create async learner pool
pool = AsyncLearnerPool(num_learners=4)
coordinator = AsyncLearningCoordinator(pool, state_manager)

# Start learning
pool.start()

# Submit tasks (non-blocking)
task_id = pool.submit_task(LearningTask(
    goal="Learn new domain knowledge",
    scope=default_scope,
))

# Tasks continue even while waiting for human approval
# Coordinator handles status notifications asynchronously
```

### AGA Knowledge Injection

```python
from aga import AGA, AGAConfig, LifecycleState

# Create AGA instance
config = AGAConfig(
    hidden_dim=4096,
    num_slots=128,
    top_k_routing=8,
)
aga = AGA(config=config)

# Inject knowledge (zero-training)
aga.inject_knowledge(
    slot_idx=0,
    key_vector=key_vec,
    value_vector=value_vec,
    lu_id="LU_001",
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# Attach to model
manager = AGAManager()
manager.attach_to_model(model, layer_indices=[-2, -1])
```

## 🔒 Core Invariants

```
🔒 Invariant 1: Self-learning cannot directly modify production parameters
   - All knowledge must go through governance approval

🔒 Invariant 2: Governance can trigger rollback at any time
   - NLGSM maintains full control over system state

🔒 Invariant 3: Learning starting point is LLM's existing knowledge
   - Not starting from zero, but building on pre-trained capabilities

🔒 Invariant 4: AGA is always bypassable
   - Any exception → AGA = NO-OP, system continues safely
```

## 📊 Monitoring

Key metrics exposed via Prometheus:

| Metric                           | Description             |
| -------------------------------- | ----------------------- |
| `nlgsm_state_transitions_total`  | State transition count  |
| `nlgsm_anomaly_events_total`     | Anomaly detection count |
| `learning_units_submitted_total` | LU submission count     |
| `aga_hit_rate`                   | AGA knowledge hit rate  |
| `aga_latency_ms`                 | AGA forward latency     |

## 🛣️ Roadmap

-   [x] **Phase 1**: Core NL Framework + NLGSM Backend
-   [x] **Phase 2**: Chainable Learning + Concurrent Execution
-   [x] **Phase 2.1**: Async Learning Model + P0/P1/P2 Features
-   [ ] **Phase 3**: AGA Production Runtime
-   [ ] **Phase 4**: Multi-model Support + Distributed Learning

---

# 中文版本

## 🎯 愿景

**构建一个基于治理的持续学习系统，动态无损扩展静态 Transformer 模型，无需预训练。**

传统 AI 系统面临一个根本性的两难困境：**能力 vs. 安全**。强大的模型难以控制，而安全的模型缺乏适应性。AGI Phase 2 通过**三层闭环架构**解决这一问题：

```
┌─────────────────────────────────────────────────────────────────┐
│  服务层 (Transformer)        │  能力强大，但参数冻结            │
├─────────────────────────────┼───────────────────────────────────┤
│  自学习层 (Nested Learning)  │  持续学习，但隔离在实验内存      │
├─────────────────────────────┼───────────────────────────────────┤
│  治理层 (NLGSM)             │  人类监督，控制知识流入生产       │
└─────────────────────────────────────────────────────────────────┘

核心原则："控制下的创新"
- 允许 AI 保留强大学习能力
- 但将其影响范围严格限制在安全边界内
```

## 🌟 核心特性

### 1. 零训练知识注入

-   **AGA（辅助治理注意力）**：热插拔式知识注入，无需梯度计算
-   **运行时动态**：运行时添加/移除知识
-   **即时隔离**：问题知识可立即隔离

### 2. 嵌套学习范式

-   **多频率优化**：PARAMETER → MEMORY → OPTIMIZER → POLICY
-   **连续记忆系统**：隔离的实验记忆用于学习
-   **上下文流**：可审计的学习过程，完整追溯

### 3. NLGSM 治理框架

-   **8 状态 FSM**：学习 → 验证 → 冻结 → 发布 → 回滚 → 安全停机 → 诊断 → 恢复
-   **事件-决策-动作管道**：结构化治理工作流
-   **人类中心**：人类定义规则、审计结果、批准生产部署

### 4. 生产就绪后端

-   **多维异常检测**：指标、行为、漂移、外部检测器
-   **事务性回滚**：原子操作与快照恢复
-   **全面可观测性**：Prometheus 指标、健康检查、告警

## 📐 架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                用户请求                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🔷 服务层                                          │
│                           Transformer 模型                                   │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │  输入 → Transformer Backbone → Hidden States → Decision Head → 输出   │  │
│  │                                      │                                 │  │
│  │                                      ▼                                 │  │
│  │                          ┌─────────────────────┐                       │  │
│  │                          │   AGA 模块          │                       │  │
│  │                          │   (知识槽位)        │                       │  │
│  │                          │   热插拔            │                       │  │
│  │                          └─────────────────────┘                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         🔶 自学习层                                          │
│                         嵌套学习范式                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐             │  │
│  │   │ 参数层   │  │ 记忆层   │  │ 优化器层 │  │ 策略层   │             │  │
│  │   │ (最快)   │  │ (中等)   │  │ (较慢)   │  │ (最慢)   │             │  │
│  │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘             │  │
│  │        └─────────────┴─────────────┴─────────────┘                    │  │
│  │                              │                                        │  │
│  │                              ▼                                        │  │
│  │                    Learning Unit 构建器                               │  │
│  │                    (链式、并发)                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           🔴 治理层                                          │
│                           NLGSM 框架                                         │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                        有限状态机                                      │  │
│  │   学习 → 验证 → 冻结 → 发布                                           │  │
│  │    ↑                      ↓                                           │  │
│  │   回滚 ← 安全停机 ← 诊断 ← 恢复                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   事件层 → 决策层 → 动作层                                             │  │
│  │   (异常)    (规则)    (迁移)                                           │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │   👤 定义规则 → 👤 审计 → 👤 管理生命周期 → 👤 批准                    │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

> **说明**：AGA（辅助治理注意力）已分离为独立项目。详见 [AGA 仓库](../AGA/README.md)。

> **说明**：`bridge/` 模块已**弃用**。知识转移现由 `backend/app/services/knowledge_transfer_service.py` + AGA API Portal 处理。

```
AIFuture/
├── self_learning/              # 🧠 自学习系统
│   ├── nl_core/               # 嵌套学习核心
│   │   ├── kernel.py          # NL 内核（基于 LLM）
│   │   ├── memory.py          # 连续记忆系统
│   │   ├── types.py           # 核心类型 & LearningScope
│   │   └── optimizer.py       # 多层优化器
│   ├── explorer.py            # 自主探索引擎
│   ├── knowledge_generator.py # 知识生成器
│   ├── knowledge_reader.py    # 生产知识读取器
│   ├── chainable_learning_builder.py  # 链式学习支持
│   ├── nl_learning_unit_builder.py    # 基于 NL 的 LU 构建器
│   ├── learning_unit_builder.py       # 基础 LU 构建器
│   ├── learning_unit_state.py         # LU 状态管理
│   ├── concurrent_learner.py  # 多线程学习
│   ├── async_learning_model.py # 非阻塞异步模型
│   ├── checkpoint.py          # 学习检查点
│   └── governance_interface.py # 治理集成
│
├── bridge/                     # ⚠️ 已弃用 - 请使用 knowledge_transfer_service
│   └── (保留旧代码供参考)
│
├── backend/                    # 🏢 NLGSM 后端（治理系统）
│   └── app/
│       ├── api/               # REST API 端点
│       ├── core/
│       │   ├── anomaly/       # 多维异常检测
│       │   ├── eda/           # 事件-决策-动作管道
│       │   └── observability/ # 指标、健康、告警
│       ├── services/
│       │   ├── knowledge_transfer_service.py  # ★ AGA Portal 集成
│       │   ├── state_machine_service.py       # FSM 实现
│       │   ├── governance_service.py          # 治理操作
│       │   ├── learning_unit_service.py       # LU 管理
│       │   ├── learning_control_service.py    # 学习控制
│       │   ├── approval_service.py            # 多签审批
│       │   ├── artifact_service.py            # 受治理工件
│       │   ├── diagnosis_service.py           # 诊断与恢复
│       │   ├── anomaly_detection_service.py   # 异常检测
│       │   ├── observability_service.py       # 可观测性
│       │   └── ...                            # 认证、用户、通知等
│       ├── models/            # 数据库模型
│       ├── schemas/           # Pydantic 模式
│       ├── middleware/        # 认证、日志中间件
│       └── db/                # 数据库配置
│
├── llm/                        # 🤖 LLM 适配器
│   ├── adapters/              # DeepSeek, Ollama, vLLM, OpenAI
│   ├── client.py              # 统一 LLM 客户端
│   ├── prompts.py             # 提示词模板
│   └── risk_evaluator.py      # 风险评估
│
├── web/                        # 🌐 前端（Vue.js）
│   └── src/                   # Vue 组件和页面
│
└── examples/                   # 📚 示例脚本
    ├── chainable_learning_demo.py
    ├── concurrent_learning_demo.py
    ├── async_learning_demo.py
    ├── governance_intervention_demo.py
    └── llm_adapter_demo.py
```

### 架构变更：知识转移

知识转移流程已重新设计：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  旧架构（已弃用）                                                        │
│  self_learning → bridge/ → AGA（内嵌）                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  新架构（当前）                                                          │
│  self_learning → knowledge_transfer_service.py → AGA Portal（HTTP API）│
│                                                                          │
│  主要变更：                                                              │
│  - AGA 现为独立项目，拥有自己的 API Portal                              │
│  - 治理系统只传递语义文本（condition/decision）                         │
│  - KV 编码由 AGA Portal 内部处理                                        │
│  - 支持多租户、分布式部署                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 前置条件

```bash
# Python 3.10+
pip install -r requirements.txt

# 可选：生产环境使用 Redis & PostgreSQL
docker-compose up -d redis postgres
```

### 基础使用

```python
from self_learning import (
    ChainableLearningUnitBuilder,
    AsyncLearnerPool,
    LearningScope,
)
from self_learning.nl_core import LLMBasedNLKernel, ContinuumMemorySystem

# 1. 初始化 NL 内核（使用 LLM 现有知识作为起点）
kernel = LLMBasedNLKernel(
    llm_client=your_llm_client,
    cms=ContinuumMemorySystem(),
)

# 2. 创建 Learning Unit 构建器
builder = ChainableLearningUnitBuilder(
    nl_kernel=kernel,
    production_knowledge_reader=reader,
    max_chain_depth=10,
)

# 3. 智能学习（自动选择起点）
learning_unit = builder.smart_learn(
    goal="学习处理客户关于配送延迟的投诉",
    scope=LearningScope(max_level=NLLevel.MEMORY),
)

# 4. 提交给治理系统审批
governance.submit_for_review(learning_unit)
```

### 并发学习

```python
from self_learning import AsyncLearnerPool, AsyncLearningCoordinator

# 创建异步学习池
pool = AsyncLearnerPool(num_learners=4)
coordinator = AsyncLearningCoordinator(pool, state_manager)

# 启动学习
pool.start()

# 提交任务（非阻塞）
task_id = pool.submit_task(LearningTask(
    goal="学习新领域知识",
    scope=default_scope,
))

# 即使等待人工审批，任务也会继续
# 协调器异步处理状态通知
```

### AGA 知识注入

```python
from aga import AGA, AGAConfig, LifecycleState

# 创建 AGA 实例
config = AGAConfig(
    hidden_dim=4096,
    num_slots=128,
    top_k_routing=8,
)
aga = AGA(config=config)

# 注入知识（零训练）
aga.inject_knowledge(
    slot_idx=0,
    key_vector=key_vec,
    value_vector=value_vec,
    lu_id="LU_001",
    lifecycle_state=LifecycleState.PROBATIONARY,
)

# 挂载到模型
manager = AGAManager()
manager.attach_to_model(model, layer_indices=[-2, -1])
```

## 🔒 核心不变量

```
🔒 不变量 1：自学习系统不能直接修改生产参数
   - 所有知识必须经过治理审批

🔒 不变量 2：治理系统可以随时触发回滚
   - NLGSM 保持对系统状态的完全控制

🔒 不变量 3：学习起点是 LLM 的现有知识
   - 不是从零开始，而是基于预训练能力构建

🔒 不变量 4：AGA 永远是可绕过的
   - 任何异常 → AGA = NO-OP，系统安全继续
```

## 📊 监控

通过 Prometheus 暴露的关键指标：

| 指标                             | 描述           |
| -------------------------------- | -------------- |
| `nlgsm_state_transitions_total`  | 状态迁移计数   |
| `nlgsm_anomaly_events_total`     | 异常检测计数   |
| `learning_units_submitted_total` | LU 提交计数    |
| `aga_hit_rate`                   | AGA 知识命中率 |
| `aga_latency_ms`                 | AGA 前向延迟   |

## 🛣️ 路线图

-   [x] **阶段 1**：核心 NL 框架 + NLGSM 后端
-   [x] **阶段 2**：链式学习 + 并发执行
-   [x] **阶段 2.1**：异步学习模型 + P0/P1/P2 特性
-   [ ] **阶段 3**：AGA 生产运行时
-   [ ] **阶段 4**：多模型支持 + 分布式学习

---

## 📜 License

MIT License

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines first.

## 📧 Contact

For questions and support, please open an issue on GitHub.
