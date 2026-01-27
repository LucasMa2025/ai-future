"""
异步学习系统演示

演示如何使用非阻塞异步学习模型，解决 Learner 等待状态通知的问题。

核心特性：
1. 提交即忘记（Fire-and-Forget）：Learner 提交 LU 后立即处理下一个任务
2. 事件驱动调度：状态变更通过异步事件触发后续操作
3. 不阻塞等待：人工审批可能需要数小时，学习系统不会被阻塞
4. 超时处理：自动检测和处理超时的 LU

工作流程：
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Learner   │────>│  提交 LU    │────>│ 立即返回    │
│  (执行学习)  │     │ (Fire&Forget)│     │ (不等待)    │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ PendingLU   │
                    │   Tracker   │
                    └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  NLGSM 治理  │     │  人工审批   │     │  超时处理   │
│  (自动分类)  │     │ (可能很长)  │     │ (可配置)    │
└─────────────┘     └─────────────┘     └─────────────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ▼
                    ┌─────────────┐
                    │ 状态变更事件 │
                    │  (异步通知)  │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ Coordinator │
                    │ (事件处理)  │
                    └─────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 继续学习    │     │ 新学习      │     │ 调整/停止   │
│ (提交新任务) │     │ (提交新任务) │     │ (更新状态)  │
└─────────────┘     └─────────────┘     └─────────────┘
"""
import sys
import os
import time
import threading

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_learning import (
    AsyncLearnerPool,
    PendingLUTracker,
    PendingLUStatus,
)


def print_separator(title: str = ""):
    """打印分隔线"""
    print(f"\n{'='*70}")
    if title:
        print(f"  {title}")
        print('='*70)


def print_statistics(pool: AsyncLearnerPool):
    """打印统计信息"""
    stats = pool.get_statistics()
    
    print("\n📊 系统状态:")
    print(f"  运行中: {stats['running']}")
    print(f"  学习器数量: {stats['num_learners']}")
    print(f"  任务队列大小: {stats['queue_size']}")
    
    print("\n📋 任务统计:")
    task_stats = stats['tasks']
    print(f"  总任务: {task_stats['total']}")
    print(f"  待处理: {task_stats['pending']}")
    print(f"  运行中: {task_stats['running']}")
    print(f"  已提交: {task_stats['submitted']}")
    print(f"  失败: {task_stats['failed']}")
    
    print("\n🔄 协调器统计:")
    coord_stats = stats['coordinator']
    print(f"  已处理事件: {coord_stats['events_processed']}")
    print(f"  自动提交任务: {coord_stats['tasks_auto_submitted']}")
    
    print("\n⏳ 待处理 LU 统计:")
    pending_stats = coord_stats['pending_tracker']
    print(f"  总提交: {pending_stats['total_submitted']}")
    print(f"  已解决: {pending_stats['total_resolved']}")
    print(f"  超时: {pending_stats['total_timeout']}")
    print(f"  当前待处理: {pending_stats['total_pending']}")
    print(f"  平均等待时间: {pending_stats['avg_wait_time_seconds']:.2f}秒")


def simulate_governance_decisions(pool: AsyncLearnerPool, lu_ids: list):
    """
    模拟治理系统的决策
    
    在实际系统中，这些决策来自 NLGSM 治理系统（可能包含人工审批）
    """
    print_separator("模拟治理系统决策")
    
    for i, lu_id in enumerate(lu_ids):
        # 模拟审批延迟（实际可能是几分钟到几天）
        time.sleep(0.5)
        
        # 模拟不同的决策
        if i % 3 == 0:
            # 审批通过，继续学习
            print(f"\n✅ 治理决策: LU {lu_id} - 审批通过，继续学习")
            pool.on_governance_decision(
                lu_id=lu_id,
                old_status="pending",
                new_status="approved",
                decision="continue",
                decision_params={
                    "new_goal": f"深入探索 {lu_id} 发现的知识",
                    "exploration_direction": "深度优先",
                    "focus_areas": ["核心概念", "实际应用"],
                },
            )
        elif i % 3 == 1:
            # 审批通过，开始新学习
            print(f"\n✅ 治理决策: LU {lu_id} - 审批通过，开始新方向")
            pool.on_governance_decision(
                lu_id=lu_id,
                old_status="pending",
                new_status="approved",
                decision="new_learning",
                decision_params={
                    "new_goal": "探索相关但不同的领域",
                    "domain": "general",
                },
            )
        else:
            # 需要调整
            print(f"\n🔧 治理决策: LU {lu_id} - 需要调整策略")
            pool.on_governance_decision(
                lu_id=lu_id,
                old_status="pending",
                new_status="corrected",
                decision="adjust",
                decision_params={
                    "adjusted_goal": f"调整后的学习目标 for {lu_id}",
                    "exploration_direction": "广度优先",
                },
            )


def demo_basic_async_learning():
    """演示基本的异步学习流程"""
    print_separator("演示 1: 基本异步学习流程")
    
    print("""
    这个演示展示了异步学习的核心特性：
    1. 提交任务后立即返回（不等待审批）
    2. 学习器持续处理任务队列
    3. 治理决策通过异步事件处理
    """)
    
    # 创建异步学习器池
    pool = AsyncLearnerPool(
        num_learners=2,  # 2 个学习器
        auto_continue=True,  # 自动继续学习
        max_chain_depth=3,  # 最大链深度
    )
    
    # 启动
    pool.start()
    print("\n🚀 异步学习器池已启动")
    
    # 提交多个任务
    print("\n📝 提交学习任务...")
    task_ids = []
    goals = [
        "学习机器学习基础概念",
        "探索深度学习架构",
        "研究自然语言处理技术",
        "分析计算机视觉应用",
    ]
    
    for goal in goals:
        task_id = pool.submit_task(
            goal=goal,
            domain="ai",
            priority="normal",
        )
        task_ids.append(task_id)
        print(f"  ✓ 任务已提交: {task_id} - {goal[:30]}...")
    
    # 等待一段时间让学习器处理
    print("\n⏳ 等待学习器处理任务...")
    time.sleep(3)
    
    # 打印统计
    print_statistics(pool)
    
    # 获取已提交的 LU
    pending_lus = pool.pending_tracker.get_all_pending()
    lu_ids = [p.lu_id for p in pending_lus]
    
    if lu_ids:
        print(f"\n📦 已提交的 LU: {len(lu_ids)} 个")
        for lu in pending_lus:
            print(f"  - {lu.lu_id} (等待时间: {lu.get_wait_time().total_seconds():.1f}秒)")
        
        # 在后台模拟治理决策
        print("\n🔄 启动治理决策模拟...")
        governance_thread = threading.Thread(
            target=simulate_governance_decisions,
            args=(pool, lu_ids),
            daemon=True,
        )
        governance_thread.start()
        
        # 等待治理决策处理
        time.sleep(5)
    
    # 最终统计
    print_separator("最终统计")
    print_statistics(pool)
    
    # 关闭
    pool.shutdown()
    print("\n✅ 异步学习器池已关闭")


def demo_fire_and_forget():
    """演示 Fire-and-Forget 模式"""
    print_separator("演示 2: Fire-and-Forget 模式")
    
    print("""
    这个演示展示了 Fire-and-Forget 的核心优势：
    - 学习器提交 LU 后立即返回
    - 不等待人工审批（可能需要数小时）
    - 学习系统持续高效运行
    """)
    
    pool = AsyncLearnerPool(
        num_learners=3,
        auto_continue=True,
    )
    pool.start()
    
    # 快速提交大量任务
    print("\n📝 快速提交 10 个任务...")
    start_time = time.time()
    
    for i in range(10):
        pool.submit_task(
            goal=f"学习任务 {i+1}: 探索知识领域 {chr(65+i)}",
            domain="general",
            priority="normal" if i % 2 == 0 else "high",
        )
    
    submit_time = time.time() - start_time
    print(f"  ✓ 10 个任务提交完成，耗时: {submit_time:.3f}秒")
    
    # 等待处理
    print("\n⏳ 等待学习器处理...")
    time.sleep(5)
    
    # 统计
    stats = pool.get_statistics()
    print(f"\n📊 处理结果:")
    print(f"  已提交 LU: {stats['coordinator']['pending_tracker']['total_submitted']}")
    print(f"  队列剩余: {stats['queue_size']}")
    
    # 关键点：学习器没有被阻塞
    print("\n💡 关键点: 学习器在等待审批期间持续处理新任务")
    print("   人工审批可能需要数小时，但学习系统不会被阻塞！")
    
    pool.shutdown()


def demo_timeout_handling():
    """演示超时处理"""
    print_separator("演示 3: 超时处理机制")
    
    print("""
    这个演示展示了超时处理机制：
    - 自动检测长时间未响应的 LU
    - 可配置的超时时间
    - 自动重试或升级处理
    """)
    
    from datetime import timedelta
    
    # 创建带有短超时的追踪器（演示用）
    tracker = PendingLUTracker(
        auto_classify_timeout=timedelta(seconds=2),  # 2秒超时（演示用）
        human_review_timeout=timedelta(seconds=5),
    )
    
    pool = AsyncLearnerPool(
        num_learners=1,
        pending_tracker=tracker,
    )
    
    # 注册超时回调
    def on_timeout(pending):
        print(f"\n⚠️ 超时检测: LU {pending.lu_id}")
        print(f"   等待时间: {pending.get_wait_time().total_seconds():.1f}秒")
        print(f"   重试次数: {pending.retry_count}")
    
    pool.coordinator.register_timeout_callback(on_timeout)
    
    pool.start()
    
    # 提交任务
    print("\n📝 提交任务...")
    pool.submit_task(goal="测试超时处理", domain="test")
    
    # 等待超时
    print("\n⏳ 等待超时检测（约 3 秒）...")
    time.sleep(4)
    
    # 统计
    stats = pool.pending_tracker.get_statistics()
    print(f"\n📊 超时统计:")
    print(f"  总超时: {stats['total_timeout']}")
    print(f"  当前待处理: {stats['total_pending']}")
    
    pool.shutdown()


def demo_chain_learning():
    """演示链式学习的异步处理"""
    print_separator("演示 4: 链式学习的异步处理")
    
    print("""
    这个演示展示了链式学习如何与异步模型配合：
    - 父 LU 审批通过后自动触发子学习
    - 链深度由治理系统控制
    - 整个过程不阻塞学习器
    """)
    
    pool = AsyncLearnerPool(
        num_learners=2,
        auto_continue=True,
        max_chain_depth=3,
    )
    pool.start()
    
    # 提交初始任务
    print("\n📝 提交初始学习任务...")
    task_id = pool.submit_task(
        goal="学习深度学习基础",
        domain="ai",
    )
    
    # 等待处理
    time.sleep(2)
    
    # 获取生成的 LU
    pending_lus = pool.pending_tracker.get_all_pending()
    if pending_lus:
        lu_id = pending_lus[0].lu_id
        
        # 模拟审批通过，触发链式学习
        print(f"\n✅ 审批 LU {lu_id}，触发链式学习...")
        pool.on_governance_decision(
            lu_id=lu_id,
            old_status="pending",
            new_status="approved",
            decision="continue",
            decision_params={
                "new_goal": "深入学习神经网络架构",
                "chain_depth": 0,
                "exploration_direction": "深度优先",
            },
        )
        
        # 等待链式学习
        time.sleep(3)
        
        # 检查是否有新的 LU
        new_pending = pool.pending_tracker.get_all_pending()
        print(f"\n📦 链式学习产生的新 LU: {len(new_pending)} 个")
        
        # 继续审批
        if new_pending:
            for p in new_pending:
                print(f"\n✅ 继续审批 LU {p.lu_id}...")
                pool.on_governance_decision(
                    lu_id=p.lu_id,
                    old_status="pending",
                    new_status="approved",
                    decision="continue",
                    decision_params={
                        "new_goal": "更深入的探索",
                        "chain_depth": 1,
                    },
                )
            
            time.sleep(3)
    
    # 最终统计
    stats = pool.get_statistics()
    print(f"\n📊 链式学习统计:")
    print(f"  自动提交的任务: {stats['coordinator']['tasks_auto_submitted']}")
    print(f"  总提交 LU: {stats['coordinator']['pending_tracker']['total_submitted']}")
    
    pool.shutdown()


def main():
    """主函数"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     异步学习系统演示                                  ║
║                                                                      ║
║  解决核心问题: Learner 如何处理 LU 状态通知                          ║
║                                                                      ║
║  关键设计:                                                           ║
║  1. 提交即忘记 (Fire-and-Forget)                                     ║
║  2. 事件驱动调度                                                     ║
║  3. 不阻塞等待人工审批                                               ║
║  4. 超时检测和处理                                                   ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 演示 1: 基本异步学习
        demo_basic_async_learning()
        
        print("\n" + "="*70)
        input("按 Enter 继续下一个演示...")
        
        # 演示 2: Fire-and-Forget
        demo_fire_and_forget()
        
        print("\n" + "="*70)
        input("按 Enter 继续下一个演示...")
        
        # 演示 3: 超时处理
        demo_timeout_handling()
        
        print("\n" + "="*70)
        input("按 Enter 继续下一个演示...")
        
        # 演示 4: 链式学习
        demo_chain_learning()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 演示被中断")
    
    print_separator("演示完成")
    print("""
    总结:
    
    1. 非阻塞设计: Learner 提交 LU 后立即返回，不等待审批
    
    2. 事件驱动: 治理决策通过异步事件通知学习系统
    
    3. 高效利用: 学习器在等待审批期间持续处理新任务
    
    4. 超时处理: 自动检测和处理长时间未响应的 LU
    
    5. 链式学习: 审批通过后自动触发后续学习
    
    这个设计解决了人工审批带来的不确定性问题，
    确保学习系统高效运行，不会被状态同步瘫痪。
    """)


if __name__ == "__main__":
    main()

