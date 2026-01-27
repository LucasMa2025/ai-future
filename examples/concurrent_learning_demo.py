"""
并发学习演示

演示如何使用 LearnerPool 进行多线程并发学习。

运行方式：
    python examples/concurrent_learning_demo.py
"""
import sys
import os
import time
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
)

from self_learning import (
    # 并发学习
    LearnerPool,
    LearningCoordinator,
    TaskPriority,
    TaskStatus,
    
    # 状态管理
    LUStateManager,
    LUStatus,
    LUDecision,
    SharedLearningUnit,
)


def demo_basic_concurrent_learning():
    """演示基本的并发学习"""
    print("\n" + "="*80)
    print("演示 1: 基本并发学习")
    print("="*80)
    
    # 创建状态管理器
    state_manager = LUStateManager()
    
    # 创建学习器池（4 个学习线程）
    pool = LearnerPool(
        num_learners=4,
        state_manager=state_manager,
    )
    
    # 启动
    pool.start()
    print(f"学习器池已启动，共 {pool.num_learners} 个学习线程")
    
    try:
        # 提交多个学习任务
        tasks = []
        
        task1 = pool.submit_task(
            goal="学习金融风险管理基础",
            domain="financial",
            priority=TaskPriority.HIGH,
        )
        tasks.append(task1)
        print(f"任务 1 已提交: {task1.task_id}")
        
        task2 = pool.submit_task(
            goal="学习客户服务最佳实践",
            domain="customer_service",
            priority=TaskPriority.NORMAL,
        )
        tasks.append(task2)
        print(f"任务 2 已提交: {task2.task_id}")
        
        task3 = pool.submit_task(
            goal="学习数据分析方法",
            domain="data_science",
            priority=TaskPriority.NORMAL,
        )
        tasks.append(task3)
        print(f"任务 3 已提交: {task3.task_id}")
        
        # 等待任务完成
        print("\n等待任务完成...")
        for task in tasks:
            result = pool.wait_for_task(task.task_id, timeout=30)
            if result:
                print(f"  任务 {task.task_id} 完成: LU {result.id}")
            else:
                print(f"  任务 {task.task_id} 失败或超时")
        
        # 打印统计
        stats = pool.get_statistics()
        print("\n统计信息:")
        print(f"  任务总数: {stats['tasks']['total']}")
        print(f"  已完成: {stats['tasks']['completed']}")
        print(f"  等待审批: {stats['tasks']['waiting_approval']}")
        print(f"  失败: {stats['tasks']['failed']}")
        
    finally:
        # 关闭
        pool.shutdown()
        print("\n学习器池已关闭")


def demo_state_notification():
    """演示状态通知机制"""
    print("\n" + "="*80)
    print("演示 2: 状态通知机制")
    print("="*80)
    
    # 创建状态管理器
    state_manager = LUStateManager()
    
    # 订阅状态变更
    received_changes = []
    
    def on_state_change(change):
        received_changes.append(change)
        print(f"  [通知] LU {change.lu_id}: {change.old_status} -> {change.new_status}")
        if change.decision:
            print(f"         决策: {change.decision.value}")
    
    state_manager.subscribe("demo_subscriber", on_state_change)
    
    # 创建一个 LU
    lu = SharedLearningUnit(
        id="lu_demo_001",
        title="演示 Learning Unit",
        learning_goal="演示状态通知",
    )
    state_manager.register_lu(lu)
    print(f"LU 已注册: {lu.id}")
    
    # 模拟治理系统更新状态
    print("\n模拟治理系统更新状态:")
    
    # 1. 自动分类
    state_manager.update_status(
        lu_id=lu.id,
        new_status=LUStatus.AUTO_CLASSIFIED,
        reason="自动分类完成",
        changed_by="audit_system",
    )
    
    time.sleep(0.1)
    
    # 2. 人工审核
    state_manager.update_status(
        lu_id=lu.id,
        new_status=LUStatus.HUMAN_REVIEW,
        reason="需要人工审核",
        changed_by="audit_system",
    )
    
    time.sleep(0.1)
    
    # 3. 审批通过，决定继续学习
    state_manager.update_status(
        lu_id=lu.id,
        new_status=LUStatus.APPROVED,
        decision=LUDecision.CONTINUE,
        reason="审批通过",
        changed_by="admin",
        continue_params={
            "new_goal": "深入学习相关主题",
            "exploration_direction": "深度探索",
        },
    )
    
    print(f"\n共收到 {len(received_changes)} 个状态变更通知")
    
    # 取消订阅
    state_manager.unsubscribe("demo_subscriber")


def demo_chain_learning_with_pool():
    """演示使用线程池的链式学习"""
    print("\n" + "="*80)
    print("演示 3: 链式学习（使用线程池）")
    print("="*80)
    
    # 创建状态管理器
    state_manager = LUStateManager()
    
    # 创建学习器池
    pool = LearnerPool(
        num_learners=2,
        state_manager=state_manager,
    )
    
    # 创建协调器（自动处理继续学习）
    coordinator = LearningCoordinator(
        learner_pool=pool,
        state_manager=state_manager,
        max_chain_depth=3,
        auto_continue=False,  # 手动控制
    )
    
    pool.start()
    
    try:
        # 第一步：从零开始学习
        print("\n步骤 1: 从零开始学习")
        task1 = pool.submit_task(
            goal="学习基础风险管理概念",
            domain="financial",
        )
        
        lu1 = pool.wait_for_task(task1.task_id, timeout=30)
        if not lu1:
            print("任务 1 失败")
            return
        
        print(f"  LU 1 创建: {lu1.id}")
        print(f"  标题: {lu1.title}")
        print(f"  链深度: {lu1.chain_depth}")
        
        # 模拟审批通过并内化
        state_manager.update_status(
            lu_id=lu1.id,
            new_status=LUStatus.APPROVED,
            changed_by="admin",
        )
        lu1.is_internalized = True
        state_manager.update_status(
            lu_id=lu1.id,
            new_status=LUStatus.INTERNALIZED,
            changed_by="system",
        )
        
        # 第二步：继续学习
        print("\n步骤 2: 继续学习")
        task2 = pool.submit_task(
            goal="深入学习信用风险评估",
            domain="financial",
            parent_lu_id=lu1.id,
            exploration_direction="信用风险",
        )
        
        lu2 = pool.wait_for_task(task2.task_id, timeout=30)
        if not lu2:
            print("任务 2 失败")
            return
        
        print(f"  LU 2 创建: {lu2.id}")
        print(f"  标题: {lu2.title}")
        print(f"  父 LU: {lu2.parent_lu_id}")
        print(f"  链深度: {lu2.chain_depth}")
        
        # 打印学习链
        print("\n学习链:")
        print(f"  {lu1.id} (depth=0)")
        print(f"    └── {lu2.id} (depth=1)")
        
    finally:
        coordinator.cleanup()
        pool.shutdown()


def demo_governance_integration():
    """演示与治理系统的集成"""
    print("\n" + "="*80)
    print("演示 4: 治理系统集成")
    print("="*80)
    
    # 创建状态管理器
    state_manager = LUStateManager()
    
    # 创建学习器池
    pool = LearnerPool(
        num_learners=2,
        state_manager=state_manager,
    )
    
    pool.start()
    
    try:
        # 提交任务
        task = pool.submit_task(
            goal="学习需要治理审批的内容",
            domain="sensitive",
            priority=TaskPriority.HIGH,
        )
        
        # 等待任务完成（生成 LU）
        lu = pool.wait_for_task(task.task_id, timeout=30)
        if not lu:
            print("任务失败")
            return
        
        print(f"LU 已生成: {lu.id}")
        print(f"当前状态: {lu.status.value}")
        
        # 模拟治理系统的三种决策
        print("\n模拟治理决策:")
        
        # 决策 1: 审批通过，继续学习
        print("\n  决策 1: APPROVED + CONTINUE")
        state_manager.update_status(
            lu_id=lu.id,
            new_status=LUStatus.APPROVED,
            decision=LUDecision.CONTINUE,
            reason="审批通过，可继续学习",
            changed_by="governance_committee",
            continue_params={
                "new_goal": "深入探索",
            },
        )
        
        # 决策 2: 拒绝，开始新学习
        print("\n  决策 2: REJECTED + NEW_LEARNING")
        # 创建另一个 LU 来演示
        lu2 = SharedLearningUnit(
            id="lu_rejected_demo",
            title="被拒绝的 LU",
            learning_goal="演示拒绝流程",
        )
        state_manager.register_lu(lu2)
        
        state_manager.update_status(
            lu_id=lu2.id,
            new_status=LUStatus.REJECTED,
            decision=LUDecision.NEW_LEARNING,
            reason="知识不准确，需要重新学习",
            changed_by="governance_committee",
            continue_params={
                "new_goal": "重新学习正确的内容",
            },
        )
        
        # 决策 3: 调整策略
        print("\n  决策 3: CORRECTED + ADJUST")
        lu3 = SharedLearningUnit(
            id="lu_corrected_demo",
            title="需要调整的 LU",
            learning_goal="演示调整流程",
        )
        state_manager.register_lu(lu3)
        
        state_manager.update_status(
            lu_id=lu3.id,
            new_status=LUStatus.CORRECTED,
            decision=LUDecision.ADJUST,
            reason="需要调整学习策略",
            changed_by="governance_committee",
            adjustment_params={
                "adjusted_goal": "调整后的学习目标",
                "exploration_direction": "新的探索方向",
                "focus_areas": ["重点领域1", "重点领域2"],
            },
        )
        
        # 打印状态管理器统计
        print("\n状态管理器统计:")
        stats = state_manager.get_statistics()
        print(f"  总 LU 数: {stats['total_lus']}")
        print(f"  按状态: {stats['by_status']}")
        print(f"  总变更数: {stats['total_changes']}")
        
    finally:
        pool.shutdown()


def demo_pool_management():
    """演示线程池管理"""
    print("\n" + "="*80)
    print("演示 5: 线程池管理")
    print("="*80)
    
    state_manager = LUStateManager()
    
    # 创建不同大小的线程池
    pool = LearnerPool(
        num_learners=3,
        state_manager=state_manager,
        max_queue_size=50,
    )
    
    pool.start()
    
    try:
        # 提交多个任务
        print("\n提交 5 个任务到 3 个学习器:")
        tasks = []
        for i in range(5):
            task = pool.submit_task(
                goal=f"学习任务 {i+1}",
                domain="test",
                priority=TaskPriority.NORMAL,
            )
            tasks.append(task)
            print(f"  任务 {i+1} 已提交: {task.task_id}")
        
        # 暂停所有学习器
        print("\n暂停所有学习器...")
        pool.pause_all()
        time.sleep(1)
        
        # 查看状态
        stats = pool.get_statistics()
        print(f"队列大小: {stats['queue_size']}")
        print("学习器状态:")
        for learner_id, learner_stats in stats['learners'].items():
            print(f"  {learner_id}: {learner_stats['status']}")
        
        # 恢复
        print("\n恢复所有学习器...")
        pool.resume_all()
        
        # 等待任务完成
        print("\n等待任务完成...")
        for task in tasks:
            result = pool.wait_for_task(task.task_id, timeout=30)
            status = pool.get_task(task.task_id).status.value
            print(f"  任务 {task.task_id}: {status}")
        
        # 最终统计
        print("\n最终统计:")
        stats = pool.get_statistics()
        print(f"  任务总数: {stats['tasks']['total']}")
        print(f"  已完成: {stats['tasks']['completed']}")
        print(f"  等待审批: {stats['tasks']['waiting_approval']}")
        
        for learner_id, learner_stats in stats['learners'].items():
            print(f"\n  {learner_id}:")
            print(f"    完成任务数: {learner_stats['tasks_completed']}")
            print(f"    总学习时间: {learner_stats['total_learning_time']:.2f}s")
        
    finally:
        pool.shutdown()


def main():
    """主函数"""
    print("="*80)
    print("并发学习系统演示")
    print("="*80)
    print("""
本演示展示了并发学习系统的核心功能：

1. 基本并发学习 - 使用 LearnerPool 并发执行多个学习任务
2. 状态通知机制 - 治理系统通过状态变更通知自学习系统
3. 链式学习 - 使用线程池进行链式学习
4. 治理系统集成 - 三种治理决策的处理
5. 线程池管理 - 暂停、恢复、统计

核心设计原则：
- Learning Unit 本身带有状态
- 状态由 NLGSM 治理系统管理
- 自学习系统通过异步通知获取状态变更
- 可配置的学习器数量，支持生产环境
""")
    
    # 运行演示
    demo_basic_concurrent_learning()
    demo_state_notification()
    demo_chain_learning_with_pool()
    demo_governance_integration()
    demo_pool_management()
    
    print("\n" + "="*80)
    print("演示完成")
    print("="*80)


if __name__ == "__main__":
    main()

