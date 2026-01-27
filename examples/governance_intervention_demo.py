"""
NLGSM 治理干预演示

本示例演示治理系统如何干预自学习过程：
1. 检查点 -> 治理审查 -> 继续/修改/终止
2. Learning Unit -> 审计系统 -> 反馈处理
3. 实时干预：暂停、修改目标、回滚

完整流程：
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  自学习系统  │────▶│  Checkpoint  │────▶│  治理审查   │
│ (LLM 探索)  │     │  (自动创建)  │     │  (人工决策) │
└─────────────┘     └──────────────┘     └─────────────┘
      ▲                                        │
      │                                        ▼
      │                    ┌──────────────────────────────┐
      │                    │ 决策: continue/modify/terminate │
      │                    └──────────────────────────────┘
      │                                        │
      └────────────────────────────────────────┘
                        (修改 Scope/Goal)
                        
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Learning    │────▶│  审计系统    │────▶│  审计反馈   │
│ Unit 提交   │     │  (LU 审核)   │     │            │
└─────────────┘     └──────────────┘     └─────────────┘
                                               │
                                               ▼
                    ┌──────────────────────────────────┐
                    │ 决策: approve/reject/needs_revision │
                    └──────────────────────────────────┘
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from self_learning import (
    AutonomousExplorer,
    KnowledgeGenerator,
    CheckpointManager,
    NLLearningUnitBuilder,
    GovernanceInterface,
    InterventionType,
    InterventionPriority,
    NLLevel,
    LearningScope,
    KernelFactory,
)
from core.enums import AuditStatus, RiskLevel


def print_section(title: str):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def demo_checkpoint_review():
    """演示检查点审查流程"""
    print_section("演示1: 检查点审查流程")
    
    # 1. 创建组件
    print("\n1. 创建自学习组件...")
    explorer = AutonomousExplorer.with_adapter("mock", max_depth=3)
    generator = KnowledgeGenerator.with_adapter("mock")
    checkpoint_manager = CheckpointManager(
        storage_dir="./data/demo_checkpoints",
        knowledge_count_threshold=2  # 每2个知识点创建检查点
    )
    kernel = KernelFactory.create_with_adapter("mock")
    
    # 2. 创建治理接口
    print("2. 创建治理接口...")
    
    def on_governance_event(event_type: str, data: dict):
        """治理事件回调"""
        print(f"\n[治理事件] {event_type}")
        if event_type == "checkpoint_created":
            print(f"  检查点: {data.get('checkpoint_id')}")
        elif event_type == "intervention_issued":
            print(f"  干预类型: {data.get('type')}")
    
    governance = GovernanceInterface(
        nl_kernel=kernel,
        checkpoint_manager=checkpoint_manager,
        notification_callback=on_governance_event,
    )
    
    # 3. 创建 Learning Unit Builder
    print("3. 创建 NL Learning Unit Builder...")
    builder = NLLearningUnitBuilder(
        explorer=explorer,
        knowledge_generator=generator,
        checkpoint_manager=checkpoint_manager,
        nl_kernel=kernel,
    )
    
    # 4. 开始学习会话
    print("\n4. 开始学习会话...")
    governance.current_session_id = builder.start_learning_session()
    governance.current_goal = "学习 Python 装饰器"
    governance.current_scope = builder.current_scope
    
    # 5. 模拟检查点创建
    print("\n5. 模拟检查点创建...")
    checkpoint = checkpoint_manager.create_checkpoint(
        exploration_data={
            "goal": "学习 Python 装饰器",
            "progress": 50,
            "findings": ["装饰器语法", "常见用例"],
        },
        reason="知识阈值触发"
    )
    
    # 6. 治理系统审查检查点
    print("\n6. 治理系统审查检查点...")
    
    # 场景 A: 继续学习
    print("\n场景 A: 决策 - 继续学习")
    review_a = governance.review_checkpoint(
        checkpoint_id=checkpoint["checkpoint_id"],
        decision="continue",
        comments="学习进展正常，继续",
        reviewed_by="governance_admin",
    )
    print(f"  审查结果: {review_a.decision}")
    
    # 场景 B: 修改学习范围
    print("\n场景 B: 决策 - 修改学习范围")
    new_scope = LearningScope(
        scope_id="modified_scope",
        max_level=NLLevel.PARAMETER,  # 降低最大层级
        allowed_levels=[NLLevel.PARAMETER],
        created_by="governance_admin",
    )
    review_b = governance.review_checkpoint(
        checkpoint_id=checkpoint["checkpoint_id"] + "_b",
        decision="modify",
        comments="风险过高，限制到 PARAMETER 层级",
        modified_scope=new_scope,
        reviewed_by="governance_admin",
    )
    print(f"  审查结果: {review_b.decision}")
    print(f"  修改后 Scope: {governance.current_scope.max_level.name if governance.current_scope else 'N/A'}")
    
    # 场景 C: 终止学习
    print("\n场景 C: 决策 - 终止学习")
    review_c = governance.review_checkpoint(
        checkpoint_id=checkpoint["checkpoint_id"] + "_c",
        decision="terminate",
        comments="发现潜在风险，终止此次学习",
        reviewed_by="governance_admin",
    )
    print(f"  审查结果: {review_c.decision}")
    
    # 7. 查看干预历史
    print("\n7. 干预历史:")
    for intervention in governance.get_intervention_history():
        print(f"  - {intervention['type']}: {intervention['reason'][:30]}...")


def demo_realtime_intervention():
    """演示实时干预"""
    print_section("演示2: 实时干预流程")
    
    # 创建组件
    kernel = KernelFactory.create_with_adapter("mock")
    checkpoint_manager = CheckpointManager(storage_dir="./data/demo_checkpoints")
    
    governance = GovernanceInterface(
        nl_kernel=kernel,
        checkpoint_manager=checkpoint_manager,
    )
    
    governance.current_session_id = "session_demo"
    governance.current_goal = "学习机器学习算法"
    
    # 解冻内核开始学习
    kernel.unfreeze()
    
    # 1. 暂停学习
    print("\n1. 暂停学习...")
    intervention1 = governance.issue_intervention(
        intervention_type=InterventionType.PAUSE,
        reason="需要人工审查当前进展",
        issued_by="governance_system",
        priority=InterventionPriority.HIGH,
    )
    print(f"  结果: {intervention1.result}")
    print(f"  当前状态: paused={governance.is_paused}")
    
    # 2. 修改学习目标
    print("\n2. 修改学习目标...")
    intervention2 = governance.issue_intervention(
        intervention_type=InterventionType.MODIFY_GOAL,
        reason="调整学习方向",
        issued_by="governance_system",
        priority=InterventionPriority.NORMAL,
        params={"new_goal": "学习监督学习算法（排除深度学习）"},
    )
    print(f"  新目标: {governance.current_goal}")
    
    # 3. 恢复学习
    print("\n3. 恢复学习...")
    intervention3 = governance.issue_intervention(
        intervention_type=InterventionType.RESUME,
        reason="审查完成，继续学习",
        issued_by="governance_system",
        priority=InterventionPriority.HIGH,
    )
    print(f"  当前状态: paused={governance.is_paused}")
    
    # 4. 请求创建检查点
    print("\n4. 请求创建检查点...")
    intervention4 = governance.issue_intervention(
        intervention_type=InterventionType.REQUEST_CHECKPOINT,
        reason="阶段性保存",
        issued_by="governance_system",
    )
    print(f"  检查点: {intervention4.result.get('checkpoint_id', 'N/A')}")
    
    # 5. 回滚到检查点
    print("\n5. 回滚到检查点...")
    if intervention4.result.get('checkpoint_id'):
        intervention5 = governance.issue_intervention(
            intervention_type=InterventionType.ROLLBACK,
            reason="发现问题，需要回滚",
            issued_by="governance_system",
            priority=InterventionPriority.CRITICAL,
            target_checkpoint_id=intervention4.result['checkpoint_id'],
        )
        print(f"  回滚结果: {intervention5.result}")
    
    # 6. 终止学习
    print("\n6. 终止学习...")
    intervention6 = governance.issue_intervention(
        intervention_type=InterventionType.TERMINATE,
        reason="本次学习目标已达成",
        issued_by="governance_system",
        priority=InterventionPriority.CRITICAL,
    )
    print(f"  终止结果: {intervention6.result}")


def demo_lu_audit_feedback():
    """演示 Learning Unit 审计反馈"""
    print_section("演示3: Learning Unit 审计反馈")
    
    # 创建组件
    kernel = KernelFactory.create_with_adapter("mock")
    checkpoint_manager = CheckpointManager(storage_dir="./data/demo_checkpoints")
    
    governance = GovernanceInterface(
        nl_kernel=kernel,
        checkpoint_manager=checkpoint_manager,
    )
    
    # 1. 审计通过
    print("\n1. 审计通过场景...")
    feedback1 = governance.receive_audit_feedback(
        learning_unit_id="lu_001",
        decision=AuditStatus.APPROVED,
        risk_level=RiskLevel.LOW,
        comments="知识质量良好，已通过审计",
        audited_by="senior_engineer",
    )
    print(f"  决策: {feedback1.decision.value}")
    print(f"  风险等级: {feedback1.risk_level.value}")
    
    # 2. 审计拒绝
    print("\n2. 审计拒绝场景...")
    feedback2 = governance.receive_audit_feedback(
        learning_unit_id="lu_002",
        decision=AuditStatus.REJECTED,
        comments="知识来源不可靠",
        rejection_reason="推理链路存在逻辑漏洞",
        relearn_guidance={
            "new_goal": "重新学习，增加验证步骤",
            "constraints": ["必须包含多源验证"],
            "max_level": "PARAMETER",
        },
        audited_by="governance_committee",
    )
    print(f"  决策: {feedback2.decision.value}")
    print(f"  拒绝原因: {feedback2.rejection_reason}")
    print(f"  重新学习指导: {feedback2.relearn_guidance}")
    
    # 3. 需要修改
    print("\n3. 需要修改场景...")
    feedback3 = governance.receive_audit_feedback(
        learning_unit_id="lu_003",
        decision=AuditStatus.NEEDS_REVISION,
        comments="部分内容需要修改",
        required_modifications=[
            {"field": "confidence", "reason": "置信度过高，需要降低"},
            {"field": "constraints", "reason": "需要添加使用限制条件"},
        ],
        audited_by="ml_engineer",
    )
    print(f"  决策: {feedback3.decision.value}")
    print(f"  修改要求: {feedback3.required_modifications}")
    
    # 4. 查看所有反馈
    print("\n4. 所有审计反馈:")
    for feedback in governance.get_audit_feedbacks():
        print(f"  - {feedback['learning_unit_id']}: {feedback['decision']}")


def demo_full_workflow():
    """演示完整工作流"""
    print_section("演示4: 完整工作流")
    
    print("""
完整的 NLGSM 治理干预流程:

1. 学习阶段:
   ┌─────────────────────────────────────────────────────────┐
   │  治理系统提供 LearningScope (定义允许的层级和预算)      │
   │                          ↓                              │
   │  自学习系统在 Scope 控制下执行 LLM 探索                 │
   │                          ↓                              │
   │  每达到阈值自动创建 Checkpoint                          │
   │                          ↓                              │
   │  治理系统收到 Checkpoint 通知                           │
   └─────────────────────────────────────────────────────────┘

2. 检查点审查:
   ┌─────────────────────────────────────────────────────────┐
   │  治理管理员查看 Checkpoint 内容                         │
   │                          ↓                              │
   │  做出决策:                                              │
   │    - continue: 继续学习                                 │
   │    - pause: 暂停等待进一步审查                          │
   │    - modify: 修改 Scope/Goal 后继续                     │
   │    - terminate: 终止此次学习                            │
   └─────────────────────────────────────────────────────────┘

3. Learning Unit 审计:
   ┌─────────────────────────────────────────────────────────┐
   │  自学习完成后生成 Learning Unit                         │
   │                          ↓                              │
   │  提交到审计系统 (LU 状态: PENDING)                      │
   │                          ↓                              │
   │  审计人员根据风险等级进行审核:                          │
   │    - LOW: 1人审批                                       │
   │    - MEDIUM: 2人审批                                    │
   │    - HIGH: 高级工程师审批                               │
   │    - CRITICAL: 治理委员会审批                           │
   │                          ↓                              │
   │  审计决策:                                              │
   │    - APPROVED: 可以进入 Release 流程                    │
   │    - REJECTED: 拒绝，可能触发重新学习                   │
   │    - NEEDS_REVISION: 需要修改后重新提交                 │
   └─────────────────────────────────────────────────────────┘

4. 干预触发条件:
   ┌─────────────────────────────────────────────────────────┐
   │  自动触发:                                              │
   │    - 检测到异常模式                                     │
   │    - 超出预算限制                                       │
   │    - 风险评估超标                                       │
   │                                                         │
   │  人工触发:                                              │
   │    - 管理员发起暂停/终止                                │
   │    - 审计人员请求修改                                   │
   │    - 监控系统发出警报                                   │
   └─────────────────────────────────────────────────────────┘

5. 回滚机制:
   ┌─────────────────────────────────────────────────────────┐
   │  触发条件:                                              │
   │    - 审计拒绝且需要回滚                                 │
   │    - 发现学习过程中的错误                               │
   │    - 人工请求回滚                                       │
   │                                                         │
   │  回滚操作:                                              │
   │    1. 冻结 NL 内核                                      │
   │    2. 保存当前状态快照                                  │
   │    3. 恢复到目标 Checkpoint 状态                        │
   │    4. 可选：使用修改后的 Scope/Goal 重新学习            │
   └─────────────────────────────────────────────────────────┘
    """)
    
    # 查看当前状态
    print("\n[状态查询示例]")
    kernel = KernelFactory.create_with_adapter("mock")
    checkpoint_manager = CheckpointManager(storage_dir="./data/demo_checkpoints")
    governance = GovernanceInterface(kernel, checkpoint_manager)
    
    status = governance.get_learning_status()
    print(f"当前状态: {json.dumps(status, indent=2, ensure_ascii=False, default=str)}")


import json

def main():
    """主函数"""
    print("="*60)
    print("  NLGSM 治理干预演示")
    print("="*60)
    
    # 演示1: 检查点审查
    demo_checkpoint_review()
    
    # 演示2: 实时干预
    demo_realtime_intervention()
    
    # 演示3: LU 审计反馈
    demo_lu_audit_feedback()
    
    # 演示4: 完整工作流
    demo_full_workflow()
    
    print("\n" + "="*60)
    print("  演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()

