"""
诊断与恢复服务

实现 NLGSM 论文中的诊断-恢复流程：
1. SOP (Standard Operating Procedure) 管理
2. SLA (Service Level Agreement) 监控
3. 诊断报告生成
4. 恢复计划创建和执行
"""
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from uuid import UUID
import uuid
import logging
from enum import Enum
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..core.enums import NLGSMState, RiskLevel
from ..core.exceptions import NotFoundError, BusinessError


logger = logging.getLogger(__name__)


class DiagnosisPhase(str, Enum):
    """诊断阶段"""
    COLLECTING = "collecting"      # 收集数据
    ANALYZING = "analyzing"        # 分析中
    IDENTIFIED = "identified"      # 已识别问题
    REPORT_READY = "report_ready"  # 报告就绪


class RecoveryPlanStatus(str, Enum):
    """恢复计划状态"""
    DRAFT = "draft"               # 草稿
    PENDING_APPROVAL = "pending_approval"  # 待审批
    APPROVED = "approved"         # 已批准
    EXECUTING = "executing"       # 执行中
    COMPLETED = "completed"       # 已完成
    FAILED = "failed"            # 失败
    REJECTED = "rejected"        # 被拒绝


class SOPStep(str, Enum):
    """SOP 步骤类型"""
    MANUAL = "manual"             # 人工操作
    AUTOMATED = "automated"       # 自动化操作
    VERIFICATION = "verification"  # 验证步骤
    APPROVAL = "approval"         # 审批步骤


@dataclass
class DiagnosticData:
    """诊断数据"""
    id: UUID = field(default_factory=uuid.uuid4)
    
    # 系统状态快照
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    # 异常信息
    anomaly_signals: List[Dict[str, Any]] = field(default_factory=list)
    
    # 日志片段
    log_excerpts: List[str] = field(default_factory=list)
    
    # 指标快照
    metric_snapshots: Dict[str, float] = field(default_factory=dict)
    
    # 时间线
    event_timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # 收集时间
    collected_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DiagnosisReport:
    """诊断报告"""
    id: UUID = field(default_factory=uuid.uuid4)
    
    # 诊断数据
    diagnostic_data: Optional[DiagnosticData] = None
    
    # 问题识别
    root_cause: Optional[str] = None
    contributing_factors: List[str] = field(default_factory=list)
    affected_components: List[str] = field(default_factory=list)
    
    # 影响评估
    severity: RiskLevel = RiskLevel.MEDIUM
    impact_scope: str = ""
    estimated_recovery_time: Optional[timedelta] = None
    
    # 建议
    recommended_actions: List[str] = field(default_factory=list)
    
    # 元数据
    phase: DiagnosisPhase = DiagnosisPhase.COLLECTING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "root_cause": self.root_cause,
            "contributing_factors": self.contributing_factors,
            "affected_components": self.affected_components,
            "severity": self.severity.value,
            "impact_scope": self.impact_scope,
            "estimated_recovery_time": str(self.estimated_recovery_time) if self.estimated_recovery_time else None,
            "recommended_actions": self.recommended_actions,
            "phase": self.phase.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


@dataclass
class RecoveryStep:
    """恢复步骤"""
    id: str
    name: str
    description: str
    step_type: SOPStep
    order: int
    
    # 执行信息
    handler: Optional[str] = None  # 处理函数名
    params: Dict[str, Any] = field(default_factory=dict)
    
    # 状态
    status: str = "pending"  # pending, executing, completed, failed, skipped
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # 依赖
    depends_on: List[str] = field(default_factory=list)
    
    # 验证
    verification_required: bool = False
    verified: bool = False
    verified_by: Optional[UUID] = None


@dataclass
class RecoveryPlan:
    """恢复计划"""
    id: UUID = field(default_factory=uuid.uuid4)
    
    # 关联诊断
    diagnosis_report_id: Optional[UUID] = None
    
    # 计划信息
    name: str = ""
    description: str = ""
    
    # 步骤
    steps: List[RecoveryStep] = field(default_factory=list)
    
    # SLA 目标
    target_recovery_time: Optional[timedelta] = None
    actual_recovery_time: Optional[timedelta] = None
    
    # 状态
    status: RecoveryPlanStatus = RecoveryPlanStatus.DRAFT
    current_step_index: int = 0
    
    # 审批
    required_approvers: int = 1
    approvals: List[Dict[str, Any]] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    approved_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "diagnosis_report_id": str(self.diagnosis_report_id) if self.diagnosis_report_id else None,
            "status": self.status.value,
            "steps": [
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "step_type": s.step_type.value,
                    "order": s.order,
                    "status": s.status,
                }
                for s in self.steps
            ],
            "target_recovery_time": str(self.target_recovery_time) if self.target_recovery_time else None,
            "actual_recovery_time": str(self.actual_recovery_time) if self.actual_recovery_time else None,
            "current_step_index": self.current_step_index,
            "created_at": self.created_at.isoformat(),
            "approved_at": self.approved_at.isoformat() if self.approved_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DiagnosisService:
    """
    诊断服务
    
    负责收集诊断数据、分析问题、生成诊断报告
    """
    
    def __init__(self, db: Session):
        self.db = db
        self._reports: Dict[UUID, DiagnosisReport] = {}
        self._diagnostic_collectors: Dict[str, Callable] = {}
        self._analyzers: List[Callable] = []
        
        # 注册默认收集器
        self._register_default_collectors()
    
    def _register_default_collectors(self):
        """注册默认数据收集器"""
        self._diagnostic_collectors["system_state"] = self._collect_system_state
        self._diagnostic_collectors["recent_events"] = self._collect_recent_events
        self._diagnostic_collectors["metrics"] = self._collect_metrics
        self._diagnostic_collectors["logs"] = self._collect_logs
    
    def start_diagnosis(
        self,
        trigger_source: str,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosisReport:
        """
        开始诊断流程
        
        Args:
            trigger_source: 触发源
            initial_context: 初始上下文
            
        Returns:
            诊断报告
        """
        report = DiagnosisReport(
            phase=DiagnosisPhase.COLLECTING,
        )
        
        # 收集诊断数据
        diagnostic_data = self._collect_diagnostic_data(initial_context)
        report.diagnostic_data = diagnostic_data
        
        # 更新阶段
        report.phase = DiagnosisPhase.ANALYZING
        
        # 执行分析
        self._analyze(report)
        
        # 存储报告
        self._reports[report.id] = report
        
        logger.info(f"Diagnosis started: id={report.id}, trigger={trigger_source}")
        
        return report
    
    def _collect_diagnostic_data(
        self,
        context: Optional[Dict[str, Any]] = None,
    ) -> DiagnosticData:
        """收集诊断数据"""
        data = DiagnosticData()
        
        for name, collector in self._diagnostic_collectors.items():
            try:
                result = collector(context)
                if name == "system_state":
                    data.system_state = result
                elif name == "recent_events":
                    data.event_timeline = result
                elif name == "metrics":
                    data.metric_snapshots = result
                elif name == "logs":
                    data.log_excerpts = result
            except Exception as e:
                logger.error(f"Collector {name} failed: {e}")
        
        return data
    
    def _analyze(self, report: DiagnosisReport):
        """分析诊断数据"""
        # 基本分析规则
        data = report.diagnostic_data
        
        if not data:
            report.root_cause = "Unknown: No diagnostic data available"
            report.phase = DiagnosisPhase.IDENTIFIED
            return
        
        # 分析异常信号
        if data.anomaly_signals:
            critical_signals = [s for s in data.anomaly_signals if s.get("severity") == "critical"]
            if critical_signals:
                report.severity = RiskLevel.CRITICAL
                report.root_cause = f"Critical anomaly detected: {critical_signals[0].get('type', 'unknown')}"
        
        # 分析指标
        if data.metric_snapshots:
            # 检查错误率
            error_rate = data.metric_snapshots.get("error_rate", 0)
            if error_rate > 0.5:
                report.contributing_factors.append(f"High error rate: {error_rate:.2%}")
            
            # 检查延迟
            latency = data.metric_snapshots.get("latency_p99", 0)
            if latency > 1000:  # > 1 秒
                report.contributing_factors.append(f"High latency: {latency}ms")
        
        # 生成建议
        self._generate_recommendations(report)
        
        report.phase = DiagnosisPhase.IDENTIFIED
        
        # 运行自定义分析器
        for analyzer in self._analyzers:
            try:
                analyzer(report)
            except Exception as e:
                logger.error(f"Analyzer failed: {e}")
        
        report.phase = DiagnosisPhase.REPORT_READY
        report.completed_at = datetime.utcnow()
    
    def _generate_recommendations(self, report: DiagnosisReport):
        """生成恢复建议"""
        severity = report.severity
        
        if severity == RiskLevel.CRITICAL:
            report.recommended_actions = [
                "立即停止所有学习活动",
                "回滚到最近的稳定检查点",
                "通知治理委员会",
                "收集详细日志进行人工分析",
            ]
            report.estimated_recovery_time = timedelta(hours=4)
        
        elif severity == RiskLevel.HIGH:
            report.recommended_actions = [
                "暂停学习活动",
                "评估回滚必要性",
                "通知运维团队",
            ]
            report.estimated_recovery_time = timedelta(hours=2)
        
        elif severity == RiskLevel.MEDIUM:
            report.recommended_actions = [
                "监控系统状态",
                "准备回滚计划",
            ]
            report.estimated_recovery_time = timedelta(hours=1)
        
        else:
            report.recommended_actions = [
                "记录问题",
                "继续监控",
            ]
            report.estimated_recovery_time = timedelta(minutes=30)
    
    def get_report(self, report_id: UUID) -> Optional[DiagnosisReport]:
        """获取诊断报告"""
        return self._reports.get(report_id)
    
    def list_reports(self, limit: int = 20) -> List[DiagnosisReport]:
        """列出诊断报告"""
        reports = list(self._reports.values())
        return sorted(reports, key=lambda r: r.created_at, reverse=True)[:limit]
    
    # ==================== 数据收集器 ====================
    
    def _collect_system_state(self, context: Optional[Dict]) -> Dict[str, Any]:
        """收集系统状态"""
        # 这里应该从状态机服务获取
        return {
            "current_state": "unknown",
            "last_transition": None,
        }
    
    def _collect_recent_events(self, context: Optional[Dict]) -> List[Dict[str, Any]]:
        """收集最近事件"""
        return []
    
    def _collect_metrics(self, context: Optional[Dict]) -> Dict[str, float]:
        """收集指标"""
        return {}
    
    def _collect_logs(self, context: Optional[Dict]) -> List[str]:
        """收集日志"""
        return []
    
    def register_collector(self, name: str, collector: Callable):
        """注册数据收集器"""
        self._diagnostic_collectors[name] = collector
    
    def register_analyzer(self, analyzer: Callable):
        """注册分析器"""
        self._analyzers.append(analyzer)


class RecoveryService:
    """
    恢复服务
    
    负责创建、审批和执行恢复计划
    """
    
    def __init__(
        self,
        db: Session,
        diagnosis_service: DiagnosisService,
        state_machine_service=None,
    ):
        self.db = db
        self.diagnosis_service = diagnosis_service
        self.state_machine = state_machine_service
        
        self._plans: Dict[UUID, RecoveryPlan] = {}
        self._step_handlers: Dict[str, Callable] = {}
        
        # 注册默认步骤处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认步骤处理器"""
        self._step_handlers["freeze_learning"] = self._handler_freeze_learning
        self._step_handlers["restore_checkpoint"] = self._handler_restore_checkpoint
        self._step_handlers["verify_system"] = self._handler_verify_system
        self._step_handlers["resume_learning"] = self._handler_resume_learning
        self._step_handlers["notify_team"] = self._handler_notify_team
    
    def create_plan(
        self,
        diagnosis_report_id: Optional[UUID] = None,
        name: str = "Recovery Plan",
        description: str = "",
        steps: Optional[List[Dict[str, Any]]] = None,
        target_recovery_time: Optional[timedelta] = None,
    ) -> RecoveryPlan:
        """
        创建恢复计划
        
        Args:
            diagnosis_report_id: 关联的诊断报告ID
            name: 计划名称
            description: 计划描述
            steps: 步骤列表
            target_recovery_time: 目标恢复时间
            
        Returns:
            恢复计划
        """
        plan = RecoveryPlan(
            name=name,
            description=description,
            diagnosis_report_id=diagnosis_report_id,
            target_recovery_time=target_recovery_time,
            status=RecoveryPlanStatus.DRAFT,
        )
        
        # 添加步骤
        if steps:
            for i, step_data in enumerate(steps):
                step = RecoveryStep(
                    id=step_data.get("id", f"step_{i}"),
                    name=step_data.get("name", f"Step {i + 1}"),
                    description=step_data.get("description", ""),
                    step_type=SOPStep(step_data.get("step_type", "automated")),
                    order=i,
                    handler=step_data.get("handler"),
                    params=step_data.get("params", {}),
                    depends_on=step_data.get("depends_on", []),
                    verification_required=step_data.get("verification_required", False),
                )
                plan.steps.append(step)
        
        self._plans[plan.id] = plan
        
        logger.info(f"Recovery plan created: id={plan.id}, name={name}")
        
        return plan
    
    def create_plan_from_diagnosis(
        self,
        diagnosis_report_id: UUID,
    ) -> RecoveryPlan:
        """根据诊断报告自动创建恢复计划"""
        report = self.diagnosis_service.get_report(diagnosis_report_id)
        if not report:
            raise NotFoundError("DiagnosisReport", str(diagnosis_report_id))
        
        # 根据严重程度生成步骤
        steps = self._generate_steps_from_severity(report.severity)
        
        return self.create_plan(
            diagnosis_report_id=diagnosis_report_id,
            name=f"Recovery from {report.root_cause or 'Unknown Issue'}",
            description=f"Auto-generated recovery plan based on diagnosis {diagnosis_report_id}",
            steps=steps,
            target_recovery_time=report.estimated_recovery_time,
        )
    
    def _generate_steps_from_severity(self, severity: RiskLevel) -> List[Dict[str, Any]]:
        """根据严重程度生成步骤"""
        if severity == RiskLevel.CRITICAL:
            return [
                {"name": "冻结学习", "handler": "freeze_learning", "step_type": "automated"},
                {"name": "通知治理委员会", "handler": "notify_team", "params": {"team": "governance_committee"}, "step_type": "automated"},
                {"name": "恢复检查点", "handler": "restore_checkpoint", "step_type": "automated"},
                {"name": "人工验证", "description": "验证系统状态", "step_type": "verification", "verification_required": True},
                {"name": "审批恢复", "step_type": "approval"},
                {"name": "恢复学习", "handler": "resume_learning", "step_type": "automated"},
            ]
        elif severity == RiskLevel.HIGH:
            return [
                {"name": "冻结学习", "handler": "freeze_learning", "step_type": "automated"},
                {"name": "通知运维团队", "handler": "notify_team", "params": {"team": "operators"}, "step_type": "automated"},
                {"name": "恢复检查点", "handler": "restore_checkpoint", "step_type": "automated"},
                {"name": "验证系统", "handler": "verify_system", "step_type": "automated"},
                {"name": "恢复学习", "handler": "resume_learning", "step_type": "automated"},
            ]
        else:
            return [
                {"name": "验证系统", "handler": "verify_system", "step_type": "automated"},
                {"name": "记录问题", "step_type": "manual", "description": "记录问题详情"},
            ]
    
    def submit_for_approval(self, plan_id: UUID) -> RecoveryPlan:
        """提交计划进行审批"""
        plan = self._plans.get(plan_id)
        if not plan:
            raise NotFoundError("RecoveryPlan", str(plan_id))
        
        if plan.status != RecoveryPlanStatus.DRAFT:
            raise BusinessError(f"Plan cannot be submitted: status={plan.status}", code="INVALID_STATUS")
        
        plan.status = RecoveryPlanStatus.PENDING_APPROVAL
        
        logger.info(f"Recovery plan {plan_id} submitted for approval")
        
        return plan
    
    def approve_plan(
        self,
        plan_id: UUID,
        approver_id: UUID,
        comments: Optional[str] = None,
    ) -> RecoveryPlan:
        """审批计划"""
        plan = self._plans.get(plan_id)
        if not plan:
            raise NotFoundError("RecoveryPlan", str(plan_id))
        
        if plan.status != RecoveryPlanStatus.PENDING_APPROVAL:
            raise BusinessError(f"Plan cannot be approved: status={plan.status}", code="INVALID_STATUS")
        
        plan.approvals.append({
            "approver_id": str(approver_id),
            "approved_at": datetime.utcnow().isoformat(),
            "comments": comments,
        })
        
        if len(plan.approvals) >= plan.required_approvers:
            plan.status = RecoveryPlanStatus.APPROVED
            plan.approved_at = datetime.utcnow()
            logger.info(f"Recovery plan {plan_id} approved")
        
        return plan
    
    def reject_plan(
        self,
        plan_id: UUID,
        rejector_id: UUID,
        reason: str,
    ) -> RecoveryPlan:
        """拒绝计划"""
        plan = self._plans.get(plan_id)
        if not plan:
            raise NotFoundError("RecoveryPlan", str(plan_id))
        
        plan.status = RecoveryPlanStatus.REJECTED
        
        logger.info(f"Recovery plan {plan_id} rejected: {reason}")
        
        return plan
    
    def execute_plan(self, plan_id: UUID) -> RecoveryPlan:
        """执行恢复计划"""
        plan = self._plans.get(plan_id)
        if not plan:
            raise NotFoundError("RecoveryPlan", str(plan_id))
        
        if plan.status != RecoveryPlanStatus.APPROVED:
            raise BusinessError(f"Plan cannot be executed: status={plan.status}", code="INVALID_STATUS")
        
        plan.status = RecoveryPlanStatus.EXECUTING
        plan.started_at = datetime.utcnow()
        
        logger.info(f"Recovery plan {plan_id} execution started")
        
        # 执行步骤
        try:
            for i, step in enumerate(plan.steps):
                plan.current_step_index = i
                self._execute_step(plan, step)
            
            plan.status = RecoveryPlanStatus.COMPLETED
            plan.completed_at = datetime.utcnow()
            plan.actual_recovery_time = plan.completed_at - plan.started_at
            
            logger.info(f"Recovery plan {plan_id} completed successfully")
            
        except Exception as e:
            plan.status = RecoveryPlanStatus.FAILED
            logger.error(f"Recovery plan {plan_id} failed: {e}")
            raise
        
        return plan
    
    def _execute_step(self, plan: RecoveryPlan, step: RecoveryStep):
        """执行单个步骤"""
        step.status = "executing"
        step.started_at = datetime.utcnow()
        
        try:
            if step.step_type == SOPStep.AUTOMATED and step.handler:
                handler = self._step_handlers.get(step.handler)
                if handler:
                    result = handler(step.params)
                    step.result = result
                else:
                    step.result = {"warning": f"No handler found: {step.handler}"}
            
            elif step.step_type == SOPStep.MANUAL:
                # 人工步骤，标记为待处理
                step.status = "waiting_manual"
                return
            
            elif step.step_type == SOPStep.APPROVAL:
                # 审批步骤，检查是否已审批
                if not plan.approved_at:
                    step.status = "waiting_approval"
                    return
            
            step.status = "completed"
            step.completed_at = datetime.utcnow()
            
        except Exception as e:
            step.status = "failed"
            step.error = str(e)
            raise
    
    def get_plan(self, plan_id: UUID) -> Optional[RecoveryPlan]:
        """获取恢复计划"""
        return self._plans.get(plan_id)
    
    def list_plans(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[RecoveryPlan]:
        """列出恢复计划"""
        plans = list(self._plans.values())
        
        if status:
            plans = [p for p in plans if p.status.value == status]
        
        return sorted(plans, key=lambda p: p.created_at, reverse=True)[:limit]
    
    # ==================== 步骤处理器 ====================
    
    def _handler_freeze_learning(self, params: Dict) -> Dict[str, Any]:
        """冻结学习"""
        logger.info("Executing: Freeze learning")
        # 这里应该调用状态机服务
        return {"action": "freeze_learning", "status": "completed"}
    
    def _handler_restore_checkpoint(self, params: Dict) -> Dict[str, Any]:
        """恢复检查点"""
        logger.info("Executing: Restore checkpoint")
        return {"action": "restore_checkpoint", "status": "completed"}
    
    def _handler_verify_system(self, params: Dict) -> Dict[str, Any]:
        """验证系统"""
        logger.info("Executing: Verify system")
        return {"action": "verify_system", "status": "completed", "healthy": True}
    
    def _handler_resume_learning(self, params: Dict) -> Dict[str, Any]:
        """恢复学习"""
        logger.info("Executing: Resume learning")
        return {"action": "resume_learning", "status": "completed"}
    
    def _handler_notify_team(self, params: Dict) -> Dict[str, Any]:
        """通知团队"""
        team = params.get("team", "operators")
        logger.info(f"Executing: Notify team {team}")
        return {"action": "notify_team", "team": team, "status": "completed"}
    
    def register_step_handler(self, name: str, handler: Callable):
        """注册步骤处理器"""
        self._step_handlers[name] = handler


class SLAMonitor:
    """
    SLA 监控器
    
    监控恢复计划是否符合 SLA 要求
    """
    
    def __init__(self):
        # 默认 SLA 配置
        self.sla_configs = {
            RiskLevel.CRITICAL: timedelta(hours=4),
            RiskLevel.HIGH: timedelta(hours=2),
            RiskLevel.MEDIUM: timedelta(hours=1),
            RiskLevel.LOW: timedelta(minutes=30),
        }
    
    def check_sla(self, plan: RecoveryPlan) -> Dict[str, Any]:
        """检查 SLA 合规性"""
        if not plan.started_at:
            return {"compliant": True, "status": "not_started"}
        
        target = plan.target_recovery_time
        if not target:
            return {"compliant": True, "status": "no_sla_defined"}
        
        if plan.completed_at:
            actual = plan.actual_recovery_time
            compliant = actual <= target if actual else True
            
            return {
                "compliant": compliant,
                "status": "completed",
                "target": str(target),
                "actual": str(actual) if actual else None,
                "variance": str(actual - target) if actual else None,
            }
        else:
            elapsed = datetime.utcnow() - plan.started_at
            remaining = target - elapsed
            
            return {
                "compliant": remaining.total_seconds() > 0,
                "status": "in_progress",
                "target": str(target),
                "elapsed": str(elapsed),
                "remaining": str(remaining) if remaining.total_seconds() > 0 else "OVERDUE",
            }
    
    def set_sla(self, risk_level: RiskLevel, target_time: timedelta):
        """设置 SLA 目标"""
        self.sla_configs[risk_level] = target_time
    
    def get_sla(self, risk_level: RiskLevel) -> timedelta:
        """获取 SLA 目标"""
        return self.sla_configs.get(risk_level, timedelta(hours=1))

