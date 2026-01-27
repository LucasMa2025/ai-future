"""
自定义异常
"""
from typing import Any, Optional, Dict


class NLGSMException(Exception):
    """NLGSM 基础异常"""
    
    def __init__(
        self,
        message: str,
        code: str = "NLGSM_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)


# ==================== 认证异常 ====================

class AuthenticationError(NLGSMException):
    """认证错误"""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTH_ERROR")


class InvalidCredentialsError(AuthenticationError):
    """凭据无效"""
    
    def __init__(self):
        super().__init__("Invalid username or password")
        self.code = "INVALID_CREDENTIALS"


class TokenExpiredError(AuthenticationError):
    """Token 过期"""
    
    def __init__(self):
        super().__init__("Token has expired")
        self.code = "TOKEN_EXPIRED"


class TokenInvalidError(AuthenticationError):
    """Token 无效"""
    
    def __init__(self, message: str = "Invalid token"):
        super().__init__(message)
        self.code = "TOKEN_INVALID"


class TokenBlacklistedError(AuthenticationError):
    """Token 已注销"""
    
    def __init__(self):
        super().__init__("Token has been revoked")
        self.code = "TOKEN_BLACKLISTED"


class UserInactiveError(AuthenticationError):
    """用户未激活"""
    
    def __init__(self):
        super().__init__("User account is inactive")
        self.code = "USER_INACTIVE"


# ==================== 权限异常 ====================

class PermissionDeniedError(NLGSMException):
    """权限不足"""
    
    def __init__(self, permission: Optional[str] = None):
        message = f"Permission denied: {permission}" if permission else "Permission denied"
        super().__init__(message, code="PERMISSION_DENIED")
        self.details = {"required_permission": permission}


class InsufficientRiskLevelError(PermissionDeniedError):
    """风险等级审批权限不足"""
    
    def __init__(self, required_level: str, user_level: Optional[str] = None):
        message = f"Cannot approve {required_level} risk level"
        super().__init__()
        self.message = message
        self.code = "INSUFFICIENT_RISK_LEVEL"
        self.details = {
            "required_level": required_level,
            "user_level": user_level
        }


# ==================== 状态机异常 ====================

class StateMachineError(NLGSMException):
    """状态机错误"""
    
    def __init__(self, message: str):
        super().__init__(message, code="STATE_MACHINE_ERROR")


class InvalidStateTransitionError(StateMachineError):
    """无效状态转换"""
    
    def __init__(self, from_state: str, to_state: str, event: str):
        message = f"Invalid transition from {from_state} to {to_state} via {event}"
        super().__init__(message)
        self.code = "INVALID_TRANSITION"
        self.details = {
            "from_state": from_state,
            "to_state": to_state,
            "event": event
        }


# ==================== 资源异常 ====================

class NotFoundError(NLGSMException):
    """资源不存在"""
    
    def __init__(self, resource: str, identifier: Any):
        message = f"{resource} not found: {identifier}"
        super().__init__(message, code="NOT_FOUND")
        self.details = {"resource": resource, "identifier": str(identifier)}


class AlreadyExistsError(NLGSMException):
    """资源已存在"""
    
    def __init__(self, resource: str, identifier: Any):
        message = f"{resource} already exists: {identifier}"
        super().__init__(message, code="ALREADY_EXISTS")
        self.details = {"resource": resource, "identifier": str(identifier)}


# ==================== 审批异常 ====================

class ApprovalError(NLGSMException):
    """审批错误"""
    
    def __init__(self, message: str):
        super().__init__(message, code="APPROVAL_ERROR")


class ApprovalAlreadyProcessedError(ApprovalError):
    """审批已处理"""
    
    def __init__(self):
        super().__init__("Approval has already been processed")
        self.code = "ALREADY_PROCESSED"


# ==================== 业务异常 ====================

class BusinessError(NLGSMException):
    """业务逻辑错误"""
    
    def __init__(self, message: str, code: str = "BUSINESS_ERROR"):
        super().__init__(message, code=code)

