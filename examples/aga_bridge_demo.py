"""
AGA Bridge 使用示例

演示如何使用 AGA 热插拔式知识系统。

运行方式：
    python examples/aga_bridge_demo.py

依赖：
    pip install transformers torch
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from dataclasses import dataclass
from typing import List


# ==================== 模拟 core.types ====================
@dataclass
class ProposedConstraint:
    """约束"""
    condition: str
    decision: str
    confidence: float = 0.9


@dataclass
class LearningUnit:
    """学习单元"""
    id: str
    proposed_constraints: List[ProposedConstraint]


@dataclass
class AuditApproval:
    """审计批准"""
    approval_id: str
    decision: str = "approve"
    
    def verify(self):
        return True


# ==================== 演示代码 ====================
def demo_without_model():
    """
    无模型演示 - 展示 AGA 核心机制
    """
    print("=" * 60)
    print("AGA Bridge 演示（无模型模式）")
    print("=" * 60)
    
    import torch
    from bridge.aga_core import (
        AuxiliaryGovernedAttention,
        LifecycleState,
    )
    
    # 1. 创建 AGA 模块
    print("\n[1] 创建 AGA 模块")
    aga = AuxiliaryGovernedAttention(
        hidden_dim=256,      # 演示用小维度
        bottleneck_dim=32,
        num_slots=10,
        num_heads=4,
    )
    aga.eval()
    print(f"    ✓ 创建成功: {aga.num_slots} 槽位, {aga.bottleneck_dim} 瓶颈维度")
    
    # 2. 注入知识
    print("\n[2] 注入知识（零训练）")
    
    # 模拟知识编码
    key_vec = torch.randn(32)    # 条件编码
    val_vec = torch.randn(256)   # 修正信号
    
    aga.inject_knowledge(
        slot_idx=0,
        key_vector=key_vec,
        value_vector=val_vec,
        lu_id="LU_001_demo",
        lifecycle_state=LifecycleState.PROBATIONARY,
    )
    print(f"    ✓ 知识注入成功")
    print(f"    ✓ 生命周期: {aga.slot_lifecycle[0].value}")
    print(f"    ✓ 可靠性: {aga.LIFECYCLE_RELIABILITY[aga.slot_lifecycle[0]]}")
    
    # 3. 模拟推理
    print("\n[3] 模拟推理")
    
    batch_size, seq_len = 2, 8
    hidden_states = torch.randn(batch_size, seq_len, 256)
    primary_output = torch.randn(batch_size, seq_len, 256)
    
    fused_output, diagnostics = aga(
        hidden_states=hidden_states,
        primary_attention_output=primary_output,
        primary_attention_weights=None,  # 没有主注意力权重
        return_diagnostics=True,
    )
    
    print(f"    ✓ 输入形状: {hidden_states.shape}")
    print(f"    ✓ 输出形状: {fused_output.shape}")
    print(f"    ✓ 活跃槽位: {diagnostics.active_slots}")
    print(f"    ✓ 最活跃槽位: {diagnostics.top_activated_slots[:3]}")
    
    # 4. 生命周期管理
    print("\n[4] 生命周期管理")
    
    # 确认知识
    aga.confirm_slot(0)
    print(f"    ✓ 确认后生命周期: {aga.slot_lifecycle[0].value}")
    print(f"    ✓ 确认后可靠性: {aga.LIFECYCLE_RELIABILITY[aga.slot_lifecycle[0]]}")
    
    # 隔离知识
    aga.quarantine_slot(0)
    print(f"    ✓ 隔离后生命周期: {aga.slot_lifecycle[0].value}")
    print(f"    ✓ 隔离后可靠性: {aga.LIFECYCLE_RELIABILITY[aga.slot_lifecycle[0]]}")
    
    # 5. 统计信息
    print("\n[5] 统计信息")
    stats = aga.get_statistics()
    print(f"    总槽位: {stats['total_slots']}")
    print(f"    活跃槽位: {stats['active_slots']}")
    print(f"    状态分布: {stats['state_distribution']}")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)


def demo_with_model():
    """
    有模型演示 - 完整流程
    
    需要安装 transformers 和足够的 GPU 内存
    """
    print("=" * 60)
    print("AGA Bridge 完整演示（需要模型）")
    print("=" * 60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from bridge import create_bridge, LifecycleState
    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请运行: pip install transformers torch")
        return
    
    # 1. 加载模型（使用小模型演示）
    print("\n[1] 加载模型...")
    model_name = "gpt2"  # 使用 GPT-2 作为演示
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()
        print(f"    ✓ 模型加载成功: {model_name}")
    except Exception as e:
        print(f"    ✗ 模型加载失败: {e}")
        return
    
    # 2. 创建 AGA Bridge
    print("\n[2] 创建 AGA Bridge...")
    
    # GPT-2 的配置
    hidden_dim = model.config.n_embd  # 768 for gpt2
    num_heads = model.config.n_head   # 12 for gpt2
    
    bridge = create_bridge(
        model=model,
        tokenizer=tokenizer,
        bridge_type="aga",
        hidden_dim=hidden_dim,
        aga_num_slots=50,
        aga_target_layers=[-2, -1],  # 最后两层
    )
    print(f"    ✓ Bridge 创建成功")
    
    # 3. 创建 Learning Unit
    print("\n[3] 创建 Learning Unit...")
    
    lu = LearningUnit(
        id="LU_DEMO_001",
        proposed_constraints=[
            ProposedConstraint(
                condition="What is the capital of France",
                decision="Paris",
            ),
        ],
    )
    print(f"    ✓ LU 创建成功: {lu.id}")
    
    # 4. 写入知识
    print("\n[4] 写入知识...")
    
    approval = AuditApproval(approval_id="APPROVAL_001")
    result = bridge.write_learning_unit(
        learning_unit=lu,
        writer_id="audit_system",
        audit_approval=approval,
    )
    print(f"    ✓ 写入结果: {result.value}")
    
    # 5. 推理测试
    print("\n[5] 推理测试...")
    
    test_input = "What is the capital of France?"
    inputs = tokenizer(test_input, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"    输入: {test_input}")
    print(f"    输出: {response}")
    
    # 6. 查看状态
    print("\n[6] 查看状态...")
    
    status = bridge.get_lu_status(lu.id)
    if status:
        print(f"    LU ID: {status['lu_id']}")
        for layer_idx, layer_info in status['layers'].items():
            print(f"    Layer {layer_idx}: slot={layer_info['slot_idx']}, "
                  f"lifecycle={layer_info['lifecycle']}, hits={layer_info['hit_count']}")
    
    # 7. 统计信息
    print("\n[7] 统计信息...")
    stats = bridge.get_statistics()
    print(f"    已挂载: {stats['attached']}")
    print(f"    活跃 LU: {stats['active_lus']}")
    print(f"    成功写入: {stats['successful_writes']}")
    
    # 8. 清理
    print("\n[8] 清理...")
    bridge.detach()
    print("    ✓ AGA 已卸载")
    
    print("\n" + "=" * 60)
    print("完整演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AGA Bridge 演示")
    parser.add_argument("--with-model", action="store_true", help="运行完整模型演示")
    args = parser.parse_args()
    
    if args.with_model:
        demo_with_model()
    else:
        demo_without_model()

