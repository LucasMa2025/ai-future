"""
LLM 适配器使用示例

演示如何使用通用 LLM 适配器框架进行自学习。

本示例展示：
1. 如何创建不同类型的 LLM 适配器
2. 如何将适配器注入到自学习系统
3. 如何验证完整的学习流程

注意：学习的起点是 LLM 的现有知识库，而不是从零开始。
"""
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.adapters import (
    LLMAdapterFactory,
    LLMAdapterRegistry,
    LLMConfig,
    DeepSeekAdapter,
    OllamaAdapter,
    VLLMAdapter,
    OpenAICompatAdapter,
)
from llm.adapters.base import MockLLMAdapter

from self_learning import (
    AutonomousExplorer,
    KnowledgeGenerator,
    KernelFactory,
    LearningScope,
    NLLevel,
)


def demo_list_adapters():
    """演示列出所有可用适配器"""
    print("\n" + "="*60)
    print("可用的 LLM 适配器")
    print("="*60)
    
    registry = LLMAdapterRegistry()
    for info in registry.list_adapters():
        print(f"\n适配器: {info.name}")
        print(f"  描述: {info.description}")
        print(f"  能力: {[c.value for c in info.capabilities]}")
        if info.default_config:
            print(f"  默认配置: {info.default_config}")


def demo_create_adapter():
    """演示创建适配器"""
    print("\n" + "="*60)
    print("创建 LLM 适配器")
    print("="*60)
    
    # 方式1：使用工厂
    print("\n1. 使用工厂创建 DeepSeek 适配器:")
    adapter = LLMAdapterFactory.create(
        "deepseek",
        base_url="http://localhost:8001/v1",
        model="deepseek-chat",
    )
    print(f"   {adapter}")
    
    # 方式2：直接实例化
    print("\n2. 直接实例化 Ollama 适配器:")
    config = LLMConfig(
        base_url="http://localhost:11434",
        model="llama3.2:latest",
    )
    adapter = OllamaAdapter(config=config)
    print(f"   {adapter}")
    
    # 方式3：使用模拟适配器（测试用）
    print("\n3. 创建模拟适配器（测试用）:")
    adapter = MockLLMAdapter()
    print(f"   {adapter}")
    
    return adapter


def demo_explorer_with_adapter():
    """演示使用适配器的探索器"""
    print("\n" + "="*60)
    print("使用 LLM 适配器进行自主探索")
    print("="*60)
    
    # 方式1：使用 with_adapter 类方法
    print("\n1. 使用 with_adapter 创建探索器:")
    print("   explorer = AutonomousExplorer.with_adapter('mock')")
    explorer = AutonomousExplorer.with_adapter("mock")
    
    # 方式2：直接注入适配器
    print("\n2. 直接注入适配器:")
    adapter = MockLLMAdapter()
    explorer = AutonomousExplorer(llm_adapter=adapter, max_depth=3)
    
    # 执行探索
    print("\n3. 执行探索...")
    result = explorer.explore(
        goal="学习 Python 元编程技术",
        initial_context={"domain": "programming", "language": "python"}
    )
    
    print(f"\n探索结果:")
    print(f"  状态: {result['status']}")
    print(f"  迭代次数: {result['iterations']}")
    print(f"  发现数量: {len(result['findings'])}")
    print(f"  LLM 信息: {explorer.get_llm_info()}")


def demo_knowledge_generator_with_adapter():
    """演示使用适配器的知识生成器"""
    print("\n" + "="*60)
    print("使用 LLM 适配器进行知识生成")
    print("="*60)
    
    # 创建生成器
    generator = KnowledgeGenerator.with_adapter("mock")
    
    # 模拟探索结果
    exploration_path = [
        {"step_id": "s1", "action": "reasoning", "query": "分析元编程", "result": "..."},
        {"step_id": "s2", "action": "hypothesis", "query": "装饰器应用", "result": "..."},
    ]
    findings = [
        {"type": "concept", "content": "装饰器是元编程的核心", "confidence": 0.85},
        {"type": "pattern", "content": "常用于横切关注点", "confidence": 0.80},
    ]
    
    # 生成知识
    result = generator.generate(
        goal="学习 Python 元编程技术",
        exploration_path=exploration_path,
        findings=findings,
    )
    
    print(f"\n生成结果:")
    print(f"  知识领域: {result['knowledge'].domain}")
    print(f"  知识类型: {result['knowledge'].type.value}")
    print(f"  置信度: {result['knowledge'].confidence}")
    print(f"  约束数量: {len(result['proposed_constraints'])}")
    print(f"  信号数量: {len(result['signals'])}")


def demo_nl_kernel_with_adapter():
    """演示使用适配器的 NL 内核"""
    print("\n" + "="*60)
    print("使用 LLM 适配器创建 NL 内核")
    print("="*60)
    
    # 使用工厂创建内核
    kernel = KernelFactory.create_with_adapter(
        adapter_type="mock",
        config={
            "memory_config": {"capacity": 100},
            "optimizer_config": {"learning_rate": 0.001},
        }
    )
    
    print(f"\n内核状态:")
    print(f"  初始化: {kernel._initialized}")
    print(f"  冻结: {kernel.is_frozen()}")
    
    # 解冻并执行学习
    kernel.unfreeze()
    
    # 创建学习范围
    scope = LearningScope(
        scope_id="demo_scope",
        max_level=NLLevel.MEMORY,
        allowed_levels=[NLLevel.PARAMETER, NLLevel.MEMORY],
        created_by="demo_user",
    )
    
    # 执行学习步骤
    context = {
        "goal": "学习 Python 元编程",
        "step": 0,
        "action": "reasoning",
        "findings": [],
    }
    
    segment = kernel.execute_learning_step(context, scope)
    
    print(f"\n学习结果:")
    print(f"  Segment ID: {segment.segment_id}")
    print(f"  影响层级: {[l.name for l in segment.get_affected_levels()]}")
    
    # 获取统计信息
    stats = kernel.get_statistics()
    print(f"\n内核统计:")
    print(f"  学习历史数: {stats['learning_history_count']}")
    print(f"  LLM 信息: {stats['llm']}")
    
    # 冻结内核
    kernel.freeze()


def demo_runtime_adapter_switch():
    """演示运行时切换适配器"""
    print("\n" + "="*60)
    print("运行时切换 LLM 适配器")
    print("="*60)
    
    # 创建探索器
    explorer = AutonomousExplorer.with_adapter("mock")
    print(f"\n初始适配器: {explorer.get_llm_info()['adapter_name']}")
    
    # 运行时切换到另一个适配器
    new_adapter = MockLLMAdapter()
    new_adapter.adapter_name = "mock_v2"  # 修改名称以区分
    explorer.set_adapter(new_adapter)
    
    print(f"切换后适配器: {explorer.get_llm_info()['adapter_name']}")


def demo_local_deployment_workflow():
    """
    演示本地部署工作流
    
    这是验证完整流程的关键示例：
    自学习 -> Learning Unit -> 审计系统 -> 桥接 -> 生产力
    """
    print("\n" + "="*60)
    print("本地部署完整工作流演示")
    print("="*60)
    
    print("""
本地部署方案说明：

1. 部署选项：
   - Ollama: 最简单，支持 Llama, Qwen, Mistral 等
     命令: ollama run llama3.2
   
   - vLLM: 高性能，支持 LoRA 动态加载
     命令: python -m vllm.entrypoints.openai.api_server --model <model>
   
   - DeepSeek 本地: 使用 vLLM 或 TensorRT-LLM 部署 DeepSeek 模型

2. 知识桥接能力：
   - Ollama: 支持通过 Modelfile 注入知识到系统提示词
   - vLLM: 支持 LoRA 适配器动态加载/卸载
   - 两者都支持 OpenAI 兼容 API

3. 验证流程：
   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
   │  自学习系统  │────▶│ Learning Unit │────▶│  审计系统   │
   │ (LLM 探索)  │     │  (新知识封装) │     │ (人工审核)  │
   └─────────────┘     └──────────────┘     └─────────────┘
                                                   │
                                                   ▼
   ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
   │   生产模型   │◀────│   桥接系统    │◀────│ 批准的 LU   │
   │ (知识已更新) │     │ (知识写入)   │     │            │
   └─────────────┘     └──────────────┘     └─────────────┘

4. 代码示例（使用 Ollama）：

   # 创建适配器
   adapter = OllamaAdapter(LLMConfig(
       base_url="http://localhost:11434",
       model="llama3.2:latest",
   ))
   
   # 创建探索器
   explorer = AutonomousExplorer(llm_adapter=adapter)
   
   # 执行探索
   result = explorer.explore(goal="...")
   
   # 生成知识
   generator = KnowledgeGenerator(llm_adapter=adapter)
   knowledge = generator.generate(...)
   
   # 提交到审计系统
   # (通过 NLLearningUnitBuilder)
   
   # 审核通过后，桥接到生产模型
   # (使用 Ollama 的 create_model_with_knowledge 方法)
    """)


def main():
    """主函数"""
    print("="*60)
    print("LLM 适配器框架演示")
    print("="*60)
    
    # 1. 列出可用适配器
    demo_list_adapters()
    
    # 2. 创建适配器
    demo_create_adapter()
    
    # 3. 探索器使用适配器
    demo_explorer_with_adapter()
    
    # 4. 知识生成器使用适配器
    demo_knowledge_generator_with_adapter()
    
    # 5. NL 内核使用适配器
    demo_nl_kernel_with_adapter()
    
    # 6. 运行时切换适配器
    demo_runtime_adapter_switch()
    
    # 7. 本地部署工作流
    demo_local_deployment_workflow()
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60)


if __name__ == "__main__":
    main()

