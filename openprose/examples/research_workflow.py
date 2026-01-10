"""
OpenProse Research Workflow Example

Demonstrates how to use OpenProse VMs to execute research analysis
pipelines using the viral research prompts collection.

This example shows:
1. Loading a prompt collection
2. Creating and configuring a ProseVM
3. Executing individual prompts
4. Running prompt chains
5. Building custom pipelines
6. Parallel analysis workflows
"""

import asyncio
from pathlib import Path
from typing import Any

# Import OpenProse components
from openprose import (
    ProseVM,
    VMConfig,
    PromptCollection,
    load_collection,
    PipelineOrchestrator,
    SandboxPolicy,
    ResourceLimits,
)
from openprose.pipelines.orchestrator import (
    create_research_pipeline,
    create_parallel_analysis_pipeline,
)
from openprose.vm.sandbox import IsolationLevel


# Mock LLM executor for demonstration
def mock_llm_executor(prompt: str) -> str:
    """
    Mock LLM executor for demonstration purposes.

    In production, replace with actual LLM API calls:
    - Anthropic Claude API
    - OpenAI API
    - Local Ollama instance
    - etc.
    """
    return f"[Analysis Result for prompt of {len(prompt)} chars]\n\nThis is a mock response."


def create_real_executor(api_key: str, model: str = "claude-sonnet-4-20250514"):
    """
    Create a real LLM executor using Anthropic's API.

    Example usage:
        executor = create_real_executor(os.environ["ANTHROPIC_API_KEY"])
        vm.set_executor(executor)
    """
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        def executor(prompt: str) -> str:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text

        return executor
    except ImportError:
        raise ImportError("Install anthropic package: pip install anthropic")


async def example_single_prompt():
    """Execute a single prompt from the collection."""
    print("\n" + "=" * 60)
    print("Example 1: Single Prompt Execution")
    print("=" * 60)

    # Load the viral research tools collection
    collection_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "collections"
        / "viral_research_tools.yaml"
    )
    collection = load_collection(collection_path)

    print(f"Loaded collection: {collection.name}")
    print(f"Available prompts: {len(collection)}")

    # Create VM from collection
    vm = ProseVM.from_collection(
        collection,
        config=VMConfig(
            name="research_vm",
            description="VM for research analysis",
            resource_limits=ResourceLimits(
                max_execution_time_seconds=60,
                max_output_tokens=10000,
            ),
            sandbox_policy=SandboxPolicy(
                audit_logging=True,
                require_human_approval=False,
            ),
        ),
    )

    # Set the executor (use mock for demo)
    vm.set_executor(mock_llm_executor)

    # Execute within VM context
    async with vm:
        # Find contradictions in a document
        result = await vm.execute_prompt(
            "contradictions_finder",
            content="""
            The company's revenue grew 15% year-over-year.
            However, due to market conditions, we experienced a decline in all metrics.
            Our customer satisfaction scores reached an all-time high of 92%.
            Unfortunately, customer complaints increased by 30% this quarter.
            We are confident in our market position despite losing 3 major clients.
            """,
        )

        print(f"\nContradictions Finder Result:")
        print(f"Success: {result['success']}")
        print(f"Output: {result['output'][:200]}...")

    # Show metrics
    print(f"\nVM Metrics: {vm.get_metrics()}")


async def example_prompt_chain():
    """Execute a chain of prompts with data flow."""
    print("\n" + "=" * 60)
    print("Example 2: Prompt Chain Execution")
    print("=" * 60)

    collection_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "collections"
        / "viral_research_tools.yaml"
    )
    collection = load_collection(collection_path)

    vm = ProseVM.from_collection(collection)
    vm.set_executor(mock_llm_executor)

    async with vm:
        # Chain: Turn into paper -> Find contradictions -> Stress test assumptions
        result = await vm.execute_chain(
            prompt_ids=[
                "turn_into_paper",
                "contradictions_finder",
                "assumption_stress_test",
            ],
            initial_content="""
            Notes from today's meeting:
            - Users want faster load times
            - Maybe cache more aggressively?
            - Security team worried about data exposure
            - Need to balance speed vs safety
            - Consider CDN but costs are high
            - Alternative: lazy loading
            """,
        )

        print(f"\nChain Execution Result:")
        print(f"Success: {result['success']}")
        print(f"Steps completed: {len(result['chain'])}")

        for step in result["chain"]:
            print(f"  - {step['prompt_id']}: {step['result']['success']}")


async def example_custom_pipeline():
    """Build and execute a custom analysis pipeline."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Pipeline")
    print("=" * 60)

    collection_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "collections"
        / "viral_research_tools.yaml"
    )
    collection = load_collection(collection_path)

    vm = ProseVM.from_collection(collection)
    vm.set_executor(mock_llm_executor)

    # Create a custom research pipeline
    pipeline = create_research_pipeline(
        name="Deep Analysis Pipeline",
        prompt_sequence=[
            "turn_into_paper",       # Structure the content
            "reviewer_2",            # Harsh critique
            "assumption_stress_test", # Test assumptions
            "one_page_mental_model",  # Compress to essentials
        ],
        description="Comprehensive research document analysis",
    )

    print(f"Pipeline: {pipeline.name}")
    print(f"Nodes: {len(pipeline.nodes)}")
    print(f"Edges: {len(pipeline.edges)}")

    async with vm:
        orchestrator = PipelineOrchestrator(vm)

        result = await orchestrator.execute(
            pipeline,
            inputs={
                "content": """
                Our new machine learning model achieves 95% accuracy on the test set.
                We used a dataset of 10,000 samples collected from public sources.
                The model was trained for 100 epochs with standard hyperparameters.
                We believe this approach will generalize to production environments.
                """,
            },
        )

        print(f"\nPipeline Result:")
        print(f"Success: {result['success']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print(f"Node results: {len(result['node_results'])}")


async def example_parallel_analysis():
    """Run multiple analyses in parallel."""
    print("\n" + "=" * 60)
    print("Example 4: Parallel Analysis")
    print("=" * 60)

    collection_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "collections"
        / "viral_research_tools.yaml"
    )
    collection = load_collection(collection_path)

    vm = ProseVM.from_collection(collection)
    vm.set_executor(mock_llm_executor)

    # Create pipeline that runs multiple analyses in parallel
    pipeline = create_parallel_analysis_pipeline(
        name="Multi-Perspective Analysis",
        parallel_prompts=[
            "contradictions_finder",  # Find logical issues
            "reviewer_2",             # Harsh critique
            "what_would_break",       # Failure modes
            "assumption_stress_test", # Test assumptions
        ],
        description="Analyze content from multiple critical perspectives",
    )

    print(f"Pipeline: {pipeline.name}")
    print(f"Parallel branches: 4")

    async with vm:
        orchestrator = PipelineOrchestrator(vm)

        result = await orchestrator.execute(
            pipeline,
            inputs={
                "content": """
                We propose a new distributed consensus algorithm that achieves
                sub-second finality with 10,000 nodes. The algorithm uses a novel
                leader election mechanism based on stake-weighted random selection.
                Our simulations show 99.99% uptime under normal conditions.
                """,
            },
        )

        print(f"\nParallel Analysis Result:")
        print(f"Success: {result['success']}")
        print(f"Outputs collected: {len(result['outputs'])}")


async def example_list_prompts():
    """List all available prompts in a collection."""
    print("\n" + "=" * 60)
    print("Example 5: List Available Prompts")
    print("=" * 60)

    collection_path = (
        Path(__file__).parent.parent
        / "prompts"
        / "collections"
        / "viral_research_tools.yaml"
    )
    collection = load_collection(collection_path)

    vm = ProseVM.from_collection(collection)

    print(f"\nCollection: {collection.name}")
    print(f"Author: {collection.metadata.author}")
    print(f"Source: {collection.metadata.source}")
    print(f"\nAvailable Prompts:")

    for prompt_info in vm.list_prompts():
        print(f"\n  [{prompt_info['id']}]")
        print(f"    Name: {prompt_info['name']}")
        print(f"    Category: {prompt_info['category']}")
        print(f"    Intent: {prompt_info['intent']}")


async def main():
    """Run all examples."""
    print("OpenProse Research Workflow Examples")
    print("=" * 60)

    await example_list_prompts()
    await example_single_prompt()
    await example_prompt_chain()
    await example_custom_pipeline()
    await example_parallel_analysis()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
