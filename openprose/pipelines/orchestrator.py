"""
Pipeline Orchestrator

Executes complex multi-prompt workflows as directed acyclic graphs (DAGs).
Supports parallel execution, conditional branching, and data aggregation.
"""

from __future__ import annotations

import asyncio
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..vm.prose_vm import ProseVM, ExecutionContext


class NodeType(Enum):
    """Types of nodes in a pipeline."""

    PROMPT = "prompt"          # Execute a prompt
    PARALLEL = "parallel"      # Fan-out to multiple branches
    AGGREGATE = "aggregate"    # Collect results from branches
    CONDITION = "condition"    # Conditional branching
    TRANSFORM = "transform"    # Transform data without prompt
    INPUT = "input"           # Pipeline input
    OUTPUT = "output"         # Pipeline output


class EdgeCondition(Enum):
    """Conditions for edge traversal."""

    ALWAYS = "always"
    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    CONDITIONAL = "conditional"


@dataclass
class PipelineNode:
    """
    A node in the pipeline DAG.

    Each node represents an operation to perform, which could be
    executing a prompt, transforming data, or controlling flow.
    """

    node_id: str
    node_type: NodeType
    prompt_id: str | None = None
    transform_fn: Callable[[Any], Any] | None = None
    condition_fn: Callable[[Any], bool] | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "prompt_id": self.prompt_id,
            "config": self.config,
            "metadata": self.metadata,
        }


@dataclass
class PipelineEdge:
    """
    An edge connecting two nodes in the pipeline DAG.

    Edges define data flow and can have conditions for traversal.
    """

    source_id: str
    target_id: str
    condition: EdgeCondition = EdgeCondition.ALWAYS
    condition_fn: Callable[[Any], bool] | None = None
    data_mapping: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "condition": self.condition.value,
            "data_mapping": self.data_mapping,
        }


@dataclass
class NodeResult:
    """Result of executing a pipeline node."""

    node_id: str
    success: bool
    output: Any
    error: str | None = None
    execution_time: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineDefinition:
    """
    Definition of a complete pipeline.

    A pipeline is a DAG of nodes connected by edges, with defined
    inputs and outputs.
    """

    pipeline_id: str
    name: str
    description: str = ""
    nodes: dict[str, PipelineNode] = field(default_factory=dict)
    edges: list[PipelineEdge] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_node(self, node: PipelineNode) -> "PipelineDefinition":
        """Add a node to the pipeline."""
        self.nodes[node.node_id] = node
        return self

    def add_edge(self, edge: PipelineEdge) -> "PipelineDefinition":
        """Add an edge to the pipeline."""
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node not found: {edge.source_id}")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node not found: {edge.target_id}")
        self.edges.append(edge)
        return self

    def connect(
        self,
        source_id: str,
        target_id: str,
        condition: EdgeCondition = EdgeCondition.ALWAYS,
        data_mapping: dict[str, str] | None = None,
    ) -> "PipelineDefinition":
        """Convenience method to connect two nodes."""
        return self.add_edge(
            PipelineEdge(
                source_id=source_id,
                target_id=target_id,
                condition=condition,
                data_mapping=data_mapping or {},
            )
        )

    def get_entry_nodes(self) -> list[PipelineNode]:
        """Get nodes with no incoming edges."""
        targets = {e.target_id for e in self.edges}
        return [n for n in self.nodes.values() if n.node_id not in targets]

    def get_exit_nodes(self) -> list[PipelineNode]:
        """Get nodes with no outgoing edges."""
        sources = {e.source_id for e in self.edges}
        return [n for n in self.nodes.values() if n.node_id not in sources]

    def get_downstream(self, node_id: str) -> list[tuple[PipelineEdge, PipelineNode]]:
        """Get edges and nodes downstream from a node."""
        result = []
        for edge in self.edges:
            if edge.source_id == node_id:
                result.append((edge, self.nodes[edge.target_id]))
        return result

    def get_upstream(self, node_id: str) -> list[tuple[PipelineEdge, PipelineNode]]:
        """Get edges and nodes upstream from a node."""
        result = []
        for edge in self.edges:
            if edge.target_id == node_id:
                result.append((edge, self.nodes[edge.source_id]))
        return result

    def validate(self) -> list[str]:
        """Validate the pipeline definition."""
        errors = []

        # Check for cycles (simple DFS)
        visited = set()
        path = set()

        def has_cycle(node_id: str) -> bool:
            if node_id in path:
                return True
            if node_id in visited:
                return False

            visited.add(node_id)
            path.add(node_id)

            for edge, _ in self.get_downstream(node_id):
                if has_cycle(edge.target_id):
                    return True

            path.remove(node_id)
            return False

        for node_id in self.nodes:
            if has_cycle(node_id):
                errors.append("Pipeline contains a cycle")
                break

        # Check for orphaned nodes
        all_connected = set()
        for edge in self.edges:
            all_connected.add(edge.source_id)
            all_connected.add(edge.target_id)

        for node_id in self.nodes:
            if node_id not in all_connected and len(self.nodes) > 1:
                errors.append(f"Orphaned node: {node_id}")

        # Check for missing prompt references
        for node in self.nodes.values():
            if node.node_type == NodeType.PROMPT and not node.prompt_id:
                errors.append(f"Prompt node missing prompt_id: {node.node_id}")

        return errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "description": self.description,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "metadata": self.metadata,
        }


class PipelineOrchestrator:
    """
    Executes pipeline definitions using a ProseVM.

    The orchestrator handles:
    - Topological execution order
    - Parallel branch execution
    - Conditional routing
    - Result aggregation
    - Error handling and recovery
    """

    def __init__(self, vm: ProseVM):
        self.vm = vm
        self._results: dict[str, NodeResult] = {}
        self._execution_id: str | None = None
        self._start_time: datetime | None = None

    async def execute(
        self,
        pipeline: PipelineDefinition,
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a pipeline with the given inputs.

        Args:
            pipeline: The pipeline definition to execute
            inputs: Input data for the pipeline

        Returns:
            Pipeline execution results
        """
        # Validate
        errors = pipeline.validate()
        if errors:
            raise ValueError(f"Invalid pipeline: {errors}")

        self._execution_id = f"pipe_exec_{uuid.uuid4().hex[:8]}"
        self._start_time = datetime.now()
        self._results = {}

        # Initialize VM context with inputs
        if self.vm.context:
            self.vm.context.inputs = inputs

        # Build execution order (topological sort)
        execution_order = self._topological_sort(pipeline)

        # Execute nodes in order
        for node_id in execution_order:
            node = pipeline.nodes[node_id]
            await self._execute_node(pipeline, node, inputs)

        # Collect outputs from exit nodes
        outputs = {}
        for node in pipeline.get_exit_nodes():
            if node.node_id in self._results:
                result = self._results[node.node_id]
                if result.success:
                    outputs[node.node_id] = result.output

        return {
            "success": all(r.success for r in self._results.values()),
            "execution_id": self._execution_id,
            "outputs": outputs,
            "node_results": {
                nid: {
                    "success": r.success,
                    "output": r.output,
                    "error": r.error,
                    "execution_time": r.execution_time,
                }
                for nid, r in self._results.items()
            },
            "execution_time": (datetime.now() - self._start_time).total_seconds()
            if self._start_time
            else 0,
        }

    def _topological_sort(self, pipeline: PipelineDefinition) -> list[str]:
        """Sort nodes in topological order."""
        in_degree: dict[str, int] = defaultdict(int)
        for edge in pipeline.edges:
            in_degree[edge.target_id] += 1

        # Start with nodes that have no dependencies
        queue = [nid for nid in pipeline.nodes if in_degree[nid] == 0]
        result = []

        while queue:
            node_id = queue.pop(0)
            result.append(node_id)

            for edge, _ in pipeline.get_downstream(node_id):
                in_degree[edge.target_id] -= 1
                if in_degree[edge.target_id] == 0:
                    queue.append(edge.target_id)

        return result

    async def _execute_node(
        self,
        pipeline: PipelineDefinition,
        node: PipelineNode,
        global_inputs: dict[str, Any],
    ) -> NodeResult:
        """Execute a single pipeline node."""
        start_time = datetime.now()

        # Collect inputs from upstream nodes
        node_inputs = dict(global_inputs)
        for edge, upstream_node in pipeline.get_upstream(node.node_id):
            if upstream_node.node_id in self._results:
                upstream_result = self._results[upstream_node.node_id]
                if upstream_result.success:
                    # Apply data mapping
                    if edge.data_mapping:
                        for target_key, source_key in edge.data_mapping.items():
                            if isinstance(upstream_result.output, dict):
                                node_inputs[target_key] = upstream_result.output.get(
                                    source_key
                                )
                            else:
                                node_inputs[target_key] = upstream_result.output
                    else:
                        # Default: pass output as 'content'
                        node_inputs["content"] = upstream_result.output

        try:
            output = await self._execute_node_type(node, node_inputs)
            result = NodeResult(
                node_id=node.node_id,
                success=True,
                output=output,
                execution_time=(datetime.now() - start_time).total_seconds(),
            )
        except Exception as e:
            result = NodeResult(
                node_id=node.node_id,
                success=False,
                output=None,
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds(),
            )

        self._results[node.node_id] = result
        return result

    async def _execute_node_type(
        self,
        node: PipelineNode,
        inputs: dict[str, Any],
    ) -> Any:
        """Execute node based on its type."""
        if node.node_type == NodeType.PROMPT:
            if not node.prompt_id:
                raise ValueError(f"Prompt node missing prompt_id: {node.node_id}")

            result = await self.vm.execute_prompt(node.prompt_id, **inputs)
            return result.get("output")

        elif node.node_type == NodeType.TRANSFORM:
            if node.transform_fn:
                return node.transform_fn(inputs)
            return inputs

        elif node.node_type == NodeType.INPUT:
            return inputs

        elif node.node_type == NodeType.OUTPUT:
            return inputs

        elif node.node_type == NodeType.PARALLEL:
            # Parallel nodes just pass through
            return inputs

        elif node.node_type == NodeType.AGGREGATE:
            # Aggregate collects all upstream results
            return inputs

        elif node.node_type == NodeType.CONDITION:
            if node.condition_fn:
                return node.condition_fn(inputs)
            return inputs

        else:
            raise ValueError(f"Unknown node type: {node.node_type}")


def create_research_pipeline(
    name: str,
    prompt_sequence: list[str],
    description: str = "",
) -> PipelineDefinition:
    """
    Helper to create a simple sequential research pipeline.

    Args:
        name: Pipeline name
        prompt_sequence: List of prompt IDs to execute in order
        description: Optional description

    Returns:
        Configured PipelineDefinition
    """
    pipeline = PipelineDefinition(
        pipeline_id=f"pipeline_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
    )

    # Add input node
    pipeline.add_node(
        PipelineNode(node_id="input", node_type=NodeType.INPUT)
    )

    # Add prompt nodes
    prev_id = "input"
    for prompt_id in prompt_sequence:
        node_id = f"prompt_{prompt_id}"
        pipeline.add_node(
            PipelineNode(
                node_id=node_id,
                node_type=NodeType.PROMPT,
                prompt_id=prompt_id,
            )
        )
        pipeline.connect(prev_id, node_id)
        prev_id = node_id

    # Add output node
    pipeline.add_node(
        PipelineNode(node_id="output", node_type=NodeType.OUTPUT)
    )
    pipeline.connect(prev_id, "output")

    return pipeline


def create_parallel_analysis_pipeline(
    name: str,
    parallel_prompts: list[str],
    aggregation_prompt: str | None = None,
    description: str = "",
) -> PipelineDefinition:
    """
    Helper to create a pipeline that runs prompts in parallel then aggregates.

    Args:
        name: Pipeline name
        parallel_prompts: List of prompt IDs to run in parallel
        aggregation_prompt: Optional prompt to aggregate results
        description: Optional description

    Returns:
        Configured PipelineDefinition
    """
    pipeline = PipelineDefinition(
        pipeline_id=f"pipeline_{uuid.uuid4().hex[:8]}",
        name=name,
        description=description,
    )

    # Input
    pipeline.add_node(
        PipelineNode(node_id="input", node_type=NodeType.INPUT)
    )

    # Parallel fan-out
    pipeline.add_node(
        PipelineNode(node_id="fanout", node_type=NodeType.PARALLEL)
    )
    pipeline.connect("input", "fanout")

    # Parallel prompt nodes
    for prompt_id in parallel_prompts:
        node_id = f"prompt_{prompt_id}"
        pipeline.add_node(
            PipelineNode(
                node_id=node_id,
                node_type=NodeType.PROMPT,
                prompt_id=prompt_id,
            )
        )
        pipeline.connect("fanout", node_id)

    # Aggregation
    pipeline.add_node(
        PipelineNode(node_id="aggregate", node_type=NodeType.AGGREGATE)
    )
    for prompt_id in parallel_prompts:
        pipeline.connect(f"prompt_{prompt_id}", "aggregate")

    if aggregation_prompt:
        pipeline.add_node(
            PipelineNode(
                node_id="final_prompt",
                node_type=NodeType.PROMPT,
                prompt_id=aggregation_prompt,
            )
        )
        pipeline.connect("aggregate", "final_prompt")
        pipeline.add_node(
            PipelineNode(node_id="output", node_type=NodeType.OUTPUT)
        )
        pipeline.connect("final_prompt", "output")
    else:
        pipeline.add_node(
            PipelineNode(node_id="output", node_type=NodeType.OUTPUT)
        )
        pipeline.connect("aggregate", "output")

    return pipeline
