"""
OpenProse Pipeline Orchestration

Defines and executes complex multi-prompt workflows with DAG-based
execution, parallel processing, and conditional branching.
"""

from .orchestrator import (
    PipelineOrchestrator,
    PipelineDefinition,
    PipelineNode,
    PipelineEdge,
    NodeType,
)

__all__ = [
    "PipelineOrchestrator",
    "PipelineDefinition",
    "PipelineNode",
    "PipelineEdge",
    "NodeType",
]
