"""
OpenProse: Prompt Collection to VM Translation Layer

OpenProse transforms structured prompt collections into independent,
sandboxed "Virtual Machines" that execute research and analysis pipelines.

Core Concepts:
- PromptCollection: A curated set of prompts with metadata
- ProseVM: An isolated execution context for running prompt pipelines
- Pipeline: A directed graph of prompt executions with data flow
"""

from .vm.prose_vm import ProseVM, VMConfig, ExecutionContext
from .vm.sandbox import Sandbox, SandboxPolicy
from .pipelines.orchestrator import PipelineOrchestrator, PipelineDefinition
from .prompts.loader import PromptCollection, Prompt, load_collection

__version__ = "0.1.0"

__all__ = [
    # VM Layer
    "ProseVM",
    "VMConfig",
    "ExecutionContext",
    # Sandbox
    "Sandbox",
    "SandboxPolicy",
    # Pipelines
    "PipelineOrchestrator",
    "PipelineDefinition",
    # Prompts
    "PromptCollection",
    "Prompt",
    "load_collection",
]
