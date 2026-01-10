"""
OpenProse VM Layer

Translates prompt collections into isolated, executable Virtual Machines
that run research and analysis pipelines in sandboxed contexts.
"""

from .prose_vm import ProseVM, VMConfig, ExecutionContext, VMState
from .sandbox import Sandbox, SandboxPolicy, ResourceLimits

__all__ = [
    "ProseVM",
    "VMConfig",
    "ExecutionContext",
    "VMState",
    "Sandbox",
    "SandboxPolicy",
    "ResourceLimits",
]
