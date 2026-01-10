"""
Sandbox Environment for OpenProse VMs

Provides isolated execution contexts with configurable resource limits
and security policies for running prompt pipelines.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable


class IsolationLevel(Enum):
    """Level of isolation for the sandbox."""

    NONE = "none"              # No isolation (development only)
    PROCESS = "process"        # Separate process
    CONTAINER = "container"    # Docker/Podman container
    VM = "vm"                  # Full VM isolation


@dataclass
class ResourceLimits:
    """Resource constraints for sandbox execution."""

    max_memory_mb: int = 512
    max_cpu_percent: float = 50.0
    max_execution_time_seconds: int = 300
    max_output_tokens: int = 100000
    max_input_tokens: int = 50000
    max_concurrent_prompts: int = 5
    max_api_calls_per_minute: int = 60

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_percent": self.max_cpu_percent,
            "max_execution_time_seconds": self.max_execution_time_seconds,
            "max_output_tokens": self.max_output_tokens,
            "max_input_tokens": self.max_input_tokens,
            "max_concurrent_prompts": self.max_concurrent_prompts,
            "max_api_calls_per_minute": self.max_api_calls_per_minute,
        }


@dataclass
class SandboxPolicy:
    """
    Security and access policy for sandbox execution.

    Controls what operations are permitted within the sandbox.
    """

    allow_network: bool = True
    allow_file_read: bool = False
    allow_file_write: bool = False
    allowed_domains: list[str] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)
    allow_shell_execution: bool = False
    allow_code_execution: bool = False
    require_human_approval: bool = False
    audit_logging: bool = True

    def validate_domain(self, domain: str) -> bool:
        """Check if a domain is allowed by policy."""
        if domain in self.blocked_domains:
            return False
        if self.allowed_domains and domain not in self.allowed_domains:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "allow_network": self.allow_network,
            "allow_file_read": self.allow_file_read,
            "allow_file_write": self.allow_file_write,
            "allowed_domains": self.allowed_domains,
            "blocked_domains": self.blocked_domains,
            "allow_shell_execution": self.allow_shell_execution,
            "allow_code_execution": self.allow_code_execution,
            "require_human_approval": self.require_human_approval,
            "audit_logging": self.audit_logging,
        }


@dataclass
class SandboxMetrics:
    """Runtime metrics for sandbox execution."""

    total_prompts_executed: int = 0
    total_tokens_used: int = 0
    total_api_calls: int = 0
    execution_time_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    errors_encountered: int = 0
    warnings_issued: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_prompts_executed": self.total_prompts_executed,
            "total_tokens_used": self.total_tokens_used,
            "total_api_calls": self.total_api_calls,
            "execution_time_seconds": self.execution_time_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "errors_encountered": self.errors_encountered,
            "warnings_issued": self.warnings_issued,
        }


@dataclass
class AuditEntry:
    """Single audit log entry."""

    timestamp: datetime
    action: str
    details: dict[str, Any]
    success: bool
    error: str | None = None


class Sandbox:
    """
    Isolated execution environment for OpenProse VMs.

    The sandbox provides:
    - Resource limiting (memory, CPU, time)
    - Security policy enforcement
    - Audit logging
    - Metrics collection
    - State isolation between executions
    """

    def __init__(
        self,
        sandbox_id: str | None = None,
        limits: ResourceLimits | None = None,
        policy: SandboxPolicy | None = None,
        isolation_level: IsolationLevel = IsolationLevel.PROCESS,
    ):
        self.sandbox_id = sandbox_id or f"sandbox_{uuid.uuid4().hex[:12]}"
        self.limits = limits or ResourceLimits()
        self.policy = policy or SandboxPolicy()
        self.isolation_level = isolation_level

        self._metrics = SandboxMetrics()
        self._audit_log: list[AuditEntry] = []
        self._state: dict[str, Any] = {}
        self._start_time: datetime | None = None
        self._api_call_times: list[datetime] = []
        self._active = False

        # Callbacks for policy enforcement
        self._approval_callback: Callable[[str], bool] | None = None

    @property
    def metrics(self) -> SandboxMetrics:
        return self._metrics

    @property
    def audit_log(self) -> list[AuditEntry]:
        return self._audit_log.copy()

    @property
    def is_active(self) -> bool:
        return self._active

    def set_approval_callback(self, callback: Callable[[str], bool]) -> None:
        """Set callback for human approval requests."""
        self._approval_callback = callback

    def _log_audit(
        self,
        action: str,
        details: dict[str, Any],
        success: bool,
        error: str | None = None,
    ) -> None:
        """Add entry to audit log."""
        if self.policy.audit_logging:
            self._audit_log.append(
                AuditEntry(
                    timestamp=datetime.now(),
                    action=action,
                    details=details,
                    success=success,
                    error=error,
                )
            )

    def _check_rate_limit(self) -> bool:
        """Check if we're within API rate limits."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        # Clean old entries
        self._api_call_times = [t for t in self._api_call_times if t > minute_ago]

        return len(self._api_call_times) < self.limits.max_api_calls_per_minute

    def _check_time_limit(self) -> bool:
        """Check if we're within execution time limits."""
        if self._start_time is None:
            return True

        elapsed = (datetime.now() - self._start_time).total_seconds()
        return elapsed < self.limits.max_execution_time_seconds

    async def enter(self) -> None:
        """Enter the sandbox context."""
        self._active = True
        self._start_time = datetime.now()
        self._log_audit("sandbox_enter", {"sandbox_id": self.sandbox_id}, True)

    async def exit(self) -> None:
        """Exit the sandbox context."""
        if self._start_time:
            self._metrics.execution_time_seconds = (
                datetime.now() - self._start_time
            ).total_seconds()

        self._active = False
        self._log_audit(
            "sandbox_exit",
            {"sandbox_id": self.sandbox_id, "metrics": self._metrics.to_dict()},
            True,
        )

    async def __aenter__(self) -> "Sandbox":
        await self.enter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.exit()

    async def execute_prompt(
        self,
        prompt: str,
        executor: Callable[[str], Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a prompt within sandbox constraints.

        Args:
            prompt: The rendered prompt to execute
            executor: Function that actually runs the prompt (e.g., LLM API call)
            **kwargs: Additional arguments for the executor

        Returns:
            Execution result with output and metadata
        """
        if not self._active:
            raise RuntimeError("Sandbox is not active. Use 'async with' context.")

        # Check limits
        if not self._check_time_limit():
            self._log_audit(
                "prompt_execution",
                {"prompt_length": len(prompt)},
                False,
                "Execution time limit exceeded",
            )
            raise TimeoutError("Sandbox execution time limit exceeded")

        if not self._check_rate_limit():
            self._log_audit(
                "prompt_execution",
                {"prompt_length": len(prompt)},
                False,
                "Rate limit exceeded",
            )
            raise RuntimeError("API rate limit exceeded")

        # Check token limits
        estimated_input_tokens = len(prompt) // 4  # Rough estimate
        if estimated_input_tokens > self.limits.max_input_tokens:
            self._log_audit(
                "prompt_execution",
                {"estimated_tokens": estimated_input_tokens},
                False,
                "Input token limit exceeded",
            )
            raise ValueError(
                f"Input exceeds token limit: {estimated_input_tokens} > {self.limits.max_input_tokens}"
            )

        # Request approval if required
        if self.policy.require_human_approval:
            if self._approval_callback is None:
                raise RuntimeError("Human approval required but no callback set")

            if not self._approval_callback(prompt):
                self._log_audit(
                    "prompt_execution",
                    {"prompt_length": len(prompt)},
                    False,
                    "Human approval denied",
                )
                raise PermissionError("Human approval denied for prompt execution")

        # Execute with timeout
        try:
            self._api_call_times.append(datetime.now())
            self._metrics.total_api_calls += 1

            result = await asyncio.wait_for(
                asyncio.to_thread(executor, prompt, **kwargs),
                timeout=self.limits.max_execution_time_seconds,
            )

            self._metrics.total_prompts_executed += 1

            # Estimate output tokens
            if isinstance(result, str):
                output_tokens = len(result) // 4
                self._metrics.total_tokens_used += estimated_input_tokens + output_tokens

            self._log_audit(
                "prompt_execution",
                {
                    "prompt_length": len(prompt),
                    "success": True,
                },
                True,
            )

            return {
                "success": True,
                "output": result,
                "metrics": {
                    "input_tokens": estimated_input_tokens,
                    "execution_time": (datetime.now() - self._start_time).total_seconds()
                    if self._start_time
                    else 0,
                },
            }

        except asyncio.TimeoutError:
            self._metrics.errors_encountered += 1
            self._log_audit(
                "prompt_execution",
                {"prompt_length": len(prompt)},
                False,
                "Execution timeout",
            )
            raise TimeoutError("Prompt execution timed out")

        except Exception as e:
            self._metrics.errors_encountered += 1
            self._log_audit(
                "prompt_execution",
                {"prompt_length": len(prompt)},
                False,
                str(e),
            )
            raise

    def get_state(self, key: str, default: Any = None) -> Any:
        """Get value from sandbox state."""
        return self._state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """Set value in sandbox state."""
        self._state[key] = value

    def clear_state(self) -> None:
        """Clear all sandbox state."""
        self._state.clear()

    def export_audit_log(self) -> list[dict[str, Any]]:
        """Export audit log as serializable dictionaries."""
        return [
            {
                "timestamp": entry.timestamp.isoformat(),
                "action": entry.action,
                "details": entry.details,
                "success": entry.success,
                "error": entry.error,
            }
            for entry in self._audit_log
        ]
