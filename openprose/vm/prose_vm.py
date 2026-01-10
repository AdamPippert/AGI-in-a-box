"""
OpenProse Virtual Machine

The ProseVM translates prompt collections into executable, sandboxed
research pipelines. Each VM instance is an isolated environment for
running prompt-based analysis workflows.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable

from ..prompts.loader import Prompt, PromptCollection
from .sandbox import Sandbox, SandboxPolicy, ResourceLimits, IsolationLevel


class VMState(Enum):
    """Lifecycle states for a ProseVM."""

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class VMConfig:
    """Configuration for a ProseVM instance."""

    name: str = "unnamed_vm"
    description: str = ""
    isolation_level: IsolationLevel = IsolationLevel.PROCESS
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    sandbox_policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    auto_checkpoint: bool = True
    checkpoint_interval: int = 5  # After every N prompts
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "isolation_level": self.isolation_level.value,
            "resource_limits": self.resource_limits.to_dict(),
            "sandbox_policy": self.sandbox_policy.to_dict(),
            "auto_checkpoint": self.auto_checkpoint,
            "checkpoint_interval": self.checkpoint_interval,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
        }


@dataclass
class ExecutionContext:
    """
    Context passed through prompt executions.

    Maintains state and data flow between prompts in a pipeline.
    """

    vm_id: str
    execution_id: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_input(self, key: str, default: Any = None) -> Any:
        """Get an input value."""
        return self.inputs.get(key, default)

    def set_output(self, key: str, value: Any) -> None:
        """Set an output value."""
        self.outputs[key] = value

    def get_output(self, key: str, default: Any = None) -> Any:
        """Get an output value."""
        return self.outputs.get(key, default)

    def set_variable(self, key: str, value: Any) -> None:
        """Set a variable for use in prompts."""
        self.variables[key] = value

    def get_variable(self, key: str, default: Any = None) -> Any:
        """Get a variable value."""
        return self.variables.get(key, default)

    def add_to_history(self, entry: dict[str, Any]) -> None:
        """Add an entry to execution history."""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            **entry,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "vm_id": self.vm_id,
            "execution_id": self.execution_id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "variables": self.variables,
            "history": self.history,
            "metadata": self.metadata,
        }


@dataclass
class Checkpoint:
    """Snapshot of VM state at a point in time."""

    checkpoint_id: str
    vm_id: str
    timestamp: datetime
    state: VMState
    context: ExecutionContext
    prompts_completed: int
    sandbox_state: dict[str, Any]


class ProseVM:
    """
    OpenProse Virtual Machine

    A ProseVM is an isolated execution environment that transforms a
    PromptCollection into a runnable research pipeline. Each VM:

    - Maintains isolated state
    - Enforces resource limits
    - Provides checkpointing and recovery
    - Manages data flow between prompts
    - Collects execution metrics

    Example:
        ```python
        collection = load_collection("viral_research_tools.yaml")
        vm = ProseVM.from_collection(collection)

        async with vm:
            result = await vm.execute_prompt(
                "contradictions_finder",
                content="Your document here..."
            )
        ```
    """

    def __init__(
        self,
        vm_id: str | None = None,
        config: VMConfig | None = None,
    ):
        self.vm_id = vm_id or f"vm_{uuid.uuid4().hex[:12]}"
        self.config = config or VMConfig()

        self._state = VMState.INITIALIZING
        self._collection: PromptCollection | None = None
        self._sandbox: Sandbox | None = None
        self._context: ExecutionContext | None = None
        self._checkpoints: list[Checkpoint] = []
        self._executor: Callable[[str], Any] | None = None
        self._prompts_executed = 0

        self._created_at = datetime.now()
        self._started_at: datetime | None = None
        self._completed_at: datetime | None = None

    @property
    def state(self) -> VMState:
        return self._state

    @property
    def collection(self) -> PromptCollection | None:
        return self._collection

    @property
    def context(self) -> ExecutionContext | None:
        return self._context

    @classmethod
    def from_collection(
        cls,
        collection: PromptCollection,
        config: VMConfig | None = None,
    ) -> "ProseVM":
        """
        Create a VM from a prompt collection.

        Args:
            collection: The PromptCollection to load
            config: Optional VM configuration

        Returns:
            Configured ProseVM instance
        """
        if config is None:
            config = VMConfig(
                name=f"vm_{collection.id}",
                description=f"VM for {collection.name}",
            )

        vm = cls(config=config)
        vm._collection = collection
        vm._state = VMState.READY

        return vm

    def set_executor(self, executor: Callable[[str], Any]) -> None:
        """
        Set the function that executes prompts.

        The executor is typically an LLM API call function.

        Args:
            executor: Function that takes a prompt string and returns a response
        """
        self._executor = executor

    async def start(self, inputs: dict[str, Any] | None = None) -> None:
        """
        Start the VM and initialize execution context.

        Args:
            inputs: Initial input data for the execution
        """
        if self._state not in (VMState.READY, VMState.PAUSED):
            raise RuntimeError(f"Cannot start VM in state: {self._state}")

        if self._collection is None:
            raise RuntimeError("No collection loaded. Use from_collection() first.")

        # Create sandbox
        self._sandbox = Sandbox(
            sandbox_id=f"{self.vm_id}_sandbox",
            limits=self.config.resource_limits,
            policy=self.config.sandbox_policy,
            isolation_level=self.config.isolation_level,
        )

        # Create execution context
        self._context = ExecutionContext(
            vm_id=self.vm_id,
            execution_id=f"exec_{uuid.uuid4().hex[:8]}",
            inputs=inputs or {},
        )

        await self._sandbox.enter()
        self._state = VMState.RUNNING
        self._started_at = datetime.now()

        self._context.add_to_history({
            "event": "vm_started",
            "config": self.config.to_dict(),
        })

    async def stop(self) -> None:
        """Stop the VM and cleanup resources."""
        if self._sandbox:
            await self._sandbox.exit()

        if self._state == VMState.RUNNING:
            self._state = VMState.COMPLETED
            self._completed_at = datetime.now()

        if self._context:
            self._context.add_to_history({
                "event": "vm_stopped",
                "prompts_executed": self._prompts_executed,
            })

    async def __aenter__(self) -> "ProseVM":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self._state = VMState.FAILED
        await self.stop()

    async def execute_prompt(
        self,
        prompt_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Execute a single prompt from the collection.

        Args:
            prompt_id: ID of the prompt to execute
            **kwargs: Variables to substitute in the prompt template

        Returns:
            Execution result with output and metadata
        """
        if self._state != VMState.RUNNING:
            raise RuntimeError(f"VM is not running (state: {self._state})")

        if self._collection is None:
            raise RuntimeError("No collection loaded")

        if self._executor is None:
            raise RuntimeError("No executor set. Call set_executor() first.")

        if self._sandbox is None or self._context is None:
            raise RuntimeError("VM not properly initialized")

        # Get prompt
        prompt = self._collection.get(prompt_id)
        if prompt is None:
            raise KeyError(f"Prompt not found: {prompt_id}")

        # Merge context variables with provided kwargs
        render_vars = {**self._context.variables, **kwargs}

        # Render prompt
        try:
            rendered = prompt.render(**render_vars)
        except KeyError as e:
            raise ValueError(f"Missing required variable for prompt: {e}")

        # Execute with retries
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                result = await self._sandbox.execute_prompt(
                    rendered,
                    self._executor,
                )

                self._prompts_executed += 1

                # Store in context
                self._context.set_output(prompt_id, result["output"])
                self._context.add_to_history({
                    "event": "prompt_executed",
                    "prompt_id": prompt_id,
                    "attempt": attempt + 1,
                    "success": True,
                })

                # Auto checkpoint
                if (
                    self.config.auto_checkpoint
                    and self._prompts_executed % self.config.checkpoint_interval == 0
                ):
                    await self.checkpoint()

                return result

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(
                        self.config.retry_delay_seconds * (attempt + 1)
                    )

        # All retries failed
        self._context.add_to_history({
            "event": "prompt_failed",
            "prompt_id": prompt_id,
            "error": str(last_error),
        })
        raise last_error  # type: ignore

    async def execute_chain(
        self,
        prompt_ids: list[str],
        initial_content: str,
        content_key: str = "content",
    ) -> dict[str, Any]:
        """
        Execute a chain of prompts, passing output as input to the next.

        Args:
            prompt_ids: List of prompt IDs to execute in order
            initial_content: Starting content for the first prompt
            content_key: Key name to use for passing content between prompts

        Returns:
            Final result and intermediate outputs
        """
        results = []
        current_content = initial_content

        for prompt_id in prompt_ids:
            result = await self.execute_prompt(prompt_id, **{content_key: current_content})
            results.append({
                "prompt_id": prompt_id,
                "result": result,
            })

            # Use output as next input
            if isinstance(result.get("output"), str):
                current_content = result["output"]

        return {
            "success": True,
            "chain": results,
            "final_output": results[-1]["result"]["output"] if results else None,
        }

    async def checkpoint(self) -> Checkpoint:
        """
        Create a checkpoint of current VM state.

        Returns:
            Checkpoint object that can be used for recovery
        """
        if self._context is None or self._sandbox is None:
            raise RuntimeError("VM not properly initialized")

        checkpoint = Checkpoint(
            checkpoint_id=f"ckpt_{uuid.uuid4().hex[:8]}",
            vm_id=self.vm_id,
            timestamp=datetime.now(),
            state=self._state,
            context=ExecutionContext(
                vm_id=self._context.vm_id,
                execution_id=self._context.execution_id,
                inputs=self._context.inputs.copy(),
                outputs=self._context.outputs.copy(),
                variables=self._context.variables.copy(),
                history=self._context.history.copy(),
                metadata=self._context.metadata.copy(),
            ),
            prompts_completed=self._prompts_executed,
            sandbox_state={"metrics": self._sandbox.metrics.to_dict()},
        )

        self._checkpoints.append(checkpoint)

        self._context.add_to_history({
            "event": "checkpoint_created",
            "checkpoint_id": checkpoint.checkpoint_id,
        })

        return checkpoint

    async def restore(self, checkpoint: Checkpoint) -> None:
        """
        Restore VM state from a checkpoint.

        Args:
            checkpoint: Checkpoint to restore from
        """
        if checkpoint.vm_id != self.vm_id:
            raise ValueError("Checkpoint belongs to different VM")

        self._context = checkpoint.context
        self._state = checkpoint.state
        self._prompts_executed = checkpoint.prompts_completed

        if self._context:
            self._context.add_to_history({
                "event": "checkpoint_restored",
                "checkpoint_id": checkpoint.checkpoint_id,
            })

    def list_prompts(self) -> list[dict[str, Any]]:
        """List all available prompts in the loaded collection."""
        if self._collection is None:
            return []

        return [
            {
                "id": p.id,
                "name": p.name,
                "category": p.category.value,
                "intent": p.intent.value,
                "description": p.description,
            }
            for p in self._collection
        ]

    def get_metrics(self) -> dict[str, Any]:
        """Get current execution metrics."""
        metrics = {
            "vm_id": self.vm_id,
            "state": self._state.value,
            "prompts_executed": self._prompts_executed,
            "checkpoints": len(self._checkpoints),
            "created_at": self._created_at.isoformat(),
        }

        if self._started_at:
            metrics["started_at"] = self._started_at.isoformat()

        if self._completed_at:
            metrics["completed_at"] = self._completed_at.isoformat()

        if self._sandbox:
            metrics["sandbox"] = self._sandbox.metrics.to_dict()

        return metrics

    def export_state(self) -> dict[str, Any]:
        """Export complete VM state for serialization."""
        return {
            "vm_id": self.vm_id,
            "config": self.config.to_dict(),
            "state": self._state.value,
            "collection_id": self._collection.id if self._collection else None,
            "context": self._context.to_dict() if self._context else None,
            "metrics": self.get_metrics(),
            "checkpoints": [
                {
                    "checkpoint_id": c.checkpoint_id,
                    "timestamp": c.timestamp.isoformat(),
                    "prompts_completed": c.prompts_completed,
                }
                for c in self._checkpoints
            ],
        }
