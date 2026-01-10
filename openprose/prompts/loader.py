"""
Prompt Collection Loader

Handles loading, parsing, and structuring prompt collections from various formats.
"""

from __future__ import annotations

import json
import yaml
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class PromptCategory(Enum):
    """Categories for organizing prompts by function."""

    ANALYSIS = "analysis"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    COMPARISON = "comparison"
    TRANSFORMATION = "transformation"
    STRESS_TEST = "stress_test"
    META = "meta"


class PromptIntent(Enum):
    """The intended effect of a prompt on input content."""

    EXTRACT = "extract"      # Pull out specific information
    EVALUATE = "evaluate"    # Assess quality or validity
    TRANSFORM = "transform"  # Change format or structure
    GENERATE = "generate"    # Create new content
    COMPRESS = "compress"    # Reduce to essentials
    EXPAND = "expand"        # Add detail or explanation


@dataclass
class Prompt:
    """
    A single prompt with metadata for VM execution.

    Attributes:
        id: Unique identifier for the prompt
        name: Human-readable name
        template: The actual prompt text (may contain {placeholders})
        category: Functional category
        intent: What the prompt does to content
        description: Explanation of purpose and use
        tags: Searchable tags
        input_schema: Expected input structure
        output_schema: Expected output structure
        chain_compatible: Whether output can feed into another prompt
    """

    id: str
    name: str
    template: str
    category: PromptCategory
    intent: PromptIntent
    description: str = ""
    tags: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    chain_compatible: bool = True

    def render(self, **kwargs: Any) -> str:
        """Render the prompt template with provided values."""
        return self.template.format(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "template": self.template,
            "category": self.category.value,
            "intent": self.intent.value,
            "description": self.description,
            "tags": self.tags,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "chain_compatible": self.chain_compatible,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Prompt:
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            template=data["template"],
            category=PromptCategory(data["category"]),
            intent=PromptIntent(data["intent"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            input_schema=data.get("input_schema"),
            output_schema=data.get("output_schema"),
            chain_compatible=data.get("chain_compatible", True),
        )


@dataclass
class CollectionMetadata:
    """Metadata about a prompt collection."""

    author: str
    date: str
    source: str
    description: str
    version: str = "1.0.0"
    license: str = "MIT"
    tags: list[str] = field(default_factory=list)


@dataclass
class PromptCollection:
    """
    A curated collection of related prompts.

    Collections can be loaded from YAML/JSON files and transformed
    into VM pipelines for execution.
    """

    id: str
    name: str
    metadata: CollectionMetadata
    prompts: dict[str, Prompt] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.prompts)

    def __iter__(self):
        return iter(self.prompts.values())

    def __getitem__(self, prompt_id: str) -> Prompt:
        return self.prompts[prompt_id]

    def get(self, prompt_id: str, default: Prompt | None = None) -> Prompt | None:
        return self.prompts.get(prompt_id, default)

    def add_prompt(self, prompt: Prompt) -> None:
        """Add a prompt to the collection."""
        self.prompts[prompt.id] = prompt

    def filter_by_category(self, category: PromptCategory) -> list[Prompt]:
        """Get all prompts in a category."""
        return [p for p in self.prompts.values() if p.category == category]

    def filter_by_intent(self, intent: PromptIntent) -> list[Prompt]:
        """Get all prompts with a specific intent."""
        return [p for p in self.prompts.values() if p.intent == intent]

    def filter_by_tag(self, tag: str) -> list[Prompt]:
        """Get all prompts with a specific tag."""
        return [p for p in self.prompts.values() if tag in p.tags]

    def to_dict(self) -> dict[str, Any]:
        """Serialize collection to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "metadata": {
                "author": self.metadata.author,
                "date": self.metadata.date,
                "source": self.metadata.source,
                "description": self.metadata.description,
                "version": self.metadata.version,
                "license": self.metadata.license,
                "tags": self.metadata.tags,
            },
            "prompts": {pid: p.to_dict() for pid, p in self.prompts.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromptCollection:
        """Deserialize from dictionary."""
        metadata = CollectionMetadata(
            author=data["metadata"]["author"],
            date=data["metadata"]["date"],
            source=data["metadata"]["source"],
            description=data["metadata"]["description"],
            version=data["metadata"].get("version", "1.0.0"),
            license=data["metadata"].get("license", "MIT"),
            tags=data["metadata"].get("tags", []),
        )

        collection = cls(
            id=data["id"],
            name=data["name"],
            metadata=metadata,
        )

        for prompt_data in data.get("prompts", {}).values():
            collection.add_prompt(Prompt.from_dict(prompt_data))

        return collection

    def save(self, path: Path | str, format: str = "yaml") -> None:
        """Save collection to file."""
        path = Path(path)
        data = self.to_dict()

        with open(path, "w") as f:
            if format == "yaml":
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            else:
                json.dump(data, f, indent=2)


def load_collection(path: Path | str) -> PromptCollection:
    """
    Load a prompt collection from a YAML or JSON file.

    Args:
        path: Path to the collection file

    Returns:
        Loaded PromptCollection instance
    """
    path = Path(path)

    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    return PromptCollection.from_dict(data)
