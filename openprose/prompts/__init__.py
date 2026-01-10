"""
OpenProse Prompt Collections

Structured prompt libraries that can be loaded and transformed into VM pipelines.
"""

from .loader import PromptCollection, Prompt, load_collection

__all__ = ["PromptCollection", "Prompt", "load_collection"]
