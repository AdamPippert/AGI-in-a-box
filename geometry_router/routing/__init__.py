"""Geometry-aware routing implementation."""

from .router import (
    GeometryAwareRouter,
    RoutingDecision,
    RoutingContext,
    RecursiveRoutingController,
    sinkhorn_knopp
)

__all__ = [
    "GeometryAwareRouter",
    "RoutingDecision",
    "RoutingContext",
    "RecursiveRoutingController",
    "sinkhorn_knopp"
]
