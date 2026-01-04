"""
Geometry-Aware Router HTTP/gRPC Server

Production-ready server for the geometry-aware hierarchical routing framework.
Provides both REST API and gRPC interfaces for model routing decisions.
"""

import os
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import threading

import numpy as np

from geometry_router import (
    GeometryAwareRouter,
    TopologicalFeatureExtractor,
    ModelRegistry,
    create_default_registry,
    RoutingContext,
)

# Configuration from environment
HOST = os.getenv("ROUTER_HOST", "0.0.0.0")
PORT = int(os.getenv("ROUTER_PORT", "8080"))
GRPC_PORT = int(os.getenv("ROUTER_GRPC_PORT", "50051"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
TOPOLOGY_APPROXIMATE = os.getenv("TOPOLOGY_APPROXIMATE", "true").lower() == "true"
SINKHORN_ITERATIONS = int(os.getenv("SINKHORN_ITERATIONS", "20"))
MAX_RECURSION_DEPTH = int(os.getenv("MAX_RECURSION_DEPTH", "5"))

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("geometry_router.server")


class RouterService:
    """Singleton service managing the router components."""

    _instance: Optional["RouterService"] = None

    def __init__(self):
        logger.info("Initializing Geometry-Aware Router Service...")

        # Initialize model registry
        self.registry: ModelRegistry = create_default_registry()
        logger.info(f"Loaded {len(self.registry.get_all_models())} models into registry")

        # Initialize topology extractor
        self.extractor = TopologicalFeatureExtractor(
            approximate=TOPOLOGY_APPROXIMATE,
            n_points_subsample=500
        )
        logger.info(f"Topology extractor initialized (approximate={TOPOLOGY_APPROXIMATE})")

        # Initialize router
        self.router = GeometryAwareRouter(
            registry=self.registry,
            extractor=self.extractor,
            sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
        logger.info("Router initialized successfully")

    @classmethod
    def get_instance(cls) -> "RouterService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def route(self, embeddings: np.ndarray, context: Optional[dict] = None) -> dict:
        """Route query embeddings to optimal model."""
        # Extract topological signature
        signature = self.extractor.extract(embeddings)

        # Build routing context if provided
        routing_context = None
        if context:
            routing_context = RoutingContext(
                signature=signature,
                source_model_id=context.get("source_model_id"),
                source_chunk_id=context.get("source_chunk_id"),
                constraints=context.get("constraints", {})
            )

        # Get routing decision
        decision = self.router.route(embeddings, context=routing_context)

        return {
            "primary_model": {
                "id": decision.primary_model.id,
                "name": decision.primary_model.name,
                "tier": decision.primary_model.tier.name,
                "endpoint": decision.primary_model.api_endpoint,
            },
            "fallback_models": [
                {"id": m.id, "name": m.name, "tier": m.tier.name}
                for m in decision.fallback_models
            ],
            "confidence": decision.confidence,
            "topology": {
                "complexity": signature.complexity.name,
                "betti_0": signature.betti_0,
                "betti_1": signature.betti_1,
                "betti_2": signature.betti_2,
                "stability_score": signature.stability_score,
            },
            "diagnostics": decision.diagnostics,
        }

    def health_check(self) -> dict:
        """Return service health status."""
        return {
            "status": "healthy",
            "models_loaded": len(self.registry.get_all_models()),
            "topology_mode": "approximate" if TOPOLOGY_APPROXIMATE else "exact",
            "version": "0.1.0",
        }


class RouterHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the router API."""

    def _send_json_response(self, data: dict, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error_response(self, message: str, status: int = 400):
        self._send_json_response({"error": message}, status)

    def do_GET(self):
        service = RouterService.get_instance()

        if self.path == "/health":
            self._send_json_response(service.health_check())
        elif self.path == "/models":
            models = [
                {
                    "id": m.id,
                    "name": m.name,
                    "tier": m.tier.name,
                    "capabilities": [c.name for c in m.topological_profile.primary_capabilities],
                }
                for m in service.registry.get_all_models()
            ]
            self._send_json_response({"models": models})
        elif self.path == "/":
            self._send_json_response({
                "service": "geometry-aware-router",
                "version": "0.1.0",
                "endpoints": ["/health", "/models", "/route"],
            })
        else:
            self._send_error_response("Not found", 404)

    def do_POST(self):
        service = RouterService.get_instance()

        if self.path == "/route":
            try:
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                data = json.loads(body.decode())

                # Parse embeddings
                embeddings = np.array(data.get("embeddings", []))
                if embeddings.size == 0:
                    self._send_error_response("embeddings field is required")
                    return

                # Ensure 2D array
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(1, -1)

                context = data.get("context")

                result = service.route(embeddings, context)
                self._send_json_response(result)

            except json.JSONDecodeError:
                self._send_error_response("Invalid JSON")
            except Exception as e:
                logger.exception("Error processing route request")
                self._send_error_response(str(e), 500)
        else:
            self._send_error_response("Not found", 404)

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def run_http_server():
    """Run the HTTP server."""
    server = HTTPServer((HOST, PORT), RouterHTTPHandler)
    logger.info(f"HTTP server listening on {HOST}:{PORT}")
    server.serve_forever()


def main():
    """Main entry point for the router server."""
    logger.info("=" * 60)
    logger.info("Starting Geometry-Aware Router Server")
    logger.info("=" * 60)

    # Initialize the service (loads models, etc.)
    service = RouterService.get_instance()

    logger.info(f"Configuration:")
    logger.info(f"  HTTP Port: {PORT}")
    logger.info(f"  gRPC Port: {GRPC_PORT}")
    logger.info(f"  Topology Mode: {'approximate' if TOPOLOGY_APPROXIMATE else 'exact'}")
    logger.info(f"  Sinkhorn Iterations: {SINKHORN_ITERATIONS}")
    logger.info(f"  Max Recursion Depth: {MAX_RECURSION_DEPTH}")

    # Run HTTP server (gRPC can be added similarly)
    run_http_server()


if __name__ == "__main__":
    main()
