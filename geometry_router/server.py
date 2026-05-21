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
from pathlib import Path
import threading

import numpy as np

from geometry_router.adaptation import (
    AdaptiveLayerConfig,
    AdaptiveLayerManager,
    DefaultResidualSignalProvider,
    DictCorrectionInjector,
    DotProductMemoryMatcher,
    FileBackedKeywordRetrievalProvider,
    FileMemorySignatureStore,
    FrequencyQualityEligibilityPolicy,
    InMemoryAdaptationCollector,
    JsonlAdapterTrainingExportService,
    SimpleRetrievalAugmentor,
    ThresholdRetrievalTriggerPolicy,
)

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

# Adaptive layer configuration
LAYER1_ENABLED = os.getenv("ADAPT_LAYER1_ENABLED", "false").lower() == "true"
LAYER2_ENABLED = os.getenv("ADAPT_LAYER2_ENABLED", "false").lower() == "true"
LAYER3_ENABLED = os.getenv("ADAPT_LAYER3_ENABLED", "false").lower() == "true"
ADAPT_RESIDUAL_THRESHOLD = float(os.getenv("ADAPT_RESIDUAL_THRESHOLD", "0.35"))
ADAPT_LOW_CONFIDENCE_THRESHOLD = float(os.getenv("ADAPT_LOW_CONFIDENCE_THRESHOLD", "0.45"))
ADAPT_LOW_MARGIN_THRESHOLD = float(os.getenv("ADAPT_LOW_MARGIN_THRESHOLD", "0.08"))
ADAPT_MEMORY_SIMILARITY_THRESHOLD = float(os.getenv("ADAPT_MEMORY_SIMILARITY_THRESHOLD", "0.75"))
ADAPT_MIN_OCCURRENCES = int(os.getenv("ADAPT_MIN_OCCURRENCES", "3"))
ADAPT_MIN_AVG_QUALITY = float(os.getenv("ADAPT_MIN_AVG_QUALITY", "0.7"))
ADAPT_MEMORY_PATH = os.getenv("ADAPT_MEMORY_PATH", "./data/adaptation/residual_memory.jsonl")
ADAPT_EXPORT_DIR = os.getenv("ADAPT_EXPORT_DIR", "./data/adaptation/exports")
ADAPT_RETRIEVAL_CORPUS_PATH = os.getenv("ADAPT_RETRIEVAL_CORPUS_PATH", "./data/adaptation/retrieval_corpus.json")

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
        logger.info(f"Loaded {len(self.registry.all_models)} models into registry")

        # Initialize topology extractor
        self.extractor = TopologicalFeatureExtractor(
            use_approximate=TOPOLOGY_APPROXIMATE,
            subsample_size=500
        )
        logger.info(f"Topology extractor initialized (approximate={TOPOLOGY_APPROXIMATE})")

        # Initialize router
        self.router = GeometryAwareRouter(
            registry=self.registry,
            extractor=self.extractor,
            sinkhorn_iterations=SINKHORN_ITERATIONS,
        )
        logger.info("Router initialized successfully")

        self.adaptive_manager = AdaptiveLayerManager(
            config=AdaptiveLayerConfig(
                layer1_enabled=LAYER1_ENABLED,
                layer2_enabled=LAYER2_ENABLED,
                layer3_enabled=LAYER3_ENABLED,
            ),
            signal_provider=DefaultResidualSignalProvider(
                low_confidence_threshold=ADAPT_LOW_CONFIDENCE_THRESHOLD,
                low_margin_threshold=ADAPT_LOW_MARGIN_THRESHOLD,
            ),
            retrieval_policy=ThresholdRetrievalTriggerPolicy(
                residual_threshold=ADAPT_RESIDUAL_THRESHOLD,
            ),
            retrieval_augmentor=SimpleRetrievalAugmentor(
                provider=FileBackedKeywordRetrievalProvider(
                    corpus_path=Path(ADAPT_RETRIEVAL_CORPUS_PATH),
                    max_results=3,
                )
            ),
            memory_store=FileMemorySignatureStore(path=Path(ADAPT_MEMORY_PATH)),
            memory_matcher=DotProductMemoryMatcher(
                similarity_threshold=ADAPT_MEMORY_SIMILARITY_THRESHOLD,
            ),
            correction_injector=DictCorrectionInjector(),
            adaptation_collector=InMemoryAdaptationCollector(state={}),
            adaptation_policy=FrequencyQualityEligibilityPolicy(
                min_occurrences=ADAPT_MIN_OCCURRENCES,
                min_avg_quality=ADAPT_MIN_AVG_QUALITY,
            ),
            export_service=JsonlAdapterTrainingExportService(output_dir=Path(ADAPT_EXPORT_DIR)),
        )
        logger.info(
            "Adaptive layers configured layer1=%s layer2=%s layer3=%s",
            LAYER1_ENABLED,
            LAYER2_ENABLED,
            LAYER3_ENABLED,
        )

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
                query_signature=signature,
                source_model_id=context.get("source_model_id"),
                source_chunk_id=context.get("source_chunk_id"),
                required_capabilities=set(context.get("required_capabilities", []))
            )

        # Get routing decision
        decision = self.router.route(embeddings, context=routing_context)

        adaptive = self.adaptive_manager.process(
            query=context.get("query", "") if context else "",
            workflow_type=context.get("workflow_type", "routing") if context else "routing",
            route_confidence=decision.confidence,
            routing_scores=decision.routing_scores,
            overlap_resolution_used=decision.overlap_resolution_used,
            stability_score=signature.stability_score,
            route_metadata={
                "primary_model": decision.primary_model_id,
                "fallback_count": str(len(decision.fallback_model_ids)),
            },
        )

        primary_model = self.registry.get(decision.primary_model_id)
        fallback_models = [self.registry.get(mid) for mid in decision.fallback_model_ids]

        return {
            "primary_model": {
                "id": primary_model.model_id if primary_model else decision.primary_model_id,
                "name": primary_model.display_name if primary_model else decision.primary_model_id,
                "tier": primary_model.tier.name if primary_model else "UNKNOWN",
                "endpoint": primary_model.api_endpoint if primary_model else None,
            },
            "fallback_models": [
                {
                    "id": m.model_id,
                    "name": m.display_name,
                    "tier": m.tier.name,
                }
                for m in fallback_models
                if m is not None
            ],
            "confidence": decision.confidence,
            "topology": {
                "complexity": signature.complexity.name,
                "betti_0": signature.betti_profile[0],
                "betti_1": signature.betti_profile[1],
                "betti_2": signature.betti_profile[2],
                "stability_score": signature.stability_score,
            },
            "diagnostics": {
                "routing_scores": decision.routing_scores,
                "overlap_resolution_used": decision.overlap_resolution_used,
                "hierarchy_distances": decision.hierarchy_distances,
                "topological_distances": decision.topological_distances,
            },
            "adaptive": {
                "retrieval_triggered": adaptive.retrieval_triggered,
                "retrieval_snippets": [
                    {"source": s.source, "content": s.content, "score": s.score}
                    for s in adaptive.retrieval_snippets
                ],
                "memory_hit": adaptive.memory_hit,
                "injected_correction": adaptive.injected_correction,
                "adaptation_candidate_created": adaptive.adaptation_candidate_created,
                "adaptation_export_path": adaptive.adaptation_export_path,
                "observability": adaptive.observability,
            },
        }

    def health_check(self) -> dict:
        """Return service health status."""
        return {
            "status": "healthy",
            "models_loaded": len(self.registry.all_models),
            "topology_mode": "approximate" if TOPOLOGY_APPROXIMATE else "exact",
            "version": "0.1.0",
            "adaptive_layers": {
                "layer1_enabled": LAYER1_ENABLED,
                "layer2_enabled": LAYER2_ENABLED,
                "layer3_enabled": LAYER3_ENABLED,
            },
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
                    "id": m.model_id,
                    "name": m.display_name,
                    "tier": m.tier.name,
                    "capabilities": [c.value for c in m.capability_tags],
                }
                for m in service.registry.all_models
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
