"""
FastAPI application for Jitterbug REST API.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import logging
import time
from datetime import datetime

from .routes import router
from .models import ErrorResponseAPI

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="Jitterbug Network Analysis API",
        description="""
        ## Jitterbug API: Framework for Jitter-Based Congestion Inference
        
        This API provides endpoints for analyzing RTT (Round-Trip Time) measurements 
        to detect network congestion through jitter analysis and change point detection.
        
        ### Features
        
        * **RTT Data Analysis**: Analyze network latency measurements
        * **Change Point Detection**: Identify significant changes in network behavior
        * **Congestion Inference**: Detect network congestion periods
        * **Algorithm Comparison**: Compare different detection algorithms
        * **Data Validation**: Validate input data quality
        * **Multiple Algorithms**: Support for Ruptures, Bayesian, and PyTorch-based detection
        
        ### Usage
        
        1. **Validate Data**: Use `/validate` to check data quality
        2. **Analyze Data**: Use `/analyze` to perform congestion analysis
        3. **Compare Algorithms**: Use `/compare-algorithms` to evaluate different methods
        4. **Monitor Health**: Use `/health` and `/status` for service monitoring
        
        ### Authentication
        
        This API currently does not require authentication for research and development use.
        For production deployments, implement appropriate authentication mechanisms.
        
        ### Rate Limits
        
        This API does not currently implement rate limiting. For production use,
        consider implementing rate limiting based on your requirements.
        """,
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add compression middleware
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response
    
    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(f"Request: {request.method} {request.url}")
        
        response = await call_next(request)
        
        # Log response
        process_time = time.time() - start_time
        logger.info(f"Response: {response.status_code} in {process_time:.2f}s")
        
        return response
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "Jitterbug Network Analysis API",
            "version": "2.0.0",
            "description": "Framework for Jitter-Based Congestion Inference",
            "docs": "/docs",
            "health": "/api/v1/health",
            "status": "/api/v1/status"
        }
    
    # Custom OpenAPI schema
    def custom_openapi():
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title="Jitterbug Network Analysis API",
            version="2.0.0",
            description=app.description,
            routes=app.routes,
        )
        
        # Add custom schema information
        openapi_schema["info"]["x-logo"] = {
            "url": "https://via.placeholder.com/120x120.png?text=JB"
        }
        
        # Add examples to schemas
        openapi_schema["components"]["schemas"]["RTTMeasurementAPI"]["example"] = {
            "timestamp": "2024-01-01T10:00:00Z",
            "epoch": 1704110400.0,
            "rtt_value": 25.6,
            "source": "192.168.1.1",
            "destination": "8.8.8.8"
        }
        
        openapi_schema["components"]["schemas"]["AnalysisRequestAPI"]["example"] = {
            "data": {
                "measurements": [
                    {
                        "timestamp": "2024-01-01T10:00:00Z",
                        "epoch": 1704110400.0,
                        "rtt_value": 25.6,
                        "source": "192.168.1.1",
                        "destination": "8.8.8.8"
                    }
                ],
                "metadata": {
                    "source": "network_monitor",
                    "interval": "1min"
                }
            },
            "algorithm": "ruptures",
            "method": "jitter_dispersion",
            "threshold": 0.25,
            "min_time_elapsed": 1800
        }
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler."""
        logger.error(f"Unhandled exception: {exc}")
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponseAPI(
                error="InternalServerError",
                message="An internal server error occurred",
                timestamp=datetime.now()
            ).dict()
        )
    
    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Startup event handler."""
        logger.info("🚀 Jitterbug API starting up...")
        logger.info("📊 Service ready for network analysis requests")
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event handler."""
        logger.info("🛑 Jitterbug API shutting down...")
        logger.info("👋 Goodbye!")
    
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )