#!/usr/bin/env python
"""
Standalone server for Jitterbug API.
"""

import argparse
import logging
import sys

try:
    import uvicorn
except ImportError:
    print("Error: uvicorn is required for the API server.")
    print("Install with: pip install jitterbug[api]")
    sys.exit(1)

from .app import create_app


def main():
    """Main entry point for the API server."""
    parser = argparse.ArgumentParser(
        description="Jitterbug Network Analysis API Server"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level (default: info)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--access-log",
        action="store_true",
        help="Enable access logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create application
    app = create_app()
    
    # Server configuration
    config = {
        "host": args.host,
        "port": args.port,
        "log_level": args.log_level,
        "access_log": args.access_log,
        "reload": args.reload,
    }
    
    # Add workers for production
    if not args.reload and args.workers > 1:
        config["workers"] = args.workers
    
    logger.info(f"🚀 Starting Jitterbug API server on {args.host}:{args.port}")
    logger.info(f"📚 API documentation available at http://{args.host}:{args.port}/docs")
    logger.info(f"🔧 Configuration: {config}")
    
    try:
        # Start server
        uvicorn.run(app, **config)
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Server failed to start: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()