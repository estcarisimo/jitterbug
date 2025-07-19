#!/bin/bash
set -e

# Docker entrypoint script for Jitterbug

# Function to log messages
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a service is ready
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-30}
    
    log "Waiting for $host:$port to be ready..."
    
    for i in $(seq 1 $timeout); do
        if nc -z "$host" "$port" >/dev/null 2>&1; then
            log "$host:$port is ready"
            return 0
        fi
        sleep 1
    done
    
    log "Timeout waiting for $host:$port"
    return 1
}

# Function to check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    # Check Python packages
    python -c "import numpy, pandas, scipy, pydantic" || {
        log "Error: Required Python packages not found"
        exit 1
    }
    
    # Check optional packages
    if python -c "import ruptures" 2>/dev/null; then
        log "✓ ruptures package available"
    else
        log "⚠ ruptures package not available (optional)"
    fi
    
    if python -c "import torch" 2>/dev/null; then
        log "✓ torch package available"
    else
        log "⚠ torch package not available (optional)"
    fi
    
    if python -c "import matplotlib" 2>/dev/null; then
        log "✓ matplotlib package available"
    else
        log "⚠ matplotlib package not available (optional)"
    fi
    
    if python -c "import plotly" 2>/dev/null; then
        log "✓ plotly package available"
    else
        log "⚠ plotly package not available (optional)"
    fi
    
    log "Dependency check complete"
}

# Function to setup directories
setup_directories() {
    log "Setting up directories..."
    
    # Create directories if they don't exist
    mkdir -p /app/data /app/output /app/config
    
    # Check permissions
    if [ ! -w /app/data ]; then
        log "Warning: /app/data is not writable"
    fi
    
    if [ ! -w /app/output ]; then
        log "Warning: /app/output is not writable"
    fi
    
    log "Directory setup complete"
}

# Function to handle signals
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    log "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main execution
main() {
    log "Starting Jitterbug Docker container..."
    
    # Setup
    setup_directories
    check_dependencies
    
    # Handle different commands
    case "$1" in
        "api"|"server")
            log "Starting API server..."
            shift
            exec python -m jitterbug.api.server "$@"
            ;;
        "cli")
            log "Starting CLI mode..."
            shift
            exec jitterbug "$@"
            ;;
        "analyze")
            log "Running analysis..."
            exec jitterbug "$@"
            ;;
        "visualize")
            log "Running visualization..."
            exec jitterbug "$@"
            ;;
        "validate")
            log "Running validation..."
            exec jitterbug "$@"
            ;;
        "bash"|"sh")
            log "Starting interactive shell..."
            exec "$@"
            ;;
        "help"|"--help"|"-h")
            log "Showing help..."
            echo "Jitterbug Docker Container"
            echo ""
            echo "Usage: docker run jitterbug [COMMAND] [ARGS...]"
            echo ""
            echo "Commands:"
            echo "  api, server    Start the API server (default)"
            echo "  cli [ARGS]     Run CLI commands"
            echo "  analyze [ARGS] Run analysis"
            echo "  visualize [ARGS] Run visualization"
            echo "  validate [ARGS] Run validation"
            echo "  bash, sh       Interactive shell"
            echo "  help           Show this help"
            echo ""
            echo "Examples:"
            echo "  docker run jitterbug api --port 8000"
            echo "  docker run jitterbug analyze /app/data/rtts.csv"
            echo "  docker run jitterbug visualize /app/data/rtts.csv --output-dir /app/output"
            echo ""
            exit 0
            ;;
        *)
            if [ -n "$1" ]; then
                log "Unknown command: $1"
                log "Use 'help' to see available commands"
                exit 1
            else
                log "No command specified, starting API server..."
                exec python -m jitterbug.api.server --host 0.0.0.0 --port 8000
            fi
            ;;
    esac
}

# Run main function
main "$@"