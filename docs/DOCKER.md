# Docker Setup for Jitterbug

This document provides comprehensive instructions for running Jitterbug in Docker containers.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start the API server
docker-compose up -d

# Check if the API is running
curl http://localhost:8000/api/v1/health

# View logs
docker-compose logs -f jitterbug-api
```

### Using Docker directly

```bash
# Build the image
docker build -t jitterbug:latest .

# Run the API server
docker run -d -p 8000:8000 --name jitterbug-api jitterbug:latest

# Run CLI commands
docker run --rm -v $(pwd)/data:/app/data jitterbug:latest jitterbug analyze /app/data/rtts.csv
```

## Docker Images

### Main Image: `jitterbug:latest`

- **Base**: Python 3.11 slim
- **Size**: ~200MB (multi-stage build)
- **Default Command**: API server
- **Ports**: 8000 (API)
- **User**: Non-root (`jitterbug`)
- **Health Check**: Enabled

### Image Features

- **Multi-stage build** for smaller production image
- **Non-root user** for security
- **Health checks** for monitoring
- **Optimized dependencies** with uv
- **Proper signal handling** for graceful shutdown

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `JITTERBUG_LOG_LEVEL` | Log level (debug, info, warning, error) | `info` |
| `JITTERBUG_WORKERS` | Number of API workers | `1` |
| `JITTERBUG_CONFIG` | Path to configuration file | None |
| `JITTERBUG_OUTPUT_FORMAT` | Default output format | `json` |

### Volume Mounts

| Path | Description |
|------|-------------|
| `/app/data` | Input data directory |
| `/app/output` | Output directory for results |
| `/app/examples` | Example scripts and data |

## Services

### API Server

The main API server provides REST endpoints for network analysis.

**Endpoints:**
- `GET /api/v1/health` - Health check
- `GET /api/v1/status` - Service status
- `POST /api/v1/analyze` - Analyze RTT data
- `POST /api/v1/validate` - Validate data
- `POST /api/v1/compare-algorithms` - Compare algorithms

**Access:**
- URL: http://localhost:8000
- Documentation: http://localhost:8000/docs
- API Explorer: http://localhost:8000/redoc

### CLI Interface

The CLI interface allows running analysis commands directly.

```bash
# Run with CLI profile
docker-compose --profile cli up jitterbug-cli

# Or run specific commands
docker-compose run --rm jitterbug-cli jitterbug analyze /app/data/rtts.csv
```

## Usage Examples

### API Usage

```bash
# Start the services
docker-compose up -d

# Check health
curl http://localhost:8000/api/v1/health

# Analyze data via API
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "measurements": [
        {
          "timestamp": "2024-01-01T10:00:00Z",
          "epoch": 1704110400.0,
          "rtt_value": 25.6,
          "source": "192.168.1.1",
          "destination": "8.8.8.8"
        }
      ]
    },
    "algorithm": "ruptures",
    "method": "jitter_dispersion"
  }'
```

### CLI Usage

```bash
# Analyze data from file
docker-compose run --rm jitterbug-cli jitterbug analyze /app/data/rtts.csv

# Generate visualization
docker-compose run --rm jitterbug-cli jitterbug visualize /app/data/rtts.csv --output-dir /app/output

# Validate data
docker-compose run --rm jitterbug-cli jitterbug validate /app/data/rtts.csv
```

### Data Processing Pipeline

```bash
# Create data directory
mkdir -p data output

# Copy your RTT data
cp your_rtt_data.csv data/

# Start services
docker-compose up -d

# Process data
docker-compose run --rm jitterbug-cli jitterbug analyze /app/data/your_rtt_data.csv --output /app/output/results.json

# Generate visualizations
docker-compose run --rm jitterbug-cli jitterbug visualize /app/data/your_rtt_data.csv --output-dir /app/output/plots

# Check results
ls output/
```

## Development

### Building Images

```bash
# Build the main image
docker build -t jitterbug:latest .

# Build with specific tag
docker build -t jitterbug:2.0.0 .

# Build with build arguments
docker build --build-arg PYTHON_VERSION=3.11 -t jitterbug:latest .
```

### Development Setup

```bash
# Build development image
docker build -t jitterbug:dev .

# Run with development volumes
docker run -it --rm \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -p 8000:8000 \
  jitterbug:dev bash
```

### Testing

```bash
# Run tests in container
docker-compose run --rm jitterbug-cli python -m pytest

# Run specific test
docker-compose run --rm jitterbug-cli python -m pytest tests/test_analyzer.py

# Run with coverage
docker-compose run --rm jitterbug-cli python -m pytest --cov=jitterbug
```

## Production Deployment

### Docker Compose Production

```yaml
version: '3.8'

services:
  jitterbug-api:
    image: jitterbug:latest
    ports:
      - "8000:8000"
    environment:
      - JITTERBUG_LOG_LEVEL=warning
      - JITTERBUG_WORKERS=4
    volumes:
      - ./data:/app/data:ro
      - ./output:/app/output
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '1.0'
          memory: 1G
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jitterbug-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: jitterbug-api
  template:
    metadata:
      labels:
        app: jitterbug-api
    spec:
      containers:
      - name: jitterbug-api
        image: jitterbug:latest
        ports:
        - containerPort: 8000
        env:
        - name: JITTERBUG_LOG_LEVEL
          value: "info"
        - name: JITTERBUG_WORKERS
          value: "1"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/v1/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Monitoring

```bash
# Check container health
docker-compose ps

# Monitor logs
docker-compose logs -f jitterbug-api

# Check resource usage
docker stats jitterbug-api

# Health check
curl -f http://localhost:8000/api/v1/health

# Service statistics
curl http://localhost:8000/api/v1/status
```

## Security

### Security Features

- **Non-root user**: Runs as `jitterbug` user
- **Read-only volumes**: Data mounted as read-only where possible
- **No secrets in environment**: Configuration via files
- **Minimal attack surface**: Multi-stage build removes build tools
- **Health checks**: Automatic health monitoring

### Best Practices

1. **Use specific tags** instead of `latest` in production
2. **Scan images** for vulnerabilities
3. **Update base images** regularly
4. **Use secrets management** for sensitive data
5. **Enable resource limits** to prevent resource exhaustion
6. **Monitor logs** for security events
7. **Use HTTPS** when exposing publicly

## Troubleshooting

### Common Issues

**Container fails to start:**
```bash
# Check logs
docker-compose logs jitterbug-api

# Check if port is in use
sudo netstat -tulpn | grep :8000

# Check disk space
docker system df
```

**Health check fails:**
```bash
# Check health manually
docker exec jitterbug-api curl -f http://localhost:8000/api/v1/health

# Check service status
docker exec jitterbug-api curl http://localhost:8000/api/v1/status
```

**Performance issues:**
```bash
# Check resource usage
docker stats jitterbug-api

# Increase workers
docker-compose up -d --scale jitterbug-api=3

# Check memory usage
docker exec jitterbug-api cat /proc/meminfo
```

**Data not accessible:**
```bash
# Check volume mounts
docker inspect jitterbug-api | grep -A 10 "Mounts"

# Check permissions
docker exec jitterbug-api ls -la /app/data

# Fix permissions
sudo chown -R 1000:1000 data/
```

### Debugging

```bash
# Interactive shell
docker-compose run --rm jitterbug-cli bash

# Debug API
docker-compose run --rm -p 8000:8000 jitterbug-api python -m jitterbug.api.server --log-level debug

# Check Python packages
docker-compose run --rm jitterbug-cli pip list

# Check system info
docker-compose run --rm jitterbug-cli python -c "import sys; print(sys.version)"
```

## Support

For issues related to Docker deployment:

1. Check the logs with `docker-compose logs`
2. Verify health status with `curl http://localhost:8000/api/v1/health`
3. Review the configuration and environment variables
4. Check the GitHub issues for known problems
5. Create a new issue with logs and system information

## Changelog

### Version 2.0.0
- Multi-stage build for smaller images
- API server as default command
- Health checks and monitoring
- Non-root user security
- Docker Compose configuration
- Comprehensive documentation