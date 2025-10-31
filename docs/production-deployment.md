# Production Deployment Guide

This guide covers deploying the Coeus Multimodal AI Platform in production environments using Docker, Kubernetes, and Helm.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Helm Deployment](#helm-deployment)
- [Configuration](#configuration)
- [Scaling](#scaling)
- [Monitoring](#monitoring)
- [Security](#security)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **CPU Deployment**: 4+ CPU cores, 8GB+ RAM
- **GPU Deployment**: NVIDIA GPU with CUDA 11.8+, 8GB+ VRAM
- **Storage**: 50GB+ for model cache, 100GB+ for data
- **Network**: Stable internet connection for model downloads

### Software Requirements

- Docker 20.10+
- Kubernetes 1.24+ (for K8s deployment)
- Helm 3.8+ (for Helm deployment)
- kubectl configured for your cluster

### Access Requirements

- Docker Hub access for pulling images
- Kubernetes cluster with appropriate permissions
- Storage classes configured (SSD recommended)
- Ingress controller (NGINX recommended)

## Quick Start

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'
services:
  coeus-api:
    image: coeus/semantic-api:v0.2.0
    ports:
      - "8080:8080"
    environment:
      - RUST_LOG=info
      - COEUS_MODEL_CACHE_DIR=/app/models
    volumes:
      - ./models:/app/models
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Start the service
docker-compose up -d

# Check health
curl http://localhost:8080/health
```

### Helm (Production)

```bash
# Add the Coeus Helm repository
helm repo add coeus https://charts.coeus.ai
helm repo update

# Install with default configuration
helm install coeus coeus/coeus --namespace ai-platform --create-namespace

# Check deployment status
kubectl get pods -n ai-platform
```

## Docker Deployment

### Building from Source

```bash
# Clone the repository
git clone https://github.com/ryancinsight/Coeus.git
cd Coeus

# Build the production image
docker build -t coeus/semantic-api:v0.2.0 .

# Or build the GPU-enabled image
docker build --target gpu-runtime -t coeus/semantic-api-gpu:v0.2.0 .
```

### Running with Docker

```bash
# CPU deployment
docker run -d \
  --name coeus-api \
  -p 8080:8080 \
  -e RUST_LOG=info \
  -e COEUS_MODEL_CACHE_DIR=/app/models \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --health-cmd "curl -f http://localhost:8080/health || exit 1" \
  --health-interval 30s \
  --health-timeout 10s \
  --health-retries 3 \
  coeus/semantic-api:v0.2.0

# GPU deployment
docker run -d \
  --name coeus-api-gpu \
  --gpus all \
  -p 8080:8080 \
  -e COEUS_GPU_ACCELERATION=true \
  -v $(pwd)/models:/app/models \
  coeus/semantic-api-gpu:v0.2.0
```

## Kubernetes Deployment

### Manual Kubernetes Deployment

```bash
# Create namespace
kubectl create namespace ai-platform

# Apply the Kubernetes manifests
kubectl apply -f k8s/ -n ai-platform

# Check deployment
kubectl get pods -n ai-platform
kubectl get svc -n ai-platform
kubectl get ingress -n ai-platform
```

### Using Kustomize

```bash
# Apply with Kustomize overlays
kubectl apply -k k8s/overlays/production/

# Check status
kubectl get kustomizations -n ai-platform
```

## Helm Deployment

### Basic Installation

```bash
# Install with default values
helm install coeus ./helm --namespace ai-platform --create-namespace

# Install with custom values
helm install coeus ./helm \
  --namespace ai-platform \
  --set deployment.replicas=5 \
  --set config.models.gpuAcceleration=true \
  --set ingress.hosts[0].host=api.yourdomain.com
```

### GPU-Enabled Deployment

```bash
# Deploy with GPU support
helm install coeus-gpu ./helm \
  --namespace ai-platform \
  --set config.models.gpuAcceleration=true \
  --set deployment.gpuResources.requests.nvidia\\.com/gpu=1 \
  --set deployment.gpuResources.limits.nvidia\\.com/gpu=1 \
  --set deployment.nodeSelector.accelerator=nvidia-tesla-k80
```

### High Availability Setup

```bash
# Deploy with high availability
helm install coeus-ha ./helm \
  --namespace ai-platform \
  --set deployment.replicas=5 \
  --set autoscaling.minReplicas=3 \
  --set autoscaling.maxReplicas=10 \
  --set podDisruptionBudget.enabled=true \
  --set podDisruptionBudget.minAvailable=2
```

### Upgrading

```bash
# Upgrade to new version
helm upgrade coeus ./helm --namespace ai-platform

# Rollback if needed
helm rollback coeus 1 --namespace ai-platform
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RUST_LOG` | `info` | Logging level |
| `COEUS_PORT` | `8080` | Server port |
| `COEUS_WORKERS` | `4` | Number of worker threads |
| `COEUS_MODEL_CACHE_DIR` | `/app/models` | Model cache directory |
| `COEUS_DATA_DIR` | `/app/data` | Data storage directory |
| `COEUS_CLIP_MODEL` | `CLIP-ViT-B/32` | CLIP model variant |
| `COEUS_GPU_ACCELERATION` | `false` | Enable GPU acceleration |
| `COEUS_METRICS_ENABLED` | `true` | Enable Prometheus metrics |
| `COEUS_PROFILING_ENABLED` | `false` | Enable performance profiling |

### Configuration File

```yaml
# config.yaml
server:
  host: "0.0.0.0"
  port: 8080
  workers: 4
  timeout: 30

models:
  clip_model: "CLIP-ViT-B/32"
  cache_dir: "/app/models"
  preload: true
  gpu_acceleration: false

logging:
  level: "INFO"
  format: "json"
  enable_tracing: true

security:
  cors_origins: ["https://app.coeus.ai"]
  rate_limit: 100
  auth_required: true

monitoring:
  metrics_enabled: true
  health_check_interval: 30
  profiling_enabled: false
```

## Scaling

### Horizontal Pod Autoscaling

The Helm chart includes HPA configuration:

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

### Vertical Scaling

Adjust resource limits in values.yaml:

```yaml
deployment:
  resources:
    requests:
      cpu: 2000m
      memory: 4Gi
    limits:
      cpu: 4000m
      memory: 16Gi
```

### Cluster Scaling

For high-throughput deployments:

```bash
# Scale deployment
kubectl scale deployment coeus-semantic-api --replicas=10 -n ai-platform

# Use cluster autoscaler for node scaling
# Configure your cloud provider's cluster autoscaler
```

## Monitoring

### Prometheus Metrics

The application exposes metrics at `/metrics`:

```bash
# Query metrics
curl http://localhost:8080/metrics

# Example metrics
# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",endpoint="/v1/search/text",status="200"} 1507
```

### Health Checks

```bash
# Health endpoint
curl http://localhost:8080/health

# Response
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.2.0"
}
```

### Logging

```bash
# View logs
kubectl logs -f deployment/coeus-semantic-api -n ai-platform

# Structured logging example
{"timestamp":"2024-01-15T10:30:15.123Z","level":"INFO","message":"Model loaded successfully","model":"CLIP-ViT-B/32","duration_ms":1250}
```

## Security

### TLS Configuration

```yaml
ingress:
  tls:
    - secretName: coeus-api-tls
      hosts:
        - api.coeus.ai
```

### Network Policies

```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - namespaceSelector:
          matchLabels:
            name: ingress-nginx
  egress:
    - to:
      - ipBlock:
          cidr: 0.0.0.0/0
        ports:
          - protocol: TCP
            port: 443
```

### Security Context

```yaml
deployment:
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000

  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
        - ALL
```

## Troubleshooting

### Common Issues

#### Pod CrashLoopBackOff

```bash
# Check pod logs
kubectl logs -f pod/coeus-semantic-api-xxx -n ai-platform

# Check events
kubectl describe pod coeus-semantic-api-xxx -n ai-platform

# Check resource usage
kubectl top pods -n ai-platform
```

#### Model Loading Failures

```bash
# Check model cache permissions
kubectl exec -it deployment/coeus-semantic-api -n ai-platform -- ls -la /app/models

# Check network connectivity for downloads
kubectl exec -it deployment/coeus-semantic-api -n ai-platform -- curl -I https://huggingface.co
```

#### High Memory Usage

```bash
# Check memory usage
kubectl top pods -n ai-platform

# Adjust resource limits
helm upgrade coeus ./helm \
  --set deployment.resources.limits.memory=32Gi \
  -n ai-platform
```

#### GPU Issues

```bash
# Check GPU status
kubectl describe nodes | grep nvidia

# Check GPU usage
kubectl exec -it deployment/coeus-semantic-api -n ai-platform -- nvidia-smi
```

### Performance Tuning

#### Memory Optimization

```yaml
config:
  models:
    preload: false  # Load models on-demand
  server:
    workers: 2      # Reduce worker threads
```

#### CPU Optimization

```yaml
deployment:
  resources:
    requests:
      cpu: 2000m
    limits:
      cpu: 4000m
  env:
  - name: RAYON_NUM_THREADS
    value: "4"
```

#### GPU Optimization

```yaml
config:
  models:
    gpuAcceleration: true
deployment:
  gpuResources:
    limits:
      nvidia.com/gpu: 1
```

### Backup and Recovery

```bash
# Backup model cache
kubectl cp ai-platform/coeus-semantic-api-xxx:/app/models ./backup/models

# Backup data
kubectl cp ai-platform/coeus-semantic-api-xxx:/app/data ./backup/data

# Restore from backup
kubectl cp ./backup/models ai-platform/coeus-semantic-api-xxx:/app/models
```

## Support

For production deployment support:

- 📧 Email: support@coeus.ai
- 📖 Documentation: https://docs.coeus.ai
- 🐛 Issues: https://github.com/ryancinsight/Coeus/issues
- 💬 Community: https://discord.gg/coeus

## License

This deployment guide is part of the Coeus Multimodal AI Platform, licensed under MIT OR Apache-2.0.





