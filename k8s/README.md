# OncoLearn Kubernetes Deployment

This directory contains Kubernetes manifests for deploying OncoLearn with distributed data downloading.

## Architecture Overview

### Components

1. **Namespace** (`namespace.yaml`): Isolated environment for OncoLearn
2. **ConfigMaps** (`configmap.yaml`, `training-config.yaml`): Configuration management
3. **Storage** (`storage.yaml`): Persistent volumes for data, caches
4. **Data Downloader Job** (`data-downloader-job.yaml`): Parallel data downloading across multiple pods
5. **Training Deployment** (`training-deployment.yaml`): Model training workload

### How Data Partitioning Works

The data downloader uses Kubernetes **Indexed Jobs** to partition cohort downloads:

- Each pod gets a unique index (0 to N-1)
- Cohorts are split evenly across all pods
- Each pod downloads only its assigned portion
- Results are written to shared persistent storage

Example with 4 pods and 32 cohorts:
- Pod 0: Downloads cohorts 0-7 (8 cohorts)
- Pod 1: Downloads cohorts 8-15 (8 cohorts)
- Pod 2: Downloads cohorts 16-23 (8 cohorts)
- Pod 3: Downloads cohorts 24-31 (8 cohorts)

## Prerequisites

### 1. Kubernetes Cluster

You need a running Kubernetes cluster with:
- **ReadWriteMany (RWX) storage**: For shared data access (NFS, CephFS, etc.)
- **GPU support** (optional): NVIDIA GPU Operator or device plugin for training
- **Sufficient resources**: Memory and CPU for parallel downloads

### 2. Storage Class

Configure a StorageClass that supports ReadWriteMany:

```yaml
# Example NFS StorageClass (create separately)
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: nfs
provisioner: example.com/nfs
parameters:
  archiveOnDelete: "false"
```

Then uncomment `storageClassName: nfs` in [storage.yaml](storage.yaml).

### 3. Build Docker Image

Build and push the OncoLearn Docker image:

```bash
# Build the image
docker build -t your-registry/oncolearn:latest .

# Push to your registry
docker push your-registry/oncolearn:latest
```

Update the image in [kustomization.yaml](kustomization.yaml):

```yaml
images:
  - name: oncolearn
    newName: your-registry/oncolearn
    newTag: latest
```

## Deployment

### Quick Start

```bash
# Deploy everything using Kustomize
kubectl apply -k k8s/

# Or deploy manually
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/storage.yaml
kubectl apply -f k8s/training-config.yaml
kubectl apply -f k8s/data-downloader-job.yaml
kubectl apply -f k8s/training-deployment.yaml
```

### Configuration

Edit [configmap.yaml](configmap.yaml) to customize:

```yaml
data:
  # Which cohorts to download (comma-separated)
  XENA_COHORTS: "BRCA,LUAD,ACC,..."
  
  # Data sources
  DOWNLOAD_XENA: "true"
  DOWNLOAD_TCIA: "false"
  
  # Data categories to download
  DATA_CATEGORY: "mrna_seq,clinical,mutation"
```

### Scaling

Adjust parallelism in [data-downloader-job.yaml](data-downloader-job.yaml):

```yaml
spec:
  parallelism: 8  # Run 8 pods simultaneously
  completions: 8  # Total pods needed
```

Also update `TOTAL_PODS` in the init container to match.

## Monitoring

### Check Download Progress

```bash
# View all downloader pods
kubectl get pods -n oncolearn -l app=oncolearn-downloader

# View logs from a specific pod
kubectl logs -n oncolearn oncolearn-data-downloader-0 -f

# View logs from all downloader pods
kubectl logs -n oncolearn -l app=oncolearn-downloader --all-containers=true
```

### Check Job Status

```bash
# Get job status
kubectl get job -n oncolearn oncolearn-data-downloader

# Expected output:
# NAME                         COMPLETIONS   DURATION   AGE
# oncolearn-data-downloader    4/4           15m        16m
```

### Check Storage

```bash
# Check PVC status
kubectl get pvc -n oncolearn

# Exec into a pod to check data
kubectl exec -it -n oncolearn oncolearn-data-downloader-0 -- ls -lh /data/xenabrowser
```

## Training

Once data download is complete, the training deployment automatically starts:

```bash
# Check training pod
kubectl get pods -n oncolearn -l app=oncolearn-training

# View training logs
kubectl logs -n oncolearn -l app=oncolearn-training -f
```

## Advanced Usage

### GPU Node Selection

For GPU-accelerated training, uncomment in [training-deployment.yaml](training-deployment.yaml):

```yaml
nodeSelector:
  nvidia.com/gpu: "true"
```

### Custom Download Script

To use a custom partitioning strategy, modify the command in [data-downloader-job.yaml](data-downloader-job.yaml).

### Federated Learning (NVFlare)

For federated learning with multiple sites:

1. Create separate Jobs for each site
2. Use `nodeAffinity` to pin sites to specific nodes
3. Mount shared storage for model aggregation

## Cleanup

```bash
# Delete all resources
kubectl delete -k k8s/

# Or manually
kubectl delete namespace oncolearn

# Keep PVCs (data persists)
kubectl delete job,deployment -n oncolearn --all
```

## Troubleshooting

### Pods Stuck in Pending

```bash
# Check events
kubectl describe pod -n oncolearn <pod-name>

# Common issues:
# - Insufficient resources (CPU/Memory/GPU)
# - PVC not bound (storage provisioner issue)
# - Image pull errors
```

### Storage Issues

```bash
# Check PVC binding
kubectl get pvc -n oncolearn

# If pending, check storage class
kubectl get storageclass

# Check persistent volumes
kubectl get pv
```

### Download Failures

```bash
# Check logs for errors
kubectl logs -n oncolearn oncolearn-data-downloader-0

# Common issues:
# - Network connectivity
# - API rate limits
# - Invalid cohort names
# - Insufficient disk space
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │              oncolearn Namespace                    │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────┐          │    │
│  │  │   Data Downloader Job (Indexed)      │          │    │
│  │  │                                      │          │    │
│  │  │  ┌────────┐  ┌────────┐            │          │    │
│  │  │  │ Pod 0  │  │ Pod 1  │  ...       │          │    │
│  │  │  │Cohorts │  │Cohorts │            │          │    │
│  │  │  │ 0-7    │  │ 8-15   │            │          │    │
│  │  │  └───┬────┘  └───┬────┘            │          │    │
│  │  │      │           │                 │          │    │
│  │  └──────┼───────────┼─────────────────┘          │    │
│  │         │           │                            │    │
│  │         └─────┬─────┘                            │    │
│  │               ▼                                  │    │
│  │  ┌────────────────────────────┐                 │    │
│  │  │  Persistent Volume (RWX)   │                 │    │
│  │  │  /data (shared storage)    │                 │    │
│  │  └────────────┬───────────────┘                 │    │
│  │               │                                  │    │
│  │               ▼                                  │    │
│  │  ┌────────────────────────────┐                 │    │
│  │  │   Training Deployment      │                 │    │
│  │  │   (GPU-enabled)            │                 │    │
│  │  └────────────────────────────┘                 │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Next Steps

1. **Customize cohorts**: Edit `XENA_COHORTS` in ConfigMap
2. **Adjust resources**: Modify CPU/memory requests based on your cluster
3. **Enable GPU**: Uncomment GPU configurations for training
4. **Add monitoring**: Integrate with Prometheus/Grafana
5. **Implement CI/CD**: Automate image builds and deployments
