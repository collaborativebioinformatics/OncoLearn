# Makefile for OncoLearn Kubernetes Deployment

# Variables
NAMESPACE = oncolearn
IMAGE_NAME = oncolearn
IMAGE_TAG = latest
REGISTRY ?= your-registry  # Override with: make deploy REGISTRY=my-registry.io

# Docker image full name
IMAGE = $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

.PHONY: help build push deploy deploy-download deploy-training status logs clean

help:
	@echo "OncoLearn Kubernetes Deployment"
	@echo ""
	@echo "Available targets:"
	@echo "  build              - Build Docker image"
	@echo "  push               - Push Docker image to registry"
	@echo "  deploy             - Deploy all resources to Kubernetes"
	@echo "  deploy-download    - Deploy only data downloader job"
	@echo "  deploy-training    - Deploy only training deployment"
	@echo "  status             - Check deployment status"
	@echo "  logs               - View logs from downloader pods"
	@echo "  logs-training      - View logs from training pod"
	@echo "  scale              - Scale downloader parallelism (use PODS=N)"
	@echo "  clean              - Delete all resources"
	@echo "  clean-keep-data    - Delete all resources but keep PVCs"
	@echo ""
	@echo "Examples:"
	@echo "  make build REGISTRY=my-registry.io"
	@echo "  make deploy"
	@echo "  make scale PODS=8"
	@echo "  make logs"

build:
	@echo "Building Docker image: $(IMAGE)"
	docker build -t $(IMAGE) -f ../Dockerfile ..
	@echo "Build complete!"

push: build
	@echo "Pushing Docker image: $(IMAGE)"
	docker push $(IMAGE)
	@echo "Push complete!"

deploy: push
	@echo "Deploying OncoLearn to Kubernetes..."
	kubectl apply -f namespace.yaml
	kubectl apply -f configmap.yaml
	kubectl apply -f storage.yaml
	kubectl apply -f training-config.yaml
	kubectl apply -f data-downloader-job.yaml
	kubectl apply -f training-deployment.yaml
	@echo "Deployment complete!"
	@echo "Check status with: make status"

deploy-download:
	@echo "Deploying data downloader job..."
	kubectl apply -f namespace.yaml
	kubectl apply -f configmap.yaml
	kubectl apply -f storage.yaml
	kubectl apply -f data-downloader-job.yaml
	@echo "Data downloader deployed!"

deploy-training:
	@echo "Deploying training deployment..."
	kubectl apply -f namespace.yaml
	kubectl apply -f training-config.yaml
	kubectl apply -f storage.yaml
	kubectl apply -f training-deployment.yaml
	@echo "Training deployment deployed!"

status:
	@echo "=== Namespace ==="
	kubectl get namespace $(NAMESPACE)
	@echo ""
	@echo "=== Persistent Volume Claims ==="
	kubectl get pvc -n $(NAMESPACE)
	@echo ""
	@echo "=== Data Downloader Job ==="
	kubectl get job -n $(NAMESPACE)
	@echo ""
	@echo "=== Downloader Pods ==="
	kubectl get pods -n $(NAMESPACE) -l app=oncolearn-downloader
	@echo ""
	@echo "=== Training Deployment ==="
	kubectl get deployment -n $(NAMESPACE)
	@echo ""
	@echo "=== Training Pods ==="
	kubectl get pods -n $(NAMESPACE) -l app=oncolearn-training

logs:
	@echo "Streaming logs from all downloader pods..."
	kubectl logs -n $(NAMESPACE) -l app=oncolearn-downloader --all-containers=true -f

logs-training:
	@echo "Streaming logs from training pod..."
	kubectl logs -n $(NAMESPACE) -l app=oncolearn-training -f

logs-pod:
	@if [ -z "$(POD)" ]; then \
		echo "Usage: make logs-pod POD=<pod-name>"; \
		exit 1; \
	fi
	kubectl logs -n $(NAMESPACE) $(POD) -f

scale:
	@if [ -z "$(PODS)" ]; then \
		echo "Usage: make scale PODS=<number>"; \
		echo "Example: make scale PODS=8"; \
		exit 1; \
	fi
	@echo "Scaling downloader to $(PODS) parallel pods..."
	@echo "Note: You need to update data-downloader-job.yaml manually and reapply"
	@echo "Set parallelism: $(PODS)"
	@echo "Set completions: $(PODS)"
	@echo "Set TOTAL_PODS env var: $(PODS)"

exec:
	@if [ -z "$(POD)" ]; then \
		echo "Usage: make exec POD=<pod-name>"; \
		exit 1; \
	fi
	kubectl exec -it -n $(NAMESPACE) $(POD) -- /bin/bash

clean:
	@echo "Deleting all OncoLearn resources..."
	kubectl delete namespace $(NAMESPACE)
	@echo "Cleanup complete!"

clean-keep-data:
	@echo "Deleting OncoLearn resources (keeping PVCs)..."
	kubectl delete job,deployment -n $(NAMESPACE) --all
	@echo "Cleanup complete (data preserved)!"

restart-download:
	@echo "Restarting data downloader job..."
	kubectl delete job -n $(NAMESPACE) oncolearn-data-downloader --ignore-not-found=true
	kubectl apply -f data-downloader-job.yaml
	@echo "Job restarted!"

restart-training:
	@echo "Restarting training deployment..."
	kubectl rollout restart deployment/oncolearn-training -n $(NAMESPACE)
	@echo "Training restarted!"

describe-pod:
	@if [ -z "$(POD)" ]; then \
		echo "Usage: make describe-pod POD=<pod-name>"; \
		exit 1; \
	fi
	kubectl describe pod -n $(NAMESPACE) $(POD)

watch:
	@echo "Watching OncoLearn pods..."
	watch -n 2 kubectl get pods -n $(NAMESPACE)
