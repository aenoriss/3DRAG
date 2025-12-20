#!/bin/bash
set -e

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-aenoris}"
IMAGE_NAME="3drag-ollama"
TAG="${1:-latest}"

FULL_IMAGE="docker.io/${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"
echo "Note: This will be ~20GB due to baked-in models"
echo ""

# Build from project root with context
cd "$(dirname "$0")/.."
docker build -f runpod/Dockerfile.ollama -t "${FULL_IMAGE}" .

echo ""
echo "Pushing to Docker Hub..."
docker push "${FULL_IMAGE}"

echo ""
echo "============================================"
echo "Done! Create a new RunPod serverless endpoint:"
echo ""
echo "  Image: ${FULL_IMAGE}"
echo "  GPU:   24GB+ VRAM (L40, A10G, RTX 4090)"
echo ""
echo "Then set in your .env:"
echo "  RUNPOD_OLLAMA_ENDPOINT_ID=<endpoint-id>"
echo "============================================"
