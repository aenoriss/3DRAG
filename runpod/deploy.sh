#!/bin/bash
set -e

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-aenoris}"
IMAGE_NAME="3drag-siglip2"
TAG="${1:-latest}"

FULL_IMAGE="docker.io/${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE}"

# Build from project root with context
cd "$(dirname "$0")/.."
docker build -f runpod/Dockerfile -t "${FULL_IMAGE}" .

echo "Pushing to Docker Hub..."
docker push "${FULL_IMAGE}"

echo ""
echo "Done! Update your RunPod template to use:"
echo "  ${FULL_IMAGE}"
echo ""
echo "Then restart the endpoint workers."
