#!/bin/bash
# Build container image for GPU kernel evaluation
# Supports both Podman (preferred) and Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="gpu-kernel-evaluator:latest"

# Detect available container runtime (prefer Podman)
if command -v podman &> /dev/null; then
    RUNTIME="podman"
    echo "✓ Using Podman to build image (preferred)"
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    echo "✓ Using Docker to build image"
else
    echo "✗ Error: Neither Podman nor Docker is available"
    echo "Please install Podman or Docker first:"
    echo "  - Podman: https://podman.io/getting-started/installation"
    echo "  - Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

echo "Building container image: $IMAGE_NAME"
echo "This may take 5-10 minutes on first build..."

cd "$SCRIPT_DIR"

# Build image (command syntax is identical for Podman and Docker)
$RUNTIME build -t "$IMAGE_NAME" .

echo ""
echo "✓ Container image built successfully: $IMAGE_NAME"
echo "✓ Runtime: $RUNTIME"
echo ""
echo "To test the image:"
if [ "$RUNTIME" = "podman" ]; then
    echo "  podman run --rm --device nvidia.com/gpu=5 $IMAGE_NAME --help"
else
    echo "  docker run --rm --gpus device=5 $IMAGE_NAME --help"
fi
