#!/bin/bash
# Build container image for GPU kernel evaluation
# Supports both Podman (preferred) and Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
IMAGE_NAME="gpu-kernel-evaluator:latest"

# Custom storage location (405GB available on /workspace partition)
PODMAN_STORAGE_ROOT="$PROJECT_ROOT/.podman_storage"

# Detect available container runtime (prefer Podman)
if command -v podman &> /dev/null; then
    RUNTIME="podman"
    echo "✓ Using Podman to build image (preferred)"
    echo "✓ Storage location: $PODMAN_STORAGE_ROOT"

    # Create storage directories
    mkdir -p "$PODMAN_STORAGE_ROOT/containers"
    mkdir -p "$PODMAN_STORAGE_ROOT/run"

    # Export storage config for Podman
    export CONTAINERS_STORAGE_CONF="$PODMAN_STORAGE_ROOT/storage.conf"

    # Check available space
    AVAIL_GB=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
    echo "✓ Available space: ${AVAIL_GB}GB (need ~10GB for image)"

elif command -v docker &> /dev/null; then
    RUNTIME="docker"
    echo "✓ Using Docker to build image"
    echo "⚠️  Note: Docker storage location is controlled by Docker daemon config"
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

# Set temporary directory to workspace partition (avoid root partition space issues)
export TMPDIR="$PODMAN_STORAGE_ROOT/tmp"
mkdir -p "$TMPDIR"
echo "✓ Build temp directory: $TMPDIR"

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
