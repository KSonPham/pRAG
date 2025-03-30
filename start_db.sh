#!/bin/bash

echo "Pulling Qdrant Docker image..."
docker pull qdrant/qdrant

echo "Starting Qdrant container..."
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant

echo "Qdrant is running on ports 6333 (REST API) and 6334 (gRPC)."