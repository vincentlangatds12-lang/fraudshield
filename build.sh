#!/bin/bash
# Render build script — runs from repo root (/opt/render/project/src)
set -e

echo "=== Working directory: $(pwd) ==="
echo "=== Contents: $(ls) ==="

echo "=== Installing Python dependencies ==="
pip install -r backend/requirements.txt

echo "=== Building Angular frontend ==="
cd frontend
npm install --legacy-peer-deps
npm run build -- --configuration production
cd ..

echo "=== Build complete ==="
echo "=== Frontend dist: ==="
ls frontend/dist/fraud-detection/browser/ 2>/dev/null || echo "WARNING: dist not found!"
