#!/bin/bash
# Render build script — runs before the app starts
set -e

echo "=== Building Angular frontend ==="
cd frontend
npm install --legacy-peer-deps
npm run build
echo "=== Frontend built ==="

echo "=== Installing Python dependencies ==="
cd ../backend
pip install -r requirements.txt
echo "=== Backend ready ==="
