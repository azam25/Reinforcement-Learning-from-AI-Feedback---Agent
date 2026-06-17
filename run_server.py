"""
Launch script for the RLAIF RAG Agent FastAPI service.

Run from anywhere inside the project:
    python3 run_server.py
    python3 run_server.py --port 8080 --reload
"""

import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so 'RLAIF_RAG_Agent' is importable
# regardless of where this script is invoked from.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Change cwd to project root so relative paths (faiss_index/, logs/) resolve
# correctly under the project directory.
# ---------------------------------------------------------------------------
os.chdir(PROJECT_ROOT)

import argparse
import uvicorn

parser = argparse.ArgumentParser(description="RLAIF RAG Agent — FastAPI server")
parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
parser.add_argument("--log-level", default="info", choices=["debug","info","warning","error"])
args = parser.parse_args()

if __name__ == "__main__":
    print(f"Starting RLAIF RAG Agent at http://{args.host}:{args.port}")
    print(f"  Docs:    http://localhost:{args.port}/docs")
    print(f"  Redoc:   http://localhost:{args.port}/redoc")
    print(f"  Health:  http://localhost:{args.port}/health")
    uvicorn.run(
        "RLAIF_RAG_Agent.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
