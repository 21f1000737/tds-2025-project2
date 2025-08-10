#!/bin/bash

# Production start script using Gunicorn with Uvicorn workers
gunicorn src.tds_project2_2025:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:$PORT --timeout 30