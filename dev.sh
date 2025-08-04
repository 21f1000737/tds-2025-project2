#!/bin/bash

# Development start script using Uvicorn with hot reload
uvicorn tds_project2_2025:app --host 0.0.0.0 --port 8000 --reload