#!/bin/bash

# Test API with file upload using curl
curl -X POST \
  -F "file=@question.txt" \
  http://0.0.0.0:8000/api/

echo ""
echo "Request completed."