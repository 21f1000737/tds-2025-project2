#!/bin/bash

# Test script for TDS Project2 API
# This script tests the API with sample CSV and JSON data files from nested directories

echo "🧪 Starting API test with sample data files..."

# API endpoint
API_URL="http://localhost:8000/api/"

# File paths (from nested directory structure)
QUESTIONS_FILE="questions.txt"
CSV_FILE="test_data/analytics/datasets/sample_data/sales_data.csv"
JSON_FILE="test_data/analytics/datasets/sample_data/customer_metrics.json"

# Check if files exist
echo "📋 Checking if test files exist..."
if [[ ! -f "$QUESTIONS_FILE" ]]; then
    echo "❌ Questions file not found: $QUESTIONS_FILE"
    exit 1
fi

if [[ ! -f "$CSV_FILE" ]]; then
    echo "❌ CSV file not found: $CSV_FILE"
    exit 1
fi

if [[ ! -f "$JSON_FILE" ]]; then
    echo "❌ JSON file not found: $JSON_FILE"
    exit 1
fi

echo "✅ All test files found"
echo "📁 Questions file: $QUESTIONS_FILE"
echo "📁 CSV file: $CSV_FILE" 
echo "📁 JSON file: $JSON_FILE"
echo ""

# Make API call
echo "🚀 Making API call to $API_URL..."
echo "📡 Uploading files with curl..."

curl -X POST \
    -F "files=@$QUESTIONS_FILE" \
    -F "files=@$CSV_FILE" \
    -F "files=@$JSON_FILE" \
    "$API_URL" \
    --max-time 300 \
    --silent \
    --show-error \
    --fail \
    -H "Accept: application/json" \
    | python3 -m json.tool

# Check curl exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ API test completed successfully!"
else
    echo ""
    echo "❌ API test failed!"
    exit 1
fi

echo ""
echo "📊 Test results should be displayed above as formatted JSON"
echo "🎯 Test completed with files from nested directory structure"