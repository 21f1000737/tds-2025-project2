from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import json
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from PIL import Image
import base64
from io import BytesIO

def scrape_url_with_requests(url: str) -> str:
    """Scrape URL using requests library and save HTML to temp folder"""
    print(f"🌐 Starting requests-based scraping for: {url}")
    try:
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        # Make the request with timeout
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        
        # Get HTML content
        html_content = response.text
        
        # Save to temp folder - hardcoded path
        temp_dir = Path("/tmp")
        filename = f"scraped_{url.split('/')[-1].replace(':', '_').replace('?', '_')}.html"
        filepath = temp_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"💾 HTML saved as: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Requests scraping failed: {str(e)}")
        raise Exception(f"Scraping failed: {str(e)}")

def extract_dom_structure(html_filename: str, work_dir: str = None) -> str:
    """Extract DOM structure from HTML file"""
    try:
        # Look for file in work_dir first, then fallback to /tmp
        if work_dir:
            html_file_path = Path(work_dir) / html_filename
            if not html_file_path.exists():
                html_file_path = Path("/tmp") / html_filename
        else:
            html_file_path = Path("/tmp") / html_filename
        
        with open(html_file_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract key structural elements for data analysis
        dom_info = {
            "title": soup.title.string if soup.title else "",
            "tables": [],
            "table_data": []
        }
        
        # Extract table structures with sample data
        for i, table in enumerate(soup.find_all('table')[:5]):
            # Get all header elements
            headers_from_th = [th.get_text(strip=True) for th in table.find_all('th')]
            
            # Also check first row for headers (in case headers are in td elements)
            first_row = table.find('tr')
            first_row_cells = []
            if first_row:
                first_row_cells = [cell.get_text(strip=True) for cell in first_row.find_all(['td', 'th'])]
            
            # Use th headers if available, otherwise use first row
            actual_headers = headers_from_th if headers_from_th else first_row_cells
            
            table_info = {
                "index": i,
                "id": table.get('id'),
                "class": table.get('class'),
                "headers": actual_headers,
                "headers_from_th": headers_from_th,
                "first_row_cells": first_row_cells,
                "row_count": len(table.find_all('tr')),
                "sample_rows": []
            }
            
            # Get first few rows for analysis
            rows = table.find_all('tr')[:5]  # Increased to 5 rows
            for row in rows:
                cells = [td.get_text(strip=True) for td in row.find_all(['td', 'th'])]
                if cells:
                    table_info["sample_rows"].append(cells)
            
            dom_info["tables"].append(table_info)
        
        return json.dumps(dom_info, indent=2)
    except Exception as e:
        raise Exception(f"DOM extraction failed: {str(e)}")

def generate_analysis_code_with_tools(question_text: str, work_dir: str, file_summaries: List[Dict[str, Any]] = None, loop_count: int = 3) -> str:
    """Generate Python code using function calling with scraping and DOM tools"""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise Exception("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    # Define available functions for OpenAI
    tools = [
        {
            "type": "function",
            "function": {
                "name": "scrape_url",
                "description": "Scrape a URL using headless Playwright and save HTML to file in temp folder which returns the filename",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to scrape and save HTML from"
                        }
                    },
                    "required": ["url"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "extract_dom",
                "description": "Extract DOM structure and table data from HTML file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "html_filename": {
                            "type": "string",
                            "description": "Name of HTML file to analyze available in temp folder"
                        }
                    },
                    "required": ["html_filename"]
                }
            }
        }
    ]
    
    # Prepare file context information
    file_context = ""
    if file_summaries:
        file_context = "\n\nAvailable additional files in the working directory:\n"
        for file_info in file_summaries:
            file_context += f"""
File: {file_info['filename']} 
- Type: {file_info['file_type']}
- Summary: {file_info['summary']}
- Path: {file_info['file_path']}"""
            
            if file_info['file_type'] == 'image' and 'image_data' in file_info:
                file_context += f"\n- Image available as base64 data"
            elif file_info.get('columns'):
                file_context += f"\n- Columns: {file_info['columns']}"
            if file_info.get('shape'):
                file_context += f"\n- Shape: {file_info['shape']}"
            if file_info.get('error'):
                file_context += f"\n- Error: {file_info['error']}"
            file_context += "\n"

    messages = [
        {
            "role": "user",
            "content": f"""You are a data analyst agent with web scraping capabilities.

Task: {question_text}

You have access to these functions:
- scrape_url(url): Scrapes a URL and saves HTML, returns filename
- extract_dom(html_filename): Extracts DOM structure and table data from HTML file

{file_context}

Process:
1. First call scrape_url() for any URLs mentioned in the task
2. Then call extract_dom() to analyze the HTML structure  
3. Use available files in the working directory for analysis (use filename directly, not full paths)
4. For data files (CSV, JSON, Parquet, Excel), load them using pandas: pd.read_csv('filename.csv')
5. For images, they are available and can be processed or referenced using PIL: Image.open('image.png')
6. Use the extracted/loaded data to answer all questions
7. For visualizations, create plots and encode as base64 data URI under 100KB
8. Return results as JSON array/object as specified in the questions

Generate your solution step by step using function calls."""
        }
    ]
    
    print(f"🤖 Making OpenAI API call with tools for question analysis...")
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            timeout=60
        )
        print(f"✅ OpenAI API call successful, processing tool calls...")
    except Exception as e:
        print(f"❌ OpenAI API call failed: {str(e)}")
        raise Exception(f"OpenAI API call failed: {str(e)}")
    
    # Process function calls and build analysis data
    analysis_data = {}
    
    # Handle function calls from OpenAI response
    print(f"🔧 Processing tool calls from OpenAI response...")
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            
            if func_name == "scrape_url":
                print(f"🌐 Scraping URL: {func_args['url']}")
                try:
                    filename = scrape_url_with_requests(func_args["url"])
                    # Copy scraped file from /tmp to work_dir
                    import shutil
                    temp_file_path = Path("/tmp") / filename
                    work_file_path = Path(work_dir) / filename
                    if temp_file_path.exists():
                        shutil.copy2(temp_file_path, work_file_path)
                        print(f"📁 Copied {filename} to working directory")
                    analysis_data['scraped_file'] = filename
                    analysis_data['scraped_url'] = func_args["url"]
                    print(f"✅ URL scraped successfully, saved as: {filename}")
                except Exception as e:
                    print(f"❌ Scraping failed: {str(e)}")
                    analysis_data['scraping_error'] = str(e)
            
            elif func_name == "extract_dom":
                print(f"🔍 Extracting DOM from: {func_args['html_filename']}")
                try:
                    dom_structure = extract_dom_structure(func_args["html_filename"], work_dir)
                    analysis_data['dom_data'] = json.loads(dom_structure)
                    print(f"✅ DOM extraction successful")
                except Exception as e:
                    print(f"❌ DOM extraction failed: {str(e)}")
                    analysis_data['dom_error'] = str(e)
    
    # Feedback loop for code generation with error correction
    current_loop = 0
    last_error = None
    
    while current_loop < loop_count:
        # Generate analysis code
        error_context = ""
        if last_error:
            error_context = f"\n\nPrevious attempt failed with this error:\n{last_error}\n\nPlease fix the issues and generate improved code."
        
        # Include file summaries in the final code generation
        file_info_context = ""
        if file_summaries:
            file_info_context = f"\n\nAvailable files for analysis: {json.dumps(file_summaries, indent=2)}"

        final_messages = [
            {
                "role": "user", 
                "content": f"""Based on the scraped data and available files, generate Python code to analyze and answer:

{question_text}

Available scraped data: {json.dumps(analysis_data, indent=2)}
{file_info_context}

Generate complete Python code that:
1. Processes the scraped HTML data using BeautifulSoup and pandas
2. Loads and analyzes additional files using relative paths: pd.read_csv('filename.csv'), Image.open('image.png')
3. Processes images if provided using PIL or other libraries
4. Answers all questions accurately  
5. Creates visualizations as base64 data URIs under 100KB
6. Returns final JSON array/object as specified
7. Handles all data cleaning and analysis with proper error handling
8. Use pd.to_numeric() with errors='coerce' for numeric conversions
9. Handle missing values and non-numeric data appropriately
10. IMPORTANT: Use json.dumps() to print the final result as valid JSON
11. IMPORTANT: All file paths should be relative filenames only (not absolute paths)

{error_context}

Generate ONLY executable Python code, no explanations or markdown formatting."""
            }
        ]
        
        print(f"🔄 Loop {current_loop + 1}/{loop_count}: Generating analysis code...")
        try:
            final_response = client.chat.completions.create(
                model="gpt-4.1",
                messages=final_messages,
                timeout=60
            )
            print(f"✅ Code generation successful for loop {current_loop + 1}")
        except Exception as e:
            print(f"❌ OpenAI API call failed in loop {current_loop + 1}: {str(e)}")
            raise Exception(f"OpenAI API call failed in loop {current_loop}: {str(e)}")
        
        # Clean and extract the generated code
        final_code = final_response.choices[0].message.content.strip()
        
        # Extract Python code from markdown blocks
        import re
        python_code_match = re.search(r'```python\n(.*?)\n```', final_code, re.DOTALL)
        if python_code_match:
            final_code = python_code_match.group(1)
        else:
            # Fallback to original cleaning logic
            if final_code.startswith('```python'):
                final_code = final_code[9:]
            if final_code.startswith('```'):
                final_code = final_code[3:]
            if final_code.endswith('```'):
                final_code = final_code[:-3]
        
        # Test the generated code (without cleanup during feedback loop)
        print(f"🧪 Testing generated code (attempt {current_loop + 1})...")
        test_result = execute_analysis_code(final_code.strip(), work_dir, cleanup=False)
        
        if test_result["success"]:
            # Code executed successfully, return it
            print(f"🎉 Code executed successfully on attempt {current_loop + 1}!")
            return final_code.strip()
        else:
            # Code failed, prepare for next iteration
            print(f"⚠️  Code execution failed on attempt {current_loop + 1}: {test_result['error']}")
            last_error = test_result["error"]
            current_loop += 1
            
            if current_loop >= loop_count:
                # All attempts failed, cleanup and raise exception
                cleanup_temp_files(work_dir)
                raise Exception(f"Failed to generate working code after {loop_count} attempts. Last error: {last_error}")
    
    return final_code.strip()

def execute_analysis_code(code: str, work_dir: str, cleanup: bool = True) -> Dict[str, Any]:
    """Execute the generated Python code and capture results"""
    print(f"⚡ Executing analysis code in directory: {work_dir}")
    try:
        # Write code to temporary file
        code_file = Path(work_dir) / "analysis.py"
        with open(code_file, 'w') as f:
            f.write(code)
        
        # Save the generated code to output.txt for visualization (in project root)
        project_output_file = Path.cwd() / "output.txt"
        with open(project_output_file, 'w') as f:
            f.write(f"Generated Analysis Code:\n\n{code}")
        
        # Execute the code
        print(f"🏃 Running: uv run python {code_file}")
        result = subprocess.run(
            ["uv", "run", "python", str(code_file)],
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=180  # 3 minute timeout
        )
        print(f"📊 Execution completed with return code: {result.returncode}")
        
        # Append execution results to output.txt
        with open(project_output_file, 'a') as f:
            f.write(f"\n\nExecution Output:\n{result.stdout}")
            if result.stderr:
                f.write(f"\n\nExecution Errors:\n{result.stderr}")
        
        # Clean up temporary files after execution (only if requested)
        if cleanup:
            cleanup_temp_files(work_dir)
        
        if result.returncode == 0:
            # Try to parse the last line as JSON
            output_lines = result.stdout.strip().split('\n')
            for line in reversed(output_lines):
                if line.strip():
                    try:
                        return {"success": True, "result": json.loads(line)}
                    except json.JSONDecodeError:
                        continue
            return {"success": False, "error": "No valid JSON output found"}
        else:
            return {"success": False, "error": result.stderr}
            
    except subprocess.TimeoutExpired:
        if cleanup:
            cleanup_temp_files(work_dir)
        return {"success": False, "error": "Analysis timed out after 3 minutes"}
    except Exception as e:
        if cleanup:
            cleanup_temp_files(work_dir)
        return {"success": False, "error": str(e)}

def analyze_file_content(file_path: Path, filename: str) -> Dict[str, Any]:
    """Analyze file content and provide summary based on file type"""
    file_info = {
        "filename": filename,
        "file_path": filename,  # Use relative path since subprocess runs in same directory
        "file_size": file_path.stat().st_size,
        "file_type": "unknown",
        "summary": "",
        "sample_data": None,
        "columns": [],
        "shape": None,
        "error": None
    }
    
    try:
        file_extension = file_path.suffix.lower()
        
        if file_extension in ['.csv', '.tsv']:
            # Handle CSV/TSV files
            try:
                df = pd.read_csv(file_path, nrows=1000)  # Read first 1000 rows for analysis
                file_info.update({
                    "file_type": "csv",
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                    "sample_data": df.head(5).to_dict('records'),
                    "summary": f"CSV file with {df.shape[0]} rows and {df.shape[1]} columns. Column types: {list(df.dtypes.index)}"
                })
            except Exception as e:
                file_info["error"] = f"CSV parsing error: {str(e)}"
                
        elif file_extension == '.json':
            # Handle JSON files  
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    sample_data = data[:5] if len(data) > 5 else data
                    file_info.update({
                        "file_type": "json",
                        "summary": f"JSON array with {len(data)} items",
                        "sample_data": sample_data,
                        "shape": [len(data), len(data[0].keys()) if data and isinstance(data[0], dict) else 0]
                    })
                elif isinstance(data, dict):
                    file_info.update({
                        "file_type": "json", 
                        "summary": f"JSON object with {len(data)} top-level keys: {list(data.keys())}",
                        "sample_data": {k: str(v)[:100] + "..." if len(str(v)) > 100 else v for k, v in list(data.items())[:5]},
                        "columns": list(data.keys())
                    })
                else:
                    file_info.update({
                        "file_type": "json",
                        "summary": f"JSON data of type {type(data).__name__}",
                        "sample_data": str(data)[:500] + "..." if len(str(data)) > 500 else data
                    })
            except Exception as e:
                file_info["error"] = f"JSON parsing error: {str(e)}"
                
        elif file_extension == '.parquet':
            # Handle Parquet files
            try:
                df = pd.read_parquet(file_path)
                file_info.update({
                    "file_type": "parquet",
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                    "sample_data": df.head(5).to_dict('records'),
                    "summary": f"Parquet file with {df.shape[0]} rows and {df.shape[1]} columns. Column types: {list(df.dtypes.index)}"
                })
            except Exception as e:
                file_info["error"] = f"Parquet parsing error: {str(e)}"
                
        elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            # Handle Image files
            try:
                with Image.open(file_path) as img:
                    # Convert to base64 for LLM
                    buffered = BytesIO()
                    img_format = img.format or 'PNG'
                    img.save(buffered, format=img_format)
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    file_info.update({
                        "file_type": "image",
                        "summary": f"Image file: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}, format: {img_format}",
                        "image_data": f"data:image/{img_format.lower()};base64,{img_base64}",
                        "shape": img.size,
                        "mode": img.mode
                    })
            except Exception as e:
                file_info["error"] = f"Image parsing error: {str(e)}"
                
        elif file_extension == '.xlsx' or file_extension == '.xls':
            # Handle Excel files
            try:
                df = pd.read_excel(file_path, nrows=1000)  # Read first 1000 rows
                file_info.update({
                    "file_type": "excel",
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()},
                    "sample_data": df.head(5).to_dict('records'),
                    "summary": f"Excel file with {df.shape[0]} rows and {df.shape[1]} columns. Column types: {list(df.dtypes.index)}"
                })
            except Exception as e:
                file_info["error"] = f"Excel parsing error: {str(e)}"
        else:
            # Handle unknown file types - read as text (max 1000 lines)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= 1000:
                            break
                        lines.append(line.rstrip())
                    
                    file_info.update({
                        "file_type": "text",
                        "summary": f"Text file with {len(lines)} lines (max 1000 shown)",
                        "sample_data": lines,
                        "total_lines": len(lines)
                    })
            except UnicodeDecodeError:
                # Try binary mode for non-text files
                file_info.update({
                    "file_type": "binary", 
                    "summary": f"Binary file of size {file_info['file_size']} bytes",
                    "sample_data": "Binary file - cannot display content"
                })
            except Exception as e:
                file_info["error"] = f"File reading error: {str(e)}"
                
    except Exception as e:
        file_info["error"] = f"File analysis error: {str(e)}"
    
    return file_info

def cleanup_temp_files(work_dir: str):
    """Clean up all temporary files in the work directory and temp folder"""
    try:
        # Clean up work directory
        work_path = Path(work_dir)
        for file in work_path.glob("*"):
            if file.is_file():
                file.unlink()
        
        # Clean up hardcoded temp folder
        temp_dir = Path("/tmp")
        for file in temp_dir.glob("scraped_*.html"):
            if file.is_file():
                file.unlink()
    except Exception as e:
        print(f"Cleanup warning: {e}")


app = FastAPI(
    title="TDS Project2 API",
    description="Data Analyst Agent",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {
        "message": "TDS Project2 Data Analyst Agent",
        "status": "success",
        "endpoints": {
            "analyze": "/api/ - POST endpoint for data analysis tasks",
            "health": "/health - GET endpoint for health check",
        },
        "description": "Upload questions.txt and optional data files to perform automated analysis"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "tds-project2-api"
    }



@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """Main endpoint for data analysis tasks"""
    print(f"🚀 API endpoint called with {len(files)} files")
    try:
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            question_text = ""
            file_summaries = []
            
            # Process uploaded files
            print(f"📁 Processing {len(files)} uploaded files...")
            for file in files:
                print(f"📄 Processing file: {file.filename}")
                file_content = await file.read()
                file_path = temp_path / file.filename
                
                # Write file to temp directory
                with open(file_path, 'wb') as f:
                    f.write(file_content)
                
                # Handle questions.txt (compulsory file)
                if file.filename == "questions.txt":
                    question_text = file_content.decode('utf-8')
                    print(f"📝 Questions loaded: {len(question_text)} characters")
                else:
                    print(f"🔍 Analyzing file: {file.filename}")
                    # Analyze all other files and create summaries
                    file_analysis = analyze_file_content(file_path, file.filename)
                    file_summaries.append(file_analysis)
                    print(f"✅ File analysis complete for: {file.filename}")
            
            # Generate analysis code using function calling with feedback loop
            print(f"🧠 Starting analysis code generation...")
            try:
                analysis_code = generate_analysis_code_with_tools(question_text, temp_dir, file_summaries, loop_count=3)
                print(f"✅ Analysis code generated successfully")
                print(f"⚡ Executing final analysis code...")
                # Execute one final time with cleanup to get the actual result for the API response
                result = execute_analysis_code(analysis_code, temp_dir, cleanup=True)
                    
            except Exception as e:
                print(f"❌ Analysis failed: {str(e)}")
                result = {"success": False, "error": str(e)}
            
            if result["success"]:
                print(f"🎉 Analysis completed successfully!")
                return result["result"]
            else:
                print(f"💥 Analysis failed with error: {result['error']}")
                raise HTTPException(status_code=500, detail=result["error"])
                
    except Exception as e:
        print(f"💥 Endpoint failed with exception: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

def main() -> None:
    uvicorn.run("tds_project2_2025:app", host="0.0.0.0", port=8000, reload=True)
