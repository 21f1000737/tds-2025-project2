from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn

app = FastAPI(
    title="TDS Project2 API",
    description="Data Analyst Agent",
    version="0.1.0"
)

@app.get("/")
async def root():
    return {
        "message": "Hello from TDS Project2 FastAPI!",
        "status": "success",
        "endpoints": {
            "upload": "/upload - POST endpoint to upload file and get size",
            "health": "/health - GET endpoint for health check",
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "tds-project2-api"
    }


@app.post("/api/")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the file content to calculate size
        contents = await file.read()
        file_size = len(contents)
        
        # Format file size in human-readable format
        def format_file_size(size_bytes):
            if size_bytes == 0:
                return "0B"
            size_names = ["B", "KB", "MB", "GB", "TB"]
            i = 0
            while size_bytes >= 1024 and i < len(size_names) - 1:
                size_bytes /= 1024.0
                i += 1
            return f"{size_bytes:.2f}{size_names[i]}"
        
        return {
            "message": "File uploaded successfully",
            "status": "success",
            "file_details": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": file_size,
                "size_formatted": format_file_size(file_size)
            },
            "confirmation": f"File '{file.filename}' of size {format_file_size(file_size)} has been processed successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

def main() -> None:
    uvicorn.run("tds_project2_2025:app", host="0.0.0.0", port=8000, reload=True)
