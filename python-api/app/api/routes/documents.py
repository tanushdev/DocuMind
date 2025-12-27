"""Document upload and management endpoints."""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from app.config import get_settings
from app.models import DocumentUploadResponse, TaskStatusResponse
from app.services import DocumentProcessorFactory

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.
    
    Accepts PDF or TXT files. Processing is done asynchronously.
    Use the returned task_id to check processing status.
    
    Args:
        file: The document file to upload
        
    Returns:
        Task ID for status tracking
    """
    settings = get_settings()
    
    # Validate file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}"
        )
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {settings.max_file_size_mb}MB"
        )
    
    # Submit for background processing
    task_runner = DocumentProcessorFactory.get_task_runner()
    task_id = await task_runner.submit_task(content, file.filename)
    
    return DocumentUploadResponse(
        task_id=task_id,
        status="processing",
        message=f"Document '{file.filename}' uploaded and processing started"
    )


@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of a document processing task.
    
    Args:
        task_id: The task ID returned from upload
        
    Returns:
        Current task status and progress
    """
    task_runner = DocumentProcessorFactory.get_task_runner()
    status = await task_runner.get_task_status(task_id)
    
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Task not found: {task_id}"
        )
    
    return TaskStatusResponse(
        task_id=task_id,
        **status
    )


@router.get("/{document_id}")
async def get_document_info(document_id: str):
    """
    Get information about a processed document.
    
    Args:
        document_id: The document ID
        
    Returns:
        Document metadata
    """
    processor = DocumentProcessorFactory.get_processor()
    info = await processor.get_document_info(document_id)
    
    if info is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document not found: {document_id}"
        )
    
    return {
        "document_id": document_id,
        **info
    }
