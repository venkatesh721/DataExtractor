import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status

from app.llm_service import (
    ExtractionConfigurationError,
    ExtractionServiceError,
    llm_service,
)
from app.schemas import ExtractRequest, ExtractResponse, HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await llm_service.close()


app = FastAPI(
    title="Structured Data Extractor",
    description="Extract structured fields from unstructured text using Google Gemini.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post(
    "/extract",
    response_model=ExtractResponse,
    responses={
        status.HTTP_502_BAD_GATEWAY: {"description": "Extraction provider failure."},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Gemini is not configured."},
    },
)
async def extract(request: ExtractRequest) -> ExtractResponse:
    try:
        extracted = await llm_service.extract(request.text)
    except ExtractionServiceError as exc:
        logger.error("Extraction request failed: %s", exc)
        status_code = (
            status.HTTP_503_SERVICE_UNAVAILABLE
            if isinstance(exc, ExtractionConfigurationError)
            else status.HTTP_502_BAD_GATEWAY
        )
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc
    return extracted
