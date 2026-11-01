import asyncio
import os

import pytest

from app.config import settings
from app.llm_service import GeminiExtractionService

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_GEMINI_INTEGRATION") != "1" or not settings.gemini_api_key,
    reason="Set GEMINI_API_KEY and RUN_GEMINI_INTEGRATION=1 to call Gemini.",
)


def test_real_gemini_structured_extraction() -> None:
    service = GeminiExtractionService()

    async def extract_and_close() -> object:
        try:
            return await service.extract(
                "Paid INR 1250 to AWS India Pvt Ltd on 2024-03-12."
            )
        finally:
            await service.close()

    response = asyncio.run(extract_and_close())

    assert response.fields.vendor_name.value is not None
    assert response.fields.amount.value is not None
    assert response.fields.currency.value == "INR"
    assert response.fields.date.value == "2024-03-12"
