import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError
from google.genai import errors

from app.llm_service import ExtractionServiceError, GeminiExtractionService
from app.main import app
from app.schemas import ExtractResponse, InternalExtraction


def _internal_result() -> dict:
    return {
        "vendor_name": {
            "value": "Example Store",
            "confidence": 0.99,
            "reason": "The text explicitly names Example Store.",
        },
        "amount": {
            "value": 12.5,
            "confidence": 0.99,
            "reason": "The text states a total of 12.50.",
        },
        "currency": {
            "value": "USD",
            "confidence": 0.9,
            "reason": "The dollar symbol and context indicate USD.",
        },
        "date": {
            "value": "2025-03-14",
            "confidence": 1.0,
            "reason": "The full date is stated explicitly.",
        },
        "category": {
            "value": "food",
            "confidence": 0.95,
            "reason": "The vendor is identified as a cafe.",
        },
        "description": {
            "value": None,
            "confidence": 0.0,
            "reason": "No description is provided.",
        },
        "invoice_id": {
            "value": None,
            "confidence": 0.0,
            "reason": "No invoice identifier is provided.",
        },
    }


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_extract_returns_specified_fields_without_internal_reasons() -> None:
    internal = _internal_result()
    public = {
        name: {
            "value": field["value"],
            "confidence": field["confidence"],
            "needs_review": field["confidence"] < 0.75,
        }
        for name, field in internal.items()
    }
    extraction = ExtractResponse.model_validate(
        {
            "fields": {
                name: {
                    "value": field["value"],
                    "confidence": field["confidence"],
                }
                for name, field in internal.items()
            }
        }
    )
    with patch(
        "app.main.llm_service.extract",
        new=AsyncMock(return_value=extraction),
    ):
        with TestClient(app) as client:
            response = client.post(
                "/extract",
                json={"text": "Paid 12.50 at Example Store"},
            )

    assert response.status_code == 200
    expected = {
        "review_required": True,
        "fields": public,
    }
    assert response.json() == expected
    assert "reason" not in response.text
    assert set(response.json()["fields"]) == {
        "vendor_name",
        "amount",
        "currency",
        "date",
        "category",
        "description",
        "invoice_id",
    }


def test_extract_rejects_blank_text() -> None:
    with TestClient(app) as client:
        response = client.post("/extract", json={"text": "   "})

    assert response.status_code == 422


def test_extract_rejects_empty_or_missing_text() -> None:
    with TestClient(app) as client:
        empty_response = client.post("/extract", json={"text": ""})
        missing_response = client.post("/extract", json={})

    assert empty_response.status_code == 422
    assert missing_response.status_code == 422


def test_extract_rejects_unexpected_request_fields() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/extract",
            json={"text": "Transaction text", "api_key": "do-not-accept"},
        )

    assert response.status_code == 422


def test_extract_provider_failure_is_not_returned_as_fake_data() -> None:
    with patch(
        "app.main.llm_service.extract",
        new=AsyncMock(side_effect=ExtractionServiceError("Provider failure.")),
    ):
        with TestClient(app) as client:
            response = client.post("/extract", json={"text": "Transaction text"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Provider failure."}


def test_parser_accepts_json_fence_and_omits_reason_from_public_result() -> None:
    response = SimpleNamespace(
        parsed=None,
        text=f"```json\n{json.dumps(_internal_result())}\n```",
    )

    internal = GeminiExtractionService._parse_response(response)
    public_data = {
        name: {
            "value": getattr(internal, name).value,
            "confidence": getattr(internal, name).confidence,
        }
        for name in type(internal).model_fields
    }
    public = ExtractResponse.model_validate({"fields": public_data})

    assert public.model_dump()["fields"]["vendor_name"] == {
        "value": "Example Store",
        "confidence": 0.99,
        "needs_review": False,
    }
    assert public.model_dump()["review_required"] is True
    assert "reason" not in public.model_dump_json()


def test_service_requests_structured_json_and_returns_public_fields() -> None:
    response = SimpleNamespace(parsed=None, text=json.dumps(_internal_result()))
    generate_content = AsyncMock(return_value=response)
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content),
        )
    )
    service = GeminiExtractionService()
    with patch.object(service, "_get_client", return_value=client):
        result = asyncio.run(service.extract("Paid at Example Store"))

    generate_content.assert_awaited_once()
    config = generate_content.await_args.kwargs["config"]
    assert config.response_schema is InternalExtraction
    assert config.response_mime_type == "application/json"
    assert result.fields.vendor_name.value == "Example Store"
    assert result.fields.description.value is None


def test_gemini_api_error_is_reported_as_provider_failure() -> None:
    generate_content = AsyncMock(
        side_effect=errors.APIError(
            code=500,
            response_json={"error": {"message": "provider failure"}},
        )
    )
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content),
        )
    )
    service = GeminiExtractionService()
    with patch.object(service, "_get_client", return_value=client):
        with pytest.raises(ExtractionServiceError, match="could not process"):
            asyncio.run(service.extract("Purchase at Example Store"))


def test_gemini_timeout_is_reported_clearly() -> None:
    async def timeout_request(**kwargs: object) -> None:
        await asyncio.sleep(0.05)

    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(
                generate_content=AsyncMock(side_effect=timeout_request),
            )
        )
    )
    service = GeminiExtractionService()
    with (
        patch.object(service, "_get_client", return_value=client),
        patch(
            "app.llm_service.settings",
            SimpleNamespace(
                gemini_timeout_seconds=0.001,
                gemini_model="gemini-test",
                gemini_fallback_model="gemini-test-fallback",
            ),
        ),
        pytest.raises(ExtractionServiceError, match="timed out"),
    ):
        asyncio.run(service.extract("Purchase at Example Store"))


def test_review_flags_follow_confidence_threshold() -> None:
    payload = {
        "fields": {
            name: {
                "value": field["value"],
                "confidence": 0.75,
            }
            for name, field in _internal_result().items()
        }
    }
    response = ExtractResponse.model_validate(payload)

    assert response.model_dump()["review_required"] is False
    assert all(
        not field["needs_review"]
        for field in response.model_dump()["fields"].values()
    )


def test_out_of_range_confidence_is_clamped_and_missing_field_reviewed() -> None:
    internal = _internal_result()
    internal["vendor_name"]["confidence"] = 1.25
    internal["description"]["confidence"] = 0.9
    response = SimpleNamespace(parsed=None, text=json.dumps(internal))
    generate_content = AsyncMock(return_value=response)
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content),
        )
    )
    service = GeminiExtractionService()
    with patch.object(service, "_get_client", return_value=client):
        result = asyncio.run(service.extract("Purchase at Example Store"))

    assert result.fields.vendor_name.confidence == 1.0
    assert result.fields.vendor_name.needs_review is False
    assert result.fields.description.confidence == 0.3
    assert result.fields.description.needs_review is True
    assert result.review_required is True


def test_parser_rejects_invalid_json() -> None:
    with pytest.raises(ExtractionServiceError, match="invalid JSON"):
        GeminiExtractionService._parse_response(
            SimpleNamespace(parsed=None, text="```json\nnot-json\n```")
        )


def test_parser_rejects_missing_fields_instead_of_fabricating_values() -> None:
    incomplete = _internal_result()
    del incomplete["amount"]
    with pytest.raises(ExtractionServiceError, match="unexpected response"):
        GeminiExtractionService._parse_response(
            SimpleNamespace(parsed=incomplete, text=None)
        )


def test_public_schema_rejects_invalid_category_and_date() -> None:
    public = {"fields": {
        name: {"value": field["value"], "confidence": field["confidence"]}
        for name, field in _internal_result().items()
    }}
    public["fields"]["category"]["value"] = "shopping"
    public["fields"]["date"]["value"] = "2025-03"
    public["fields"]["currency"]["value"] = "ZZZ"

    with pytest.raises(ValidationError):
        ExtractResponse.model_validate(public)
