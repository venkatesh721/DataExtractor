import json
from unittest.mock import AsyncMock
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.request import Request

from scripts.run_samples import SAMPLE_INPUTS, run_samples


def _api_response(
    *,
    confidence: float = 0.9,
    transaction_present: bool = True,
) -> dict:
    if transaction_present:
        values = {
            "vendor_name": "Example Vendor",
            "amount": 349,
            "currency": "INR",
            "date": "2024-03-12",
            "category": "other",
            "description": None,
            "invoice_id": None,
        }
    else:
        values = dict.fromkeys(
            (
                "vendor_name",
                "amount",
                "currency",
                "date",
                "category",
                "description",
                "invoice_id",
            )
        )
        confidence = 0.2

    fields = {
        name: {
            "value": value,
            "confidence": confidence,
            "needs_review": confidence < 0.75,
        }
        for name, value in values.items()
    }
    return {
        "review_required": any(field["needs_review"] for field in fields.values()),
        "fields": fields,
    }


def _assert_api_response_contract(response: dict) -> None:
    assert "review_required" in response
    fields = response["fields"]
    assert set(fields) == {
        "vendor_name",
        "amount",
        "currency",
        "date",
        "category",
        "description",
        "invoice_id",
    }
    for field in fields.values():
        assert 0.0 <= field["confidence"] <= 1.0
        assert field["needs_review"] is (field["confidence"] < 0.75)
    assert response["review_required"] is any(
        field["needs_review"] for field in fields.values()
    )


def test_five_assessment_inputs_return_valid_responses_and_garbage_is_not_invented() -> None:
    from fastapi.testclient import TestClient

    from app.main import app
    from app.schemas import ExtractResponse

    async def fake_extract(text: str) -> ExtractResponse:
        payload = _api_response(transaction_present=text != SAMPLE_INPUTS[4])
        public_fields = {
            name: {
                "value": field["value"],
                "confidence": field["confidence"],
            }
            for name, field in payload["fields"].items()
        }
        return ExtractResponse.model_validate({"fields": public_fields})

    with patch(
        "app.main.llm_service.extract",
        new=AsyncMock(side_effect=fake_extract),
    ):
        with TestClient(app) as client:
            results = [
                client.post("/extract", json={"text": sample})
                for sample in SAMPLE_INPUTS
            ]

    assert len(results) == 5
    for result in results:
        assert result.status_code == 200
        _assert_api_response_contract(result.json())

    garbage_fields = results[4].json()["fields"]
    for name in ("vendor_name", "amount", "currency", "date", "category"):
        field = garbage_fields[name]
        assert field["value"] is None or field["confidence"] < 0.75


def test_runner_posts_each_input_and_saves_actual_responses(tmp_path: Path) -> None:
    api_payload = _api_response()
    body = json.dumps(api_payload).encode("utf-8")
    response = Mock(status=200, read=Mock(return_value=body))
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    output_path = tmp_path / "results.json"

    with patch("scripts.run_samples.urlopen", return_value=response) as urlopen_mock:
        success = run_samples("http://api.test/extract", output_path)

    assert success is True
    assert urlopen_mock.call_count == 5
    requests = [call.args[0] for call in urlopen_mock.call_args_list]
    assert all(isinstance(request, Request) for request in requests)
    assert all(request.get_method() == "POST" for request in requests)
    assert [json.loads(request.data)["text"] for request in requests] == list(
        SAMPLE_INPUTS
    )
    saved_results = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved_results) == 5
    assert [result["input_number"] for result in saved_results] == [1, 2, 3, 4, 5]
    assert all(result["response"] == api_payload for result in saved_results)


def test_runner_records_failure_and_continues_to_all_samples(tmp_path: Path) -> None:
    api_payload = json.dumps(_api_response()).encode("utf-8")
    response = Mock(status=200, read=Mock(return_value=api_payload))
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)

    def fake_urlopen(request: Request, timeout: int) -> Mock:
        if json.loads(request.data)["text"] == SAMPLE_INPUTS[1]:
            raise HTTPError(
                request.full_url,
                502,
                "upstream failure",
                hdrs=None,
                fp=BytesIO(b'{"detail":"provider unavailable"}'),
            )
        return response

    output_path = tmp_path / "results.json"
    with patch("scripts.run_samples.urlopen", side_effect=fake_urlopen) as urlopen_mock:
        success = run_samples("http://api.test/extract", output_path)

    assert success is False
    assert urlopen_mock.call_count == 5
    saved_results = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved_results) == 5
    assert saved_results[1]["error"] == {
        "status_code": 502,
        "body": {"detail": "provider unavailable"},
    }
    assert "response" in saved_results[2]


def test_runner_records_invalid_success_response_and_continues(tmp_path: Path) -> None:
    invalid_response = Mock(status=200, read=Mock(return_value=b'{"unexpected":true}'))
    invalid_response.__enter__ = Mock(return_value=invalid_response)
    invalid_response.__exit__ = Mock(return_value=False)
    valid_body = json.dumps(_api_response()).encode("utf-8")
    valid_response = Mock(status=200, read=Mock(return_value=valid_body))
    valid_response.__enter__ = Mock(return_value=valid_response)
    valid_response.__exit__ = Mock(return_value=False)

    responses = iter([invalid_response, *[valid_response for _ in SAMPLE_INPUTS[1:]]])
    output_path = tmp_path / "results.json"
    with patch("scripts.run_samples.urlopen", side_effect=lambda *args, **kwargs: next(responses)):
        success = run_samples("http://api.test/extract", output_path)

    assert success is False
    saved_results = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(saved_results) == 5
    assert saved_results[0]["input_number"] == 1
    assert "error" in saved_results[0]
    assert all("response" in result for result in saved_results[1:])
