import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import ExtractResponse

SAMPLE_INPUTS = (
    "Paid ₹ 1,250 to AWS India Pvt Ltd on 12 March 2024. Invoice #INV-2024-0312. Services: cloud compute.",
    "Hi team, please find attached the invoice from Zoho Corp for ₹ 4,800 (software subscription, April 2024). Reference: ZHO-88821.",
    "hey can someone reimburse me for the uber last tuesday? it was like 340 rupees, went to the airport for the client meeting",
    "DELHI METRO RAIL CORP Token: 0042 Fare: INR 30 Date: 29-04-2024 From: Rajiv Chowk To: Hauz Khas",
    "meeting notes: discussed q2 targets, revenue up 12%, john to follow up with vendors next week",
)

logger = logging.getLogger("structured_data_extractor.samples")


def _validate_api_response(result: Any) -> None:
    if not isinstance(result, dict) or set(result) != {"review_required", "fields"}:
        raise ValueError("API response is missing required top-level fields.")
    raw_fields = result["fields"]
    expected_names = {
        "vendor_name",
        "amount",
        "currency",
        "date",
        "category",
        "description",
        "invoice_id",
    }
    if not isinstance(raw_fields, dict) or set(raw_fields) != expected_names:
        raise ValueError("API response does not contain exactly the seven fields.")

    fields_without_computed_values: dict[str, dict[str, Any]] = {}
    for name, field in raw_fields.items():
        if not isinstance(field, dict) or set(field) != {
            "value",
            "confidence",
            "needs_review",
        }:
            raise ValueError(f"API response field {name!r} has an invalid shape.")
        fields_without_computed_values[name] = {
            "value": field["value"],
            "confidence": field["confidence"],
        }

    try:
        validated = ExtractResponse.model_validate(
            {"fields": fields_without_computed_values}
        )
    except ValidationError as exc:
        raise ValueError("API response failed schema validation.") from exc

    expected_fields = validated.model_dump()["fields"]
    for name, expected in expected_fields.items():
        if raw_fields[name]["needs_review"] is not expected["needs_review"]:
            raise ValueError(f"API response field {name!r} has inconsistent review flag.")
    if result["review_required"] is not validated.review_required:
        raise ValueError("API response has an inconsistent review_required flag.")


def _read_error_body(error: HTTPError) -> Any:
    body = error.read().decode("utf-8", errors="replace")
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def run_samples(api_url: str, output_path: Path) -> bool:
    """POST all sample texts to the live API and persist each result or error."""
    results: list[dict[str, Any]] = []
    all_succeeded = True

    for input_number, text in enumerate(SAMPLE_INPUTS, start=1):
        request = Request(
            api_url,
            data=json.dumps({"text": text}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
                result = json.loads(body)
                _validate_api_response(result)
                results.append(
                    {
                        "input_number": input_number,
                        "input": text,
                        "response": result,
                    }
                )
                logger.info("Input %d: API returned HTTP %d.", input_number, response.status)
        except HTTPError as exc:
            all_succeeded = False
            results.append(
                {
                    "input_number": input_number,
                    "input": text,
                    "error": {
                        "status_code": exc.code,
                        "body": _read_error_body(exc),
                    },
                }
            )
            logger.error("Input %d: API returned HTTP %d.", input_number, exc.code)
        except (
            URLError,
            TimeoutError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            all_succeeded = False
            results.append(
                {
                    "input_number": input_number,
                    "input": text,
                    "error": {"message": str(exc)},
                }
            )
            logger.error("Input %d: request failed: %s", input_number, exc)

    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(output_path)
    except OSError:
        logger.exception("Could not save results to %s.", output_path)
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not clean up temporary results file.")
        return False

    succeeded = sum("response" in result for result in results)
    logger.info(
        "Saved %d results to %s (%d successful, %d failed).",
        len(results),
        output_path,
        succeeded,
        len(results) - succeeded,
    )
    return all_succeeded


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Send the five assessment samples to the live extraction API."
    )
    parser.add_argument(
        "--url",
        default=os.getenv("EXTRACT_API_URL", "http://127.0.0.1:8000/extract"),
        help="POST /extract URL (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results.json",
        help="Path for the JSON results array (default: project results.json)",
    )
    args = parser.parse_args()
    return 0 if run_samples(args.url, args.output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
