# Nestack Structured Data Extractor

**Backend-only technical assessment project** — no frontend is included.

## Project overview

The API accepts unstructured transaction text, sends it to Google Gemini for
extraction, and returns structured, database-ready fields. Each field includes
its own confidence score, and low-confidence fields are flagged for human
review. It does not substitute local keyword extraction or fixed sample answers
for the model response.

## Architecture

```text
Client
  -> FastAPI POST /extract
    -> Pydantic request validation
    -> GeminiExtractionService (Google GenAI SDK, JSON schema response)
    -> Pydantic extraction validation and confidence/review derivation
  <- structured response with per-field confidence and review flags
```

- `app/main.py` defines `/health` and `/extract` and maps provider failures to
  explicit HTTP errors.
- `app/schemas.py` defines strict request, internal Gemini, and public response
  models, including field validation and the `0.75` review threshold.
- `app/llm_service.py` calls Gemini asynchronously, enforces a timeout, requests
  structured JSON, validates output, and does not return fabricated values on
  provider failures.
- `app/config.py` loads the root `.env` and reads settings without logging
  credentials.
- `scripts/run_samples.py` submits the assessment samples to a running API and
  writes actual responses or explicit errors to `results.json`.

## Technology stack

- Python 3.11
- FastAPI
- Google Gemini API using the official `google-genai` SDK
- Pydantic
- Uvicorn
- `python-dotenv`
- `pycountry` for ISO 4217 currency validation
- pytest and httpx for automated tests

## Gemini model choice

The configured primary model is **`gemini-2.5-flash-lite`**, with
**`gemini-2.5-flash`** as a fallback if the API reports the primary model as
unavailable. The Google model documentation describes Flash-Lite as its fastest
and most budget-friendly 2.5 model and recommends it for simple data extraction
and high-volume, low-latency tasks. The model strings are configured in
`app/config.py` defaults and can be overridden with `GEMINI_MODEL` and
`GEMINI_FALLBACK_MODEL`.

The project requests Gemini structured JSON output using the internal Pydantic
response schema. The configured model names are documented API model IDs; the
live availability of a model for a particular API key/account is only confirmed
when an authenticated API request succeeds.

## Extraction and confidence strategy

The Gemini system instruction treats input as untrusted source data and asks
the model to return all seven named fields. For every field it requests a
separate `value`, `confidence`, and internal evidence `reason`.

- Fields are evaluated independently; there is no single overall confidence
  score.
- Confidence should reflect evidence in the source text: explicit,
  unambiguous information receives higher confidence; contextual inference
  receives lower confidence; missing or ambiguous information receives low
  confidence.
- The model is instructed not to invent information, to return `null` for
  unsupported values, and to use null transaction values when no transaction
  is present.
- Internal reasons are validated but omitted from the public response.
- Confidence is bounded to `[0.0, 1.0]` before the response is validated.

Confidence is model-assessed evidence, not a separately calibrated statistical
probability. The API validates the numeric range and derives review flags
deterministically.

### Review logic

```text
needs_review = confidence < 0.75
review_required = true if any field has needs_review = true
```

The same threshold constant in `app/schemas.py` drives per-field and
top-level review flags.

## API

### `GET /health`

Returns `{"status":"ok"}`. This checks API process health; it does not make a
Gemini request.

### `POST /extract`

Request:

```json
{
  "text": "Paid ₹ 1,250 to AWS India Pvt Ltd on 12 March 2024."
}
```

Response shape (schema-only example; the API does not hardcode these values):

```json
{
  "review_required": true,
  "fields": {
    "vendor_name": {"value": null, "confidence": 0.0, "needs_review": true},
    "amount": {"value": null, "confidence": 0.0, "needs_review": true},
    "currency": {"value": null, "confidence": 0.0, "needs_review": true},
    "date": {"value": null, "confidence": 0.0, "needs_review": true},
    "category": {"value": null, "confidence": 0.0, "needs_review": true},
    "description": {"value": null, "confidence": 0.0, "needs_review": true},
    "invoice_id": {"value": null, "confidence": 0.0, "needs_review": true}
  }
}
```

Every response contains `vendor_name`, `amount`, `currency`, `date`, `category`,
`description`, and `invoice_id`, each with `value`, `confidence`, and
`needs_review`. Amount is numeric or null; date is normalized to `YYYY-MM-DD`
or null; currency is a valid uppercase ISO 4217 code or null; category is
`food`, `travel`, `utilities`, `software`, `other`, or null for a non-transaction.
The response does not expose internal evidence reasons.

Example request:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/extract" `
  -H "Content-Type: application/json" `
  -d "{\"text\":\"Paid INR 349 at Example Cafe on 2025-03-14\"}"
```

Errors: `422` for invalid input, `503` when `GEMINI_API_KEY` is not configured,
and `502` for Gemini/provider, timeout, or invalid model-output failures.

## Local setup

Run these commands from the repository root in Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` locally and set `GEMINI_API_KEY` to a valid Google AI Studio API
key. Never put the key in source code, `results.json`, `render.yaml`, or Git.
The root `.env` is ignored by `.gitignore`. For Render, configure the key in the
service's environment-variable settings instead of committing it.

Start the API:

```powershell
python -m uvicorn app.main:app --reload
```

Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

## Testing

Run unit and API contract tests (mocked Gemini; no API quota used):

```powershell
python -m pytest
```

The opt-in real Gemini integration test is separate and may consume API quota.
With a valid local `.env`, run it in PowerShell using:

```powershell
$env:RUN_GEMINI_INTEGRATION = "1"
python -m pytest tests\test_gemini_integration.py -q
Remove-Item Env:\RUN_GEMINI_INTEGRATION
```

The integration test skips unless both `GEMINI_API_KEY` and the explicit opt-in
variable are set. A skipped or mocked test does not verify live Gemini access.

## Assessment samples and `results.json`

The five exact assessment inputs are kept in `scripts/run_samples.py`. With the
API server running and a working Gemini key configured, run:

```powershell
python scripts\run_samples.py
```

The script POSTs each input to the live `/extract` endpoint and writes a JSON
array to `results.json`. Each entry includes its input number, input text, and
the actual API response. Failed requests are recorded as errors and do not
stop subsequent inputs; the process exits non-zero if any sample fails.
Automated runner tests use mocked HTTP responses and do not populate the
submission `results.json` with fabricated outputs. Regenerate the file with the
live runner before submission. If no live API credential is available, retain
the empty array or clearly labeled error output rather than inventing answers.

## Render deployment

`render.yaml` defines a Python web service, installs `requirements.txt`, exposes
`GET /health` as its health check, and starts:

```text
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

The Gemini key is declared as `sync: false`, so it must be entered as a
Render environment variable and is not stored in the repository. To deploy:

1. Create a **private** GitHub repository and push the submission (see below).
2. In Render, choose **New → Web Service**, connect GitHub, authorize access to
   the private `Venkatesh_Nestack_Submission` repository, and select it.
3. Use the repository root as the service root directory. If `render.yaml` is
   detected, apply its settings; otherwise choose Python and use the build and
   start commands above.
4. Add `GEMINI_API_KEY` under the service's Environment settings. Do not put
   the value in `render.yaml`.
5. Deploy and verify the Render-provided `/health` URL and then `/docs` and
   `/extract`.

**Live Deployment:** `<ADD_DEPLOYMENT_URL_HERE>`

Deployment has not been performed by this project setup.

## Private GitHub submission

Create a **private** repository named `Venkatesh_Nestack_Submission` in GitHub
first. Do not change it to public. From the project root, after configuring
your Git identity:

```powershell
git init
git add app tests scripts README.md requirements.txt results.json .env.example .gitignore render.yaml
git commit -m "Prepare Nestack structured data extractor submission"
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/Venkatesh_Nestack_Submission.git
git push -u origin main
```

The working `.env`, virtual environments, and Python caches are excluded by
`.gitignore`. Confirm that `.env` is not staged before committing:

```powershell
git status --short
git check-ignore .env
```

To add the three assessment evaluators, open the private repository on GitHub,
go to **Settings → Collaborators** (or **Collaborators and teams**), choose
**Add people**, search each evaluator's GitHub username, and send the invitation.
Grant only the repository access level required by the assessment. Do not make
the repository public.

## ZIP submission

From the project root, create a ZIP containing source, tests, documentation,
results, dependencies, environment template, and Render configuration while
excluding `.env`, virtual environments, caches, and `.git`:

```powershell
tar.exe -a -c -f .\Venkatesh_Nestack_Submission.zip app scripts tests README.md requirements.txt results.json .env.example .gitignore render.yaml
```

## Submission checklist

- [ ] `app/` source code
- [ ] `tests/` and `scripts/`
- [ ] `README.md`
- [ ] `requirements.txt`
- [ ] `.env.example` containing placeholders only
- [ ] `results.json` generated by the live sample runner
- [ ] `.gitignore`
- [ ] `render.yaml` if deploying on Render
- [ ] `.env`, credentials, virtual environments, caches, and `.git/` excluded
