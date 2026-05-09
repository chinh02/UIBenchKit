# UIBenchKit

UIBenchKit is a unified toolkit for running and evaluating UI-to-code
experiments. It takes webpage or UI screenshots as input, generates HTML/CSS
with a selected method and multimodal model, renders the generated page with
Playwright, and reports visual, structural, and cost-related metrics.

The UIBenchKit ecosystem is split across a small set of companion
repositories:

- [`UIBenchKit`](https://github.com/chinh02/UIBenchKit): the backend API, local
  runner, method integrations, dataset manager, model wrappers, renderer, and
  evaluators.
- [`uibenchkit-cli`](https://github.com/chinh02/uibenchkit-cli): a Typer-based
  command-line client for submitting, monitoring, reporting, and managing runs
  through the backend API.
- [`uibenchkit-GUI`](https://github.com/chinh02/uibenchkit-GUI): the React web
  frontend for the public leaderboard, model/method comparison views, result
  submission guidance, and the interactive live demo.
- [`uibenchkit-experiments`](https://github.com/chinh02/uibenchkit-experiments):
  the experiment artifact repository for raw benchmark outputs, generated
  leaderboard CSV/JSON files, and the scripts that summarize completed runs.

## What UIBenchKit Supports

Generation methods:

- `direct`: direct image-to-HTML prompting.
- `dcgen`: divide-and-conquer generation using image segmentation.
- `latcoder`: block-based Layout-as-Thought style generation and assembly.
- `uicopilot`: Pix2Struct-based DOM/bounding-box prediction plus LLM generation.
- `layoutcoder`: layout-tree extraction, atomic-region generation, and assembly.

Datasets:

- `design2code`: `SALT-NLP/Design2Code-hf`
- `dcgen`: `iforgott/DCGen`

Model families are configured in `config.py` and exposed by the `/health`
endpoint. The current backend includes OpenAI-compatible models, GPT-5,
Gemini, Claude, Qwen, DeepSeek, Grok, Doubao, Kimi, and AWS Bedrock models for
Mistral/Llama families.

## Repository Layout

```text
UIBenchKit/
  api.py                  # Flask API server
  run.py                  # Local runner without the API server
  dataset_manager.py      # Hugging Face dataset download/preparation
  config.py               # Models, methods, prompts, pricing, paths
  run_model.py            # Persistent run metadata/results model
  methods/                # dcgen/direct helpers plus LatCoder/UICopilot/LayoutCoder
  models/                 # Provider wrappers behind bot.ask(...)
  routes/                 # Flask route modules
  services/               # Run orchestration, generation, eval, auth, filesystem helpers
  evaluation/             # Code similarity, CLIP, and fine-grained metrics
  scripts/metric/         # Design2Code metric integration
  data/                   # Local datasets and custom input folders
  results/                # Generated artifacts and reports
```

## Backend Setup

The commands below assume PowerShell on Windows. On Linux/macOS, use
`python3 -m venv .venv` and `source .venv/bin/activate` instead.

```powershell
cd <path-to-UIBenchKit>

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m playwright install chromium
```

For Linux servers, Playwright may also require system packages:

```bash
python -m playwright install-deps chromium
```

### Environment Variables

Copy the example file and fill in the keys you need:

```powershell
Copy-Item .env.example .env
```

Minimum backend/API setting:

```text
UIBENCHKIT_API_KEY=your-dev-or-server-api-key
```

Provider keys are only needed for the model families you plan to run:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
QWEN_API_KEY=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

The backend also accepts OpenAI-compatible aliases such as
`OPENROUTER_API_KEY`, `OPENAI_COMPAT_API_KEY`, `OPENAI_API_BASE`,
`OPENAI_COMPAT_BASE_URL`, and `OPENROUTER_BASE_URL`.

## Dataset Setup

Datasets are downloaded into `data/hf_datasets`. The backend can list and use
downloaded datasets, but dataset downloading is currently handled by
`dataset_manager.py`.

```powershell
# From the UIBenchKit repo with the virtual environment activated
python dataset_manager.py list
python dataset_manager.py download design2code
python dataset_manager.py download dcgen
python dataset_manager.py info design2code
```

For private or rate-limited Hugging Face access, set `HF_TOKEN` before
downloading.

Custom input folders are also supported. A custom folder should contain PNG
screenshots:

```text
my_inputs/
  0.png
  1.png
  2.png
  placeholder.png   # optional helper image used by generated HTML
  0.html            # optional reference HTML for code-similarity evaluation
  1.html
  2.html
```

Only `.png` inputs are discovered by the run pipeline. Files containing
`placeholder` or `bbox` in the filename are ignored as benchmark instances.

## Local Backend Runner

Use `run.py` when you want to run experiments directly without starting the
Flask API.

```powershell
python run.py preflight

python run.py quick `
  --image .\data\demo\0.png `
  --method direct `
  --model gpt4 `
  --with-screenshot `
  --with-eval

python run.py run `
  --input .\data\demo `
  --method dcgen `
  --model gemini `
  --max-instances 5
```

Useful local-run options:

- `--method`: one of `direct`, `dcgen`, `latcoder`, `uicopilot`, `layoutcoder`.
- `--model`: model family or exact configured model version.
- `--output-root`: output directory, default `results`.
- `--max-instances`: limit the number of screenshots.
- `--no-screenshot`: skip Playwright screenshot rendering.
- `--no-eval`: skip evaluation.
- `--force`: overwrite existing generated outputs.
- `--user-api-key` and `--user-base-url`: override provider credentials for a run.

## Start the API Server

```powershell
cd <path-to-UIBenchKit>
.\.venv\Scripts\Activate.ps1

python api.py --host 127.0.0.1 --port 5000
```

The default API URL is:

```text
http://localhost:5000
```

Check server health:

```powershell
Invoke-RestMethod http://localhost:5000/health
```

Authenticated endpoints use the `x-api-key` header. If `UIBENCHKIT_API_KEY` is
not set, the development default is `dev-api-key-12345`.

## API Workflow

Submit a run:

```powershell
$headers = @{ "x-api-key" = "dev-api-key-12345" }
$body = @{
  model = "gpt4"
  method = "direct"
  dataset = "design2code"
  sample_ids = @("0", "1", "2")
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:5000/submit `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $body
```

Poll a run:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:5000/poll-jobs?run_id=<run_id>" `
  -Headers $headers
```

Get a report:

```powershell
$reportBody = @{ run_id = "<run_id>" } | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://localhost:5000/get-report `
  -Headers $headers `
  -ContentType "application/json" `
  -Body $reportBody
```

Other run-management endpoints include:

- `POST /list-runs`
- `POST /stop-run`
- `POST /resume-run`
- `POST /retry-failed`
- `POST /rerun-evaluation`
- `DELETE` or `POST /delete-run`
- `GET /download-artifacts?run_id=<run_id>`

## CLI Setup

The CLI lives in the separate
[`uibenchkit-cli`](https://github.com/chinh02/uibenchkit-cli) repository.

Install it in editable mode. You can use a separate virtual environment or the
same one as the backend.

```powershell
cd <path-to-uibenchkit-cli>

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

The package installs three equivalent commands:

```text
uibenchkit
uibenchkit_cli
uibenchkit-cli
```

Configure the CLI to talk to the backend:

```powershell
$env:UIBENCHKIT_API_URL = "http://localhost:5000"
$env:UIBENCHKIT_API_KEY = "dev-api-key-12345"
```

On Linux/macOS:

```bash
export UIBENCHKIT_API_URL="http://localhost:5000"
export UIBENCHKIT_API_KEY="dev-api-key-12345"
```

## CLI Workflow

Start the backend first, then use the CLI from another terminal.

```powershell
uibenchkit health
uibenchkit health --models

uibenchkit datasets list
uibenchkit datasets info design2code
uibenchkit datasets samples design2code --limit 5
```

Submit a benchmark run and wait for completion:

```powershell
uibenchkit submit gpt4 direct --dataset design2code --sample-ids 0,1,2
```

Submit without waiting:

```powershell
uibenchkit submit gemini dcgen --dataset dcgen --sample-ids 0,1,2 --no-wait
```

Monitor and fetch results:

```powershell
uibenchkit poll <run_id>
uibenchkit poll <run_id> --watch
uibenchkit get-report <run_id>
```

Run on a custom input folder:

```powershell
uibenchkit submit claude direct --input-dir .\data\my_inputs
```

Run the backend's `run-all` helper, which currently launches `dcgen` and
`direct` runs for the same input:

```powershell
uibenchkit run-all gpt4 --dataset design2code --sample-ids 0,1,2
```

Manage existing runs:

```powershell
uibenchkit list-runs
uibenchkit stop-run <run_id>
uibenchkit resume-run <run_id>
uibenchkit retry-failed <run_id>
uibenchkit rerun-evaluation <run_id>
uibenchkit delete-run <run_id>
```

Reports fetched by the CLI are saved by default under the CLI repository's
`results/` directory. Use `--output-dir` on `get-report` or `submit` to choose a
different location.

## Output Artifacts

Each backend run writes to:

```text
UIBenchKit/results/<run_id>/
```

Typical files include:

- `<sample_id>.html`: generated HTML.
- `<sample_id>.png`: screenshot rendered from generated HTML.
- `placeholder.png`: copied helper asset when available.
- `run_metadata.json`: run configuration, status, token usage, and cost estimate.
- `results.json`: per-instance status and output paths.
- `evaluation.json`: metric outputs when evaluation succeeds.
- `cost_report.json`: token/cost summary when token usage is available.

## Experiment Results And Leaderboards

Large benchmark runs and published leaderboard data are organized in the
separate
[`uibenchkit-experiments`](https://github.com/chinh02/uibenchkit-experiments)
repository. That repository stores raw run folders under `raw-data/`, including
generated HTML, rendered screenshots, run metadata, evaluation outputs, and cost
reports. It also contains the generated leaderboard files consumed by the web
frontend, such as `leaderboard/comparison_dcgen.csv`,
`leaderboard/comparison_design2code.csv`, `leaderboard/dcgen-results.json`, and
`leaderboard/design2code-results.json`.

After copying completed run artifacts into `uibenchkit-experiments`, run:

```bash
python summarize_leaderboard.py
```

The script scans the raw experiment folders, extracts metrics from each
`evaluation.json`, and regenerates the CSV/JSON files used for analysis and the
public leaderboard.

## Evaluation Notes

The current evaluation layer includes:

- Code similarity through `evaluation/code_similarity.py`.
- CLIP visual similarity through `evaluation/clip_score.py`.
- Fine-grained Design2Code-style visual metrics through
  `evaluation/fine_grained.py` and `scripts/metric`.

Fine-grained metrics are enabled only on Linux/macOS in the current backend.
On Windows, the report records that fine-grained metrics are unavailable.

CLIP evaluation requires the relevant ML dependencies from `requirements.txt`.
The first run may download model weights.

## Method-Specific Notes

- `direct` is the lightest setup path and is best for smoke tests.
- `dcgen` uses the built-in segmentation and grid implementation in `utils.py`.
- `latcoder` depends on OCR/CV/CLIP-related packages such as EasyOCR, OpenCV,
  and OpenCLIP.
- `uicopilot` lazy-loads the Pix2Struct checkpoint
  `xcodemind/uicopilot_structure` and may require substantial memory. It uses
  CUDA when available and falls back to CPU.
- `layoutcoder` can use an external LayoutCoder checkout for UIED
  preprocessing. Set `LAYOUTCODER_PROJECT_PATH` if you have one. If it is not
  available, the implementation falls back to direct screenshot analysis.

## Troubleshooting

`uibenchkit health` cannot connect:

- Confirm `python api.py --host 127.0.0.1 --port 5000` is running.
- Confirm `UIBENCHKIT_API_URL` matches the backend URL.

API requests return access errors:

- Set `UIBENCHKIT_API_KEY` in the CLI shell.
- Make sure it matches the backend `UIBENCHKIT_API_KEY` or the dev key printed
  when the server starts.

Dataset run says the dataset is not downloaded:

- Run `python dataset_manager.py download design2code` or
  `python dataset_manager.py download dcgen` from the backend repo.
- Then restart the API server if it was already running.

Playwright screenshot generation fails:

- Run `python -m playwright install chromium`.
- On Linux, also run `python -m playwright install-deps chromium`.

Provider/model errors:

- Run `python run.py preflight` or `uibenchkit health --models`.
- Check that the provider API key is set for the selected model family.
- For OpenAI-compatible gateways, check `OPENAI_BASE_URL` and model support for
  vision inputs.

Fine-grained metrics are missing:

- They are currently enabled only on Linux/macOS.
- Make sure the Design2Code metric dependencies are installed.

Large UICopilot/LayoutCoder runs are slow:

- Start with `--sample-ids 0` or `--max-instances 1`.
- Prefer `direct` first to verify model credentials and rendering.
