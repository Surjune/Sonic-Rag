# Sonic-RAG setup (Windows / PowerShell)
#
#   powershell -ExecutionPolicy Bypass -File setup.ps1
#
# Creates the backend virtual environment, installs both dependency trees, and
# writes a .env from the template. The FAISS index ships with the repository,
# so nothing is downloaded or embedded here and the app is runnable as soon as
# this finishes and the API keys are filled in.

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# Where the prebuilt index is published. Pinned to a tag rather than
# 'latest' so a clone always gets artifacts matching its code.
$INDEX_RELEASE_TAG = 'index-10k'
$INDEX_RELEASE_URL = "https://github.com/Surjune/Sonic-Rag/releases/download/$INDEX_RELEASE_TAG"

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "  ok   $text" -ForegroundColor Green }
function Warn($text) { Write-Host "  warn $text" -ForegroundColor Yellow }

Step 'Checking prerequisites'

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) { throw 'python not found on PATH. Install Python 3.11 or newer.' }
$pyVersion = (python --version) -replace 'Python ', ''
$major, $minor = $pyVersion.Split('.')[0..1]
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
  throw "Python $pyVersion found; 3.11 or newer is required."
}
Ok "python $pyVersion"

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) { throw 'node not found on PATH. Install Node 18 or newer.' }
Ok "node $(node --version)"

Step 'Backend virtual environment'
$venv = Join-Path $root 'backend\.venv'
if (Test-Path $venv) {
  Ok 'already exists, reusing'
} else {
  python -m venv $venv
  Ok 'created backend\.venv'
}

$venvPython = Join-Path $venv 'Scripts\python.exe'
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet --timeout 90 --retries 5 -r (Join-Path $root 'backend\requirements.txt')
Ok 'python dependencies installed'

Step 'Frontend dependencies'
Push-Location (Join-Path $root 'frontend')
npm install --no-audit --no-fund
Pop-Location
Ok 'npm packages installed'

Step 'Environment file'
$envFile = Join-Path $root '.env'
if (Test-Path $envFile) {
  Ok '.env already exists, left untouched'
} else {
  Copy-Item (Join-Path $root '.env.example') $envFile
  Warn 'created .env from the template - add your API keys before running'
}

Step 'Index artifacts'
# The index is a build output, not source. It is 781MB at 10,000 rows, far
# past GitHub's 100MB per-file limit, so it ships as Release assets and is
# fetched here. Rebuilding locally instead costs a 5.4-hour embedding pass,
# which is not a reasonable thing to ask of someone evaluating the project.
$artifactDir = Join-Path $root 'backend\artifacts'
New-Item -ItemType Directory -Force -Path $artifactDir | Out-Null

foreach ($asset in @('vector_index.faiss', 'chunks.db')) {
  $target = Join-Path $artifactDir $asset
  if (Test-Path $target) { Ok "$asset already present"; continue }
  Write-Host "  downloading $asset from the $INDEX_RELEASE_TAG release..."
  try {
    # The progress bar slows large downloads by an order of magnitude on
    # Windows PowerShell, so it is silenced for the transfer only.
    $prior = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    Invoke-WebRequest -Uri "$INDEX_RELEASE_URL/$asset" -OutFile $target -UseBasicParsing
    $ProgressPreference = $prior
    Ok "downloaded $asset"
  } catch {
    Warn "could not download ${asset}: $_"
    Warn "fetch it manually from $INDEX_RELEASE_URL"
  }
}

$index = Join-Path $artifactDir 'vector_index.faiss'
if (Test-Path $index) {
  $vectors = & $venvPython -c "import faiss,sys; print(faiss.read_index(sys.argv[1]).ntotal)" $index
  Ok "index ready ($vectors vectors) - no rebuild needed"
} else {
  Warn 'index missing. Rebuild: python test_dataset_connection.py; python -m app.indexer --rows 10000'
}

Write-Host "`nSetup complete.`n" -ForegroundColor Green
Write-Host 'Next:' -ForegroundColor Cyan
Write-Host '  1. Put your GROQ_API_KEY and SARVAM_API_KEY in .env'
Write-Host '  2. Terminal 1:  cd backend; .\.venv\Scripts\Activate.ps1; uvicorn app.main:app --port 8000'
Write-Host '  3. Terminal 2:  cd frontend; npm run dev'
Write-Host '  4. Open:        http://localhost:5173'
