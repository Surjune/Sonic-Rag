#!/usr/bin/env bash
# Sonic-RAG setup (macOS / Linux)
#
#   bash setup.sh
#
# Creates the backend virtual environment, installs both dependency trees, and
# writes a .env from the template. The FAISS index ships with the repository,
# so nothing is downloaded or embedded here and the app is runnable as soon as
# this finishes and the API keys are filled in.

set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

step() { printf '\n=== %s ===\n' "$1"; }
ok()   { printf '  ok   %s\n' "$1"; }
warn() { printf '  warn %s\n' "$1"; }

step 'Checking prerequisites'
command -v python3 >/dev/null || { echo 'python3 not found. Install Python 3.11+.'; exit 1; }
command -v node >/dev/null    || { echo 'node not found. Install Node 18+.'; exit 1; }

py_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || { echo "Python $py_version found; 3.11 or newer is required."; exit 1; }
ok "python $py_version"
ok "node $(node --version)"

step 'Backend virtual environment'
if [ -d "$root/backend/.venv" ]; then
  ok 'already exists, reusing'
else
  python3 -m venv "$root/backend/.venv"
  ok 'created backend/.venv'
fi

venv_python="$root/backend/.venv/bin/python"
"$venv_python" -m pip install --quiet --upgrade pip
"$venv_python" -m pip install --quiet --timeout 90 --retries 5 -r "$root/backend/requirements.txt"
ok 'python dependencies installed'

step 'Frontend dependencies'
(cd "$root/frontend" && npm install --no-audit --no-fund)
ok 'npm packages installed'

step 'Environment file'
if [ -f "$root/.env" ]; then
  ok '.env already exists, left untouched'
else
  cp "$root/.env.example" "$root/.env"
  warn 'created .env from the template - add your API keys before running'
fi

step 'Index artifacts'
if [ -f "$root/backend/artifacts/vector_index.faiss" ]; then
  vectors="$("$venv_python" -c "import faiss,sys; print(faiss.read_index(sys.argv[1]).ntotal)" \
    "$root/backend/artifacts/vector_index.faiss")"
  ok "prebuilt index present ($vectors vectors) - no rebuild needed"
else
  warn 'index missing. Run: python test_dataset_connection.py && python -m app.indexer --rows 250'
fi

printf '\nSetup complete.\n\nNext:\n'
printf '  1. Put your GROQ_API_KEY and SARVAM_API_KEY in .env\n'
printf '  2. Terminal 1:  cd backend && source .venv/bin/activate && uvicorn app.main:app --port 8000\n'
printf '  3. Terminal 2:  cd frontend && npm run dev\n'
printf '  4. Open:        http://localhost:5173\n'
