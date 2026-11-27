# Music backend

This directory contains two independent Python services:

| Service | Code | Default port | Purpose |
| --- | --- | --- | --- |
| Cover Studio | `apps/cover_studio/`, `apps/piano_*/` | 8100 | Transcription and piano cover generation |
| Accounts | `apps/accounts/`, `apps/roles_permissions/`, `config/`, `manage.py` | 8000 | Registration, verification, login, roles |

The accounts folder layout matches the Django backend in `D:\APP\NodeJS\music\music-backend`. Cover Studio keeps its existing FastAPI entry point and dependencies.

## Run accounts (PowerShell)

```powershell
cd D:\APP\NodeJS\tsumugi-piano-cover\music-backend
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-accounts.txt
.\.venv\Scripts\python.exe manage.py migrate
.\.venv\Scripts\python.exe manage.py runserver 127.0.0.1:8000
```

SQLite and console email work without an `.env` file. Verification codes appear in the server terminal. To use PostgreSQL and SMTP, copy `.env.example` to `.env` and set the corresponding values before running migrations. Use a unique `SECRET_KEY` outside local development. No admin account is created automatically. If needed, create one interactively:

```powershell
.\.venv\Scripts\python.exe manage.py create_account_admin admin@example.com
```

Accounts endpoints are at `http://127.0.0.1:8000/api/accounts/`; Swagger is at `http://127.0.0.1:8000/swagger/`. The Expo auth dialog is currently presentation only and must be connected separately to use these endpoints.

## Run Cover Studio

From the repository root, with its existing model environment installed:

```powershell
python -m pip install -e .\music-backend --no-deps
python -m pip install "yt-dlp[default]"
python -m cover_studio
```

Cover Studio stays at `http://127.0.0.1:8100`. It does not start the accounts service. Run each service in its own terminal.
The upload form at `http://localhost:8081` accepts audio or MIDI files up to 100 MB, or a single YouTube video up to 20 minutes. YouTube import uses yt-dlp, a JavaScript runtime (Node.js 22+ or Deno), and FFmpeg/FFprobe. The bundled FFmpeg tools in `.tsumugi/ffmpeg-shared/bin` are used when available; otherwise install both on `PATH`. The backend needs internet access to download a YouTube video.

## Songscription: lyrics, ad-libs and scat

**Qwen3-ASR 1.7B** (`qwen3-asr-1.7b`) is the default recognizer. **Whisper Large v3**
remains an alternative. The frontend reads the model list from the backend;
restart the backend to refresh the creation dialog. Legacy model selections map
to the new default.

The Modal lyrics worker runs Qwen recognition and Whisper word alignment on its
GPU. Deploy the worker with
`modal deploy music-backend/apps/cover_studio/modal_app.py` after code changes.
The lyrics image includes `qwen-asr==0.0.6`; the piano-cover image is separate.
With `USE_MODAL_GPU=true`, `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in
`music-backend/.env`, Songscription runs separation and recognition on Modal.
Configuration or remote errors stop the job with a message; they never start a
local GPU fallback. Only the lightweight client and audio/file handling run locally.
For local recognition, install the lyrics dependencies in a dedicated model
environment. Model weights download on first use. Selecting Qwen without its
runtime reports an error instead of silently substituting another recognizer.
Whisper alignment estimates word timings, including Vietnamese; when alignment
is unavailable, recognized text keeps segment timing.

Vocal recovery reuses the loaded recognizer on acoustic gaps and suspect covered
spans. It does not request a second transcription of every recognized line. The
recovery pass can compare the vocal stem with the original mix, retains repeated
syllables, and marks uncertain additions for review. Diagnostics report recovery
results and failures. Additional streams represent simultaneous event lanes;
they do not establish singer identities or imply isolated backing recordings.
Songscription does not use Gemini or require a transcription API key.

Post-processing preserves recognized repetitions and foreign-script text rather
than shortening them into a readable summary. Cross-pass matching accounts for
each occurrence so repeated phrases survive without duplicating the same
observation. Automatic reference lyrics do not replace recognized ad-libs.
Re-run extraction on an existing song to
apply these changes; saved transcripts are not rewritten automatically. No model
guarantees complete scat or ad-lib transcription: compare against manually labeled
audio and measure omissions separately from overall word error rate.

Regression checks (from the repository root, using the Cover Studio environment):

```powershell
python -c "import sys, unittest; sys.path.insert(0, 'music-backend/apps'); suite = unittest.TestSuite(unittest.defaultTestLoader.discover('music-backend/tests', pattern=name) for name in ('test_lyrics_transcription.py', 'test_vocal_preservation.py', 'test_vocal_recovery.py', 'test_qwen_songscription.py')); result = unittest.TextTestRunner().run(suite); sys.exit(not result.wasSuccessful())"
```

## Check accounts

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py test apps.accounts apps.roles_permissions
```
