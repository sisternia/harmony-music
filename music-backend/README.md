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

## Check accounts

```powershell
.\.venv\Scripts\python.exe manage.py check
.\.venv\Scripts\python.exe manage.py test apps.accounts apps.roles_permissions
```
