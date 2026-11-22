"""Cover Studio の API (FastAPI)。重い処理 (採譜・生成) は Runner の列に流し、画面は状態を問い合わせて進み具合を出す。

  設定         : GET /api/config、POST /api/engine/unload
  プロジェクト : GET/POST /api/projects、GET/PATCH/DELETE /api/projects/{pid}
  原曲         : POST …/transcribe、PUT …/source (手元の MIDI)、GET …/audio、…/source.mid、…/source/view
  テイク       : POST …/takes (生成)、PATCH/DELETE …/takes/{tid}、GET …/takes/{tid}/cover.mid、…/view
  好み         : GET/POST /api/styles (お気に入りから学習)、PATCH/DELETE /api/styles/{sid}
  ジョブ       : GET /api/jobs、DELETE /api/jobs/{id} (取り消し・停止)

{pid} はプロジェクトのフォルダ名。music-frontend/ のビルド (music-backend/apps/cover_studio/static) があれば / で配る。
"""

from __future__ import annotations

import logging
import random
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from .engine import Engine
from .jobs import Job, LiveTranscription, Runner
from .lyrics_service import (
    LYRICS_MODELS,
    ProjectLyricsStore,
    build_full_text,
    execute_lyrics_pipeline,
    language_label,
    sanitize_edited_segments,
)
from .project import Project, now, projects, safe_name
from .songscription import SONGSCRIPTION_MODELS, SongscriptionStore, run_songscription_job
from .styles import StyleStore
from .web import STATIC_DIR
from .youtube import MAX_AUDIO_BYTES, YouTubeImportError, download_youtube_audio

log = logging.getLogger(__name__)

# API を変えたら上げる。画面 (music-frontend/src/api.ts の API_VERSION) と違えば、画面がサーバの起動し直しを促す
API_VERSION = 5
AUDIO_EXTS = (".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac")
MEDIA_EXTS = (*AUDIO_EXTS, ".mp4", ".webm", ".mkv")
MIDI_EXTS = (".mid", ".midi")
# 画面の設定の既定値 (CoverParams の既定値のうち、Cover Studio だけで変えるもの)。拍のトークンのモデル (b4 / 5a) を
# 聴いて決めた: 原曲のリズムに寄せ、強弱と編曲の性質は学習データのまま。演奏者は公開番号 227 (Hugging Face の V1 / V2 の
# 番号。学習のチェックポイントは学習時の番号なので、同じ 227 でも別の演奏者になる)
STUDIO_DEFAULTS = {"onset_bias": 4.0, "dynamics": 1.0, "fill": 0.0, "span": 0.0, "channel": 227, "top_p": 1.0}


def studio_params(values: dict | None = None) -> dict:
    """生成の設定。style は好みのベクトルの id (Cover Studio だけのもの。あれば演奏者の番号の代わりに使う)"""
    from piano_cover.generate import CoverParams

    values = values or {}
    return {**asdict(CoverParams.from_dict({**STUDIO_DEFAULTS, **values})), "style": values.get("style") or None}


class ContinueFrom(BaseModel):
    take: str
    seconds: float = Field(ge=0)


class GenerateIn(BaseModel):
    params: dict = {}
    model: str | None = None
    count: int = Field(default=2, ge=1, le=4)
    seed: int | None = None
    continue_from: ContinueFrom | None = None


class ProjectPatch(BaseModel):
    title: str | None = None


class ScoreIn(BaseModel):
    model: str | None = None


class StyleIn(BaseModel):
    name: str = ""
    model: str | None = None


class StylePatch(BaseModel):
    name: str


class TakePatch(BaseModel):
    name: str | None = None
    favorite: bool | None = None
    memo: str | None = None


class SongscriptionPatch(BaseModel):
    title: str | None = None
    segments: list[dict] | None = None


class SongscriptionAlignIn(BaseModel):
    lyrics_text: str
    language: str = "auto"
    model: str = "turbo"
    isolate_vocals: bool = True


class SongscriptionRetranscribeIn(BaseModel):
    segment_ids: list[int] | None = None
    model: str | None = None


class LyricsExtractIn(BaseModel):
    language: str = "auto"
    model: str = "turbo"
    isolate_vocals: bool = True
    force_separate: bool = False


class LyricsAlignIn(BaseModel):
    lyrics_text: str
    language: str = "auto"
    model: str = "turbo"
    isolate_vocals: bool = True


class LyricsSaveIn(BaseModel):
    segments: list[dict]


def create_app(root: str | Path, engine: Engine) -> FastAPI:
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    for p in projects(root):
        p.recover()
        ProjectLyricsStore(p).recover()
    styles = StyleStore(root)
    styles.recover()
    songscriptions = SongscriptionStore(root)
    songscriptions.recover()

    def open_project(pid: str) -> Project:
        d = root / pid
        if pid in ("", ".", "..") or Path(pid).name != pid or not (d / "project.json").is_file():
            raise HTTPException(404, f"プロジェクトがありません: {pid}")
        return Project(d)

    # ------------------------------------------------------------ ジョブの中身
    def handle(job: Job) -> None:
        def progress(value: float | None, message: str) -> None:
            job.progress, job.message = value, message

        if job.kind == "songscription":
            sid = job.takes[0]
            meta = songscriptions.get(sid)
            chords = None
            if meta.get("project_id"):
                try:
                    from .midi_view import source_view

                    proj = open_project(meta["project_id"])
                    if proj.has_source:
                        chords = source_view(proj.source_path).get("chords")
                except Exception:
                    chords = None
            run_songscription_job(songscriptions, engine, sid, chords, progress, job._stop.is_set, job.add_log)
            return

        if job.kind == "lyrics":
            p = open_project(job.project)
            if p.audio_path is None or not p.audio_path.is_file():
                raise ValueError("Project này không có tệp âm thanh gốc để trích xuất lời bài hát.")
            lstore = ProjectLyricsStore(p)
            state_meta = lstore.update(state="running", error=None)
            mode = "align" if state_meta.get("mode") == "align" else "extract"
            input_txt_path = lstore.dir / "input_lyrics.txt"
            lyrics_text = input_txt_path.read_text(encoding="utf-8") if (mode == "align" and input_txt_path.is_file()) else ""
            chords = None
            if p.has_source:
                try:
                    from .midi_view import source_view

                    chords = source_view(p.source_path).get("chords")
                except Exception:
                    chords = None

            transcript = execute_lyrics_pipeline(
                engine=engine,
                cache=songscriptions.cache,
                source_audio=p.audio_path,
                audio_sha1=p.data.get("audio_sha1"),
                mode=mode,
                user_lyrics_text=lyrics_text,
                requested_language=str(state_meta.get("requested_language") or "auto"),
                model_id=str(state_meta.get("model") or "turbo"),
                isolate_vocals=bool(state_meta.get("isolate_vocals", True)),
                force_separate=bool(state_meta.get("force_separate", False)),
                chords=chords,
                progress=progress,
                should_stop=job._stop.is_set,
                log=job.add_log,
            )
            lstore.update(
                transcript=transcript,
                state="done",
                error=None,
                completed=now(),
                language=transcript["language"],
                language_name=transcript["language_name"],
                language_confidence=transcript["language_confidence"],
                detected_languages=transcript["detected_languages"],
                vocal_isolated=transcript["vocal_isolated"],
                vocal_cached=transcript["vocal_cached"],
                alignment_method=transcript["alignment_method"],
                duration=transcript["duration"],
                segment_count=len(transcript["segments"]),
                force_separate=False,
            )
            return

        if job.kind == "style":  # 曲に属さないジョブ。takes[0] が好みの id
            style = styles.update(job.takes[0], state="running", error=None)
            favorites = []
            for ref in style["takes"]:
                try:
                    project = open_project(ref["project"])
                    favorites.append((project, project.take(ref["take"])))
                except (HTTPException, KeyError):
                    job.add_log(f"見つからないので外します: {ref['project']} {ref['take']}")
            result = engine.train_style(
                style["model"]["id"], favorites, style["init"], progress, job._stop.is_set, job.add_log
            )
            styles.update(job.takes[0], state="done", trained=now(), **result)
            return

        p = open_project(job.project)
        if job.kind == "transcribe":
            p.update(transcribe={"state": "running", "error": None})
            live = job._live = LiveTranscription()

            def log(line: str) -> None:
                job.add_log(line)
                live.on_log(line)

            engine.transcribe(p, progress, job._stop.is_set, log, live.on_live)
        elif job.kind == "score":
            take_id = job.takes[0]
            p.update_take(take_id, score={**p.take(take_id)["score"], "state": "running", "error": None})
            measures = engine.score(p, take_id, progress, job._stop.is_set, job.add_log)
            p.update_take(
                take_id, score={**p.take(take_id)["score"], "state": "done", "measures": measures, "created": now()}
            )
        else:
            style = p.take(job.takes[0])["params"].get("style")
            vector = None
            if style:
                try:
                    vector = styles.get(style).get("vector")
                except KeyError:
                    vector = None
                if vector is None:
                    raise ValueError("好みのベクトルが見つかりません (消したか、まだ学習が終わっていません)")
            engine.generate(p, job.takes, progress, job._stop.is_set, job.add_log, style=vector)

    def on_end(job: Job) -> None:
        if job.kind == "songscription":
            try:
                item = songscriptions.get(job.takes[0])
            except KeyError:
                return
            if item.get("state") in ("queued", "running"):
                state = "cancelled" if job.state == "cancelled" else "error"
                songscriptions.update(item["id"], state=state, error=job.error)
            return
        if job.kind == "style":
            try:
                style = styles.get(job.takes[0])
            except KeyError:
                return  # 消された
            if style["state"] in ("queued", "running"):
                state = "cancelled" if job.state == "cancelled" else "error"
                styles.update(style["id"], state=state, error=job.error)
            return
        try:
            p = open_project(job.project)
        except HTTPException:
            return  # プロジェクトごと消された
        if job.kind == "transcribe":
            state = {"done": "done", "cancelled": "cancelled"}.get(job.state, "error")
            p.update(transcribe={"state": state, "error": job.error})
            return
        if job.kind == "lyrics":
            lstore = ProjectLyricsStore(p)
            cur = lstore.get()
            if cur.get("state") in ("queued", "running"):
                state = "cancelled" if job.state == "cancelled" else "error"
                lstore.update(state=state, error=job.error)
            return
        if job.kind == "score":
            try:
                score = p.take(job.takes[0]).get("score") or {}
            except KeyError:
                return
            if score.get("state") in ("queued", "running"):
                state = "cancelled" if job.state == "cancelled" else "error"
                p.update_take(job.takes[0], score={**score, "state": state, "error": job.error})
            return
        for take_id in job.takes:
            try:
                take = p.take(take_id)
            except KeyError:
                continue
            if take["state"] in ("queued", "running"):
                state = "cancelled" if job.state == "cancelled" else "error"
                p.update_take(take_id, state=state, error=job.error)

    runner = Runner(handle, on_end)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        runner.close()

    app = FastAPI(title="Cover Studio", lifespan=lifespan)
    # Expo web dev runs on port 8081; Expo DOM components on native use an
    # opaque WebView origin (null) when calling the API on port 8100.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["null", "http://localhost:8081", "http://127.0.0.1:8081"],
        allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}):\d+$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.runner = runner
    app.state.engine = engine

    def project_lyrics_view(p: Project) -> dict:
        lstore = ProjectLyricsStore(p)
        view = lstore.get()
        sha1 = p.data.get("audio_sha1")
        if sha1:
            view["vocal_cached"] = songscriptions.cache.has_vocal(sha1)
        job = next((j for j in runner.jobs(p.id) if j.kind == "lyrics"), None)
        if job is not None and job.active:
            view.update(job=job.id, state=job.state, progress=job.progress, message=job.message)
        if job is not None:
            view["log"] = job.log[-60:]
        view["models"] = list(LYRICS_MODELS)
        view["vocal_isolation_available"] = engine.tsumugi is not None
        return view

    def detail(p: Project) -> dict:
        data = p.data
        jobs = runner.jobs(p.id)
        live, scoring = {}, {}
        for job in jobs:
            if job.active and job.kind == "generate":
                for take_id in job.takes:
                    live[take_id] = job
            elif job.active and job.kind == "score":
                scoring[job.takes[0]] = job
        takes = []
        for take in p.takes():
            job = live.get(take["id"])
            if job is not None:
                take = {**take, "job": job.id, "progress": job.progress, "message": job.message}
            job = scoring.get(take["id"])
            if job is not None and take.get("score"):
                score = {**take["score"], "job": job.id, "progress": job.progress, "message": job.message}
                take = {**take, "score": score}
            takes.append(take)
        transcribe = data.get("transcribe")
        job = next((j for j in jobs if j.kind == "transcribe"), None)
        if transcribe is not None and job is not None:
            transcribe = {**transcribe, "job": job.id, "message": job.message, "log": job.log[-80:]}
            if job.active:
                transcribe["state"] = job.state
        lyrics_summary = ProjectLyricsStore(p).summary()
        ljob = next((j for j in jobs if j.kind == "lyrics"), None)
        if ljob is not None and ljob.active:
            lyrics_summary = {
                **(lyrics_summary or {}),
                "state": ljob.state,
                "job": ljob.id,
                "progress": ljob.progress,
                "message": ljob.message,
            }
        return {
            "id": p.id,
            "title": data["title"],
            "created": data["created"],
            "audio": p.audio_path.name if p.audio_path else None,
            "source": data.get("source") if p.has_source else None,
            "transcribe": transcribe,
            "lyrics": lyrics_summary,
            "takes": takes,
            "busy": any(j.active for j in jobs),
        }

    def summary(p: Project) -> dict:
        data = p.data
        takes = p.takes()
        return {
            "id": p.id,
            "title": data["title"],
            "created": data["created"],
            "audio": p.audio_path.name if p.audio_path else None,
            "has_source": p.has_source,
            "transcribe": data.get("transcribe"),
            "lyrics": ProjectLyricsStore(p).summary(),
            "takes": len(takes),
            "favorites": sum(t.get("favorite", False) for t in takes),
            "busy": bool(runner.active_for(p.id)),
        }

    def save_upload(upload: UploadFile, exts: tuple[str, ...]) -> Path:
        name = Path(upload.filename or "").name
        if Path(name).suffix.lower() not in exts:
            raise HTTPException(400, f"対応していないファイルです ({' / '.join(exts)}): {name}")
        path = root / ".uploads" / uuid.uuid4().hex / name
        path.parent.mkdir(parents=True)
        size = 0
        try:
            with open(path, "wb") as f:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_AUDIO_BYTES:
                        raise HTTPException(413, "File phải nhỏ hơn 100 MB.")
                    f.write(chunk)
        except BaseException:
            shutil.rmtree(path.parent, ignore_errors=True)
            raise
        return path

    def submit_transcribe(p: Project) -> Job:
        if engine.tsumugi is None:
            raise HTTPException(400, "tsumugi が見つからないので採譜できません (MIDI を一緒に入れてください)")
        p.update(transcribe={"state": "queued", "error": None})
        return runner.submit("transcribe", p.id)

    # ------------------------------------------------------------ 設定
    @app.get("/api/config")
    def config():
        return {
            **engine.describe(),
            "defaults": studio_params(),
            "busy": any(j.active for j in runner.jobs()),
            "api_version": API_VERSION,
        }

    @app.post("/api/engine/unload")
    def unload():
        if any(j.state == "running" for j in runner.jobs()):
            raise HTTPException(409, "実行中のジョブがあります")
        engine.unload()
        return engine.describe()

    # ------------------------------------------------------------ プロジェクト
    @app.get("/api/projects")
    def list_projects():
        return [summary(p) for p in projects(root)]

    @app.post("/api/projects")
    def create_project(
        audio: Annotated[UploadFile | None, File()] = None,
        midi: Annotated[UploadFile | None, File()] = None,
        title: Annotated[str, Form()] = "",
        youtube: Annotated[str, Form()] = "",
    ):
        """Create from an audio file, MIDI file, or one YouTube video."""
        youtube = youtube.strip()
        if youtube and (audio is not None or midi is not None):
            raise HTTPException(400, "Hãy chọn file hoặc link YouTube, không dùng cả hai.")
        if audio is None and midi is None and not youtube:
            raise HTTPException(400, "Cần file âm thanh, MIDI hoặc link YouTube.")
        if midi is None and engine.tsumugi is None:
            raise HTTPException(400, "Cần cài tsumugi để chuyển âm thanh thành MIDI.")

        audio_path = midi_path = None
        with tempfile.TemporaryDirectory(prefix="cover-studio-youtube-") as temp_dir:
            try:
                video_title = ""
                if youtube:
                    try:
                        audio_path, video_title = download_youtube_audio(youtube, Path(temp_dir))
                    except YouTubeImportError as exc:
                        raise HTTPException(exc.status_code, str(exc)) from exc
                else:
                    audio_path = save_upload(audio, AUDIO_EXTS) if audio is not None else None
                    midi_path = save_upload(midi, MIDI_EXTS) if midi is not None else None
                name = title.strip() or video_title or Path((audio_path or midi_path).name).stem
                p = Project.create(root, name, audio_path, midi_path)
            finally:
                for path in (audio_path, midi_path):
                    if path is not None and path.is_relative_to(root / ".uploads"):
                        shutil.rmtree(path.parent, ignore_errors=True)
        if midi_path is None:
            submit_transcribe(p)
        return detail(p)

    @app.get("/api/projects/{pid}")
    def get_project(pid: str):
        return detail(open_project(pid))

    @app.patch("/api/projects/{pid}")
    def update_project(pid: str, body: ProjectPatch):
        p = open_project(pid)
        if body.title is not None and body.title.strip():
            p.update(title=body.title.strip())
        return detail(p)

    @app.delete("/api/projects/{pid}", status_code=204)
    def delete_project(pid: str):
        p = open_project(pid)
        if runner.active_for(pid):
            raise HTTPException(409, "実行中・待ちのジョブがあります")
        p.delete()
        runner.forget(pid)

    # ------------------------------------------------------------ 原曲
    @app.post("/api/projects/{pid}/transcribe")
    def transcribe(pid: str):
        p = open_project(pid)
        if any(j.kind == "transcribe" for j in runner.active_for(pid)):
            raise HTTPException(409, "採譜は実行中です")
        submit_transcribe(p)
        return detail(p)

    @app.get("/api/projects/{pid}/transcribe/live")
    def transcribe_live(pid: str, since: int = 0):
        """採譜中に確定したノート (since 番目から後ろ) とステムごとの進み具合。採譜していなければ空"""
        open_project(pid)
        job = next((j for j in runner.jobs(pid) if j.kind == "transcribe"), None)
        if job is None or job._live is None:
            return {"seq": 0, "events": [], "stems": [], "stage": None, "active": bool(job and job.active)}
        return {**job._live.since(since), "active": job.active}

    @app.put("/api/projects/{pid}/source")
    def upload_source(pid: str, midi: Annotated[UploadFile, File()]):
        p = open_project(pid)
        if any(j.kind == "transcribe" for j in runner.active_for(pid)):
            raise HTTPException(409, "採譜の実行中は差し替えられません")
        path = save_upload(midi, MIDI_EXTS)
        try:
            p.set_source(path, "upload")
        finally:
            shutil.rmtree(path.parent, ignore_errors=True)
        return detail(p)

    @app.get("/api/projects/{pid}/audio")
    def get_audio(pid: str):
        p = open_project(pid)
        if p.audio_path is None or not p.audio_path.is_file():
            raise HTTPException(404, "音源がありません")
        return FileResponse(p.audio_path)

    @app.get("/api/projects/{pid}/source.mid")
    def get_source_midi(pid: str):
        p = open_project(pid)
        if not p.has_source:
            raise HTTPException(404, "原曲の MIDI はまだありません")
        return FileResponse(p.source_path, media_type="audio/midi", filename=f"{safe_name(p.data['title'])}_source.mid")

    @app.get("/api/projects/{pid}/source/view")
    def get_source_view(pid: str):
        from .midi_view import source_view

        p = open_project(pid)
        if not p.has_source:
            raise HTTPException(404, "原曲の MIDI はまだありません")
        return source_view(p.source_path)

    # ------------------------------------------------------------ テイク
    @app.post("/api/projects/{pid}/takes")
    def generate(pid: str, body: GenerateIn):
        p = open_project(pid)
        if not p.has_source:
            raise HTTPException(400, "原曲の MIDI がまだありません")
        try:
            entry = engine.entry(body.model)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if body.continue_from is not None:
            try:
                source_take = p.take(body.continue_from.take)
            except KeyError as e:
                raise HTTPException(404, f"テイクがありません: {body.continue_from.take}") from e
            if source_take["state"] != "done":
                raise HTTPException(400, "生成が終わったテイクからだけ続きを作れます")
        params = studio_params(body.params)
        if params["style"]:
            try:
                style = styles.get(params["style"])
            except KeyError as e:
                raise HTTPException(400, "好みが見つかりません (消したかもしれません)") from e
            if style["state"] != "done":
                raise HTTPException(400, "この好みはまだ学習が終わっていません")
            if style["model"]["id"] != entry.id:
                raise HTTPException(
                    400, f"この好みは {style['model']['label']} で学習したものです。同じモデルで使ってください"
                )
        record = {
            "params": params,
            "model": {"id": entry.id, "label": entry.label},
            "seed": body.seed if body.seed is not None else random.randrange(2**31),
            "count": body.count,
            "continue_from": body.continue_from.model_dump() if body.continue_from else None,
        }
        takes = p.new_takes(body.count, record)
        runner.submit("generate", pid, [t["id"] for t in takes])
        return detail(p)

    def find_take(p: Project, tid: str) -> dict:
        try:
            return p.take(tid)
        except KeyError as e:
            raise HTTPException(404, f"テイクがありません: {tid}") from e

    @app.patch("/api/projects/{pid}/takes/{tid}")
    def update_take(pid: str, tid: str, body: TakePatch):
        p = open_project(pid)
        find_take(p, tid)
        return p.update_take(tid, **body.model_dump(exclude_none=True))

    @app.delete("/api/projects/{pid}/takes/{tid}", status_code=204)
    def delete_take(pid: str, tid: str):
        p = open_project(pid)
        find_take(p, tid)
        if any(tid in j.takes for j in runner.active_for(pid)):
            raise HTTPException(409, "生成中・待ちのテイクは先に止めてください")
        p.delete_take(tid)

    @app.get("/api/projects/{pid}/takes/{tid}/cover.mid")
    def get_take_midi(pid: str, tid: str):
        p = open_project(pid)
        take = find_take(p, tid)
        path = p.take_midi(tid)
        if not path.is_file():
            raise HTTPException(404, "まだ生成していません")
        # メタデータなしで書かれた古いテイクだけ、ここで 1 回だけ直す (書き込みは一時ファイル -> 置換)。
        # 毎回書き直すと秒 -> tick の丸めが積もって音が少しずつずれるうえ、同時のダウンロードと
        # ぶつかって書きかけのファイルを返すことがある。直したら take.json に印を残す。
        source_created = (p.data.get("source") or {}).get("created")
        recorded = take.get("source_created")
        if recorded is not None:
            same_source = recorded == source_created
        else:  # 印がない古いテイク: 原曲がテイクより後に作られていれば、別の原曲に入れ替わっている
            same_source = source_created is None or source_created <= (take.get("created") or "")
        # 原曲の拍を書き直した (SheetSage の拍にした・小節ごとのテンポにした) 後のテイクも、同じように 1 回だけ写し直す
        source_beats = (p.data.get("source") or {}).get("beats_map")
        stale = not take.get("metadata_copied") or take.get("source_beats") != source_beats
        if p.has_source and stale and same_source:
            try:
                from piano_cover.metadata import copy_metadata

                copy_metadata(p.source_path, path)
                p.update_take(tid, metadata_copied=True, source_beats=source_beats)
            except Exception:  # 写せなくても、中身 (音) は返す
                log.exception("cover.mid にメタデータを写せませんでした: %s", path)
        name = f"{safe_name(p.data['title'])}_{take['name'] or tid}.mid"
        return FileResponse(path, media_type="audio/midi", headers={"Content-Disposition": _attachment(name)})

    @app.post("/api/projects/{pid}/takes/{tid}/score")
    def score_take(pid: str, tid: str, body: ScoreIn):
        """テイクを楽譜 (MusicXML) にするジョブを始める。原曲の拍 (SheetSage2-Refined) が要るので音源のあるプロジェクトだけ"""
        p = open_project(pid)
        take = find_take(p, tid)
        if take["state"] != "done":
            raise HTTPException(400, "生成が終わったテイクだけを楽譜にできます")
        if p.audio_path is None and not (p.dir / "sheetsage.json").is_file():
            raise HTTPException(400, "楽譜にするには原曲の拍が要ります。音源のあるプロジェクトで使ってください")
        if any(j.kind == "score" and tid in j.takes for j in runner.active_for(pid)):
            raise HTTPException(409, "このテイクは楽譜にしているところです")
        try:
            entry = engine.score_entry(body.model)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        score = {"state": "queued", "error": None, "model": {"id": entry.id, "label": entry.label}}
        p.update_take(tid, score={**(take.get("score") or {}), **score})
        runner.submit("score", pid, [tid])
        return detail(p)

    @app.get("/api/projects/{pid}/takes/{tid}/score.musicxml")
    def get_take_score(pid: str, tid: str):
        p = open_project(pid)
        take = find_take(p, tid)
        path = p.take_dir(tid) / "score.musicxml"
        if not path.is_file():
            raise HTTPException(404, "まだ楽譜にしていません")
        name = f"{safe_name(p.data['title'])}_{take['name'] or tid}.musicxml"
        return FileResponse(
            path,
            media_type="application/vnd.recordare.musicxml+xml",
            headers={"Content-Disposition": _attachment(name), "Cache-Control": "no-cache"},
        )

    @app.get("/api/projects/{pid}/takes/{tid}/view")
    def get_take_view(pid: str, tid: str):
        from .midi_view import cover_view

        p = open_project(pid)
        find_take(p, tid)
        path = p.take_midi(tid)
        if not path.is_file():
            raise HTTPException(404, "まだ生成していません")
        return cover_view(path)

    # ------------------------------------------------------------ 好み
    def style_view(style: dict) -> dict:
        """一覧に返す形 (ベクトルは除き、学習中なら進み具合を付ける)"""
        view = {k: v for k, v in style.items() if k != "vector"}
        job = next((j for j in runner.jobs("") if j.kind == "style" and j.takes[0] == style["id"]), None)
        if job is not None and job.active:
            view.update(job=job.id, state=job.state, progress=job.progress, message=job.message)
        if job is not None:
            view["log"] = job.log[-40:]
        return view

    def favorite_takes() -> list[dict]:
        out = []
        for p in projects(root):
            for take in p.takes():
                if take.get("favorite") and take["state"] == "done":
                    out.append(
                        {
                            "project": p.id,
                            "title": p.data["title"],
                            "take": take["id"],
                            "number": take["number"],
                            "name": take["name"],
                            "model": take["model"]["id"],
                            "channel": int(take["params"].get("channel") or 0),
                            "style": take["params"].get("style"),
                            "audio": p.audio_path is not None,
                        }
                    )
        return out

    @app.get("/api/styles")
    def list_styles():
        return {"styles": [style_view(s) for s in styles.list()], "favorites": favorite_takes()}

    @app.post("/api/styles")
    def create_style(body: StyleIn):
        """今のお気に入り (全曲の、音源のある曲のもの) から、好みのベクトルを学習するジョブを始める"""
        try:
            entry = engine.entry(body.model)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        favorites = [f for f in favorite_takes() if f["audio"]]
        if not favorites:
            raise HTTPException(400, "音源のある曲のお気に入りがありません")
        # 初期値: このモデルで作ったお気に入りの演奏者 (なければ全部のお気に入りの演奏者) の埋め込みの平均。
        # 好みで作ったテイクの演奏者の番号は意味がないので数えない
        numbered = [f for f in favorites if not f["style"]]
        init = [f["channel"] for f in numbered if f["model"] == entry.id] or [f["channel"] for f in numbered] or [0]
        if entry.num_channels:
            init = [c if c < entry.num_channels else 0 for c in init]
        name = body.name.strip() or f"好み {len(styles.list()) + 1}"
        style = styles.create(
            name,
            {"id": entry.id, "label": entry.label},
            [{"project": f["project"], "take": f["take"]} for f in favorites],
            init,
        )
        runner.submit("style", "", [style["id"]])
        return style_view(style)

    def find_style(sid: str) -> dict:
        try:
            return styles.get(sid)
        except KeyError as e:
            raise HTTPException(404, f"好みがありません: {sid}") from e

    @app.patch("/api/styles/{sid}")
    def rename_style(sid: str, body: StylePatch):
        find_style(sid)
        if not body.name.strip():
            raise HTTPException(400, "名前が空です")
        return style_view(styles.update(sid, name=body.name.strip()))

    @app.delete("/api/styles/{sid}", status_code=204)
    def delete_style(sid: str):
        find_style(sid)
        if any(j.kind == "style" and j.takes[0] == sid for j in runner.active_for("")):
            raise HTTPException(409, "学習中・待ちの好みは先に止めてください")
        styles.delete(sid)

    # ------------------------------------------------------------ Project AI Lyrics Transcription
    @app.get("/api/projects/{pid}/lyrics")
    def get_project_lyrics(pid: str):
        p = open_project(pid)
        return project_lyrics_view(p)

    @app.post("/api/projects/{pid}/lyrics/extract")
    def extract_project_lyrics(pid: str, body: LyricsExtractIn):
        p = open_project(pid)
        if p.audio_path is None or not p.audio_path.is_file():
            raise HTTPException(
                400,
                "Project này chỉ có MIDI, không có tệp âm thanh gốc chứa giọng hát để trích xuất lời.",
            )
        if any(j.kind == "lyrics" for j in runner.active_for(pid)):
            raise HTTPException(409, "Đang có tác vụ xử lý lời bài hát cho Project này.")
        lstore = ProjectLyricsStore(p)
        lstore.update(
            state="queued",
            mode="extract",
            error=None,
            requested_language=body.language or "auto",
            model=body.model or "turbo",
            isolate_vocals=bool(body.isolate_vocals),
            force_separate=bool(body.force_separate),
        )
        runner.submit("lyrics", pid)
        return project_lyrics_view(p)

    @app.post("/api/projects/{pid}/lyrics/align")
    def align_project_lyrics(pid: str, body: LyricsAlignIn):
        p = open_project(pid)
        if p.audio_path is None or not p.audio_path.is_file():
            raise HTTPException(
                400,
                "Project này chỉ có MIDI, không có tệp âm thanh gốc chứa giọng hát để căn chỉnh lời.",
            )
        if not body.lyrics_text.strip():
            raise HTTPException(400, "Vui lòng nhập hoặc dán lời bài hát để căn chỉnh dấu thời gian.")
        if any(j.kind == "lyrics" for j in runner.active_for(pid)):
            raise HTTPException(409, "Đang có tác vụ xử lý lời bài hát cho Project này.")
        lstore = ProjectLyricsStore(p)
        lstore.dir.mkdir(parents=True, exist_ok=True)
        (lstore.dir / "input_lyrics.txt").write_text(body.lyrics_text, encoding="utf-8")
        lstore.update(
            state="queued",
            mode="align",
            error=None,
            requested_language=body.language or "auto",
            model=body.model or "turbo",
            isolate_vocals=bool(body.isolate_vocals),
            force_separate=False,
        )
        runner.submit("lyrics", pid)
        return project_lyrics_view(p)

    @app.put("/api/projects/{pid}/lyrics")
    def save_project_lyrics(pid: str, body: LyricsSaveIn):
        p = open_project(pid)
        lstore = ProjectLyricsStore(p)
        current = lstore.get()
        transcript = current.get("transcript") or {}
        fallback_lang = str(transcript.get("language") or current.get("language") or "en")
        cleaned = sanitize_edited_segments(body.segments, fallback_lang=fallback_lang)
        new_transcript = {
            **transcript,
            "language": fallback_lang,
            "language_name": transcript.get("language_name") or language_label(fallback_lang),
            "duration": transcript.get("duration") or current.get("duration") or 0.0,
            "segments": cleaned,
            "full_text": build_full_text(cleaned, include_sections=False),
            "structured_text": build_full_text(cleaned, include_sections=True),
        }
        lstore.update(
            transcript=new_transcript,
            state="done",
            error=None,
            language=fallback_lang,
            language_name=new_transcript["language_name"],
            segment_count=len(cleaned),
        )
        return project_lyrics_view(p)

    @app.delete("/api/projects/{pid}/lyrics", status_code=204)
    def delete_project_lyrics(pid: str):
        p = open_project(pid)
        if any(j.kind == "lyrics" for j in runner.active_for(pid)):
            raise HTTPException(409, "Tác vụ đang chạy, hãy bấm dừng trước khi xóa.")
        ProjectLyricsStore(p).clear()

    # ------------------------------------------------------------ Songscription
    def songscription_view(item: dict) -> dict:
        view = dict(item)
        sha1 = view.get("audio_sha1")
        if sha1:
            view["vocal_cached"] = songscriptions.cache.has_vocal(sha1)
        job = next((j for j in runner.jobs("") if j.kind == "songscription" and j.takes[0] == item["id"]), None)
        if job is not None and job.active:
            view.update(job=job.id, state=job.state, progress=job.progress, message=job.message)
        if job is not None:
            view["log"] = job.log[-50:]
        return view

    def find_songscription(sid: str) -> dict:
        try:
            return songscriptions.get(sid)
        except KeyError as e:
            raise HTTPException(404, f"Không tìm thấy Songscription: {sid}") from e

    @app.get("/api/songscriptions")
    def list_songscriptions():
        return {
            "models": list(SONGSCRIPTION_MODELS),
            "vocal_isolation_available": engine.tsumugi is not None,
            "items": [songscription_view(item) for item in songscriptions.list()],
        }

    @app.post("/api/songscriptions")
    def create_songscription(
        audio: Annotated[UploadFile | None, File()] = None,
        youtube: Annotated[str, Form()] = "",
        project_id: Annotated[str, Form()] = "",
        title: Annotated[str, Form()] = "",
        mode: Annotated[str, Form()] = "extract",
        lyrics_text: Annotated[str, Form()] = "",
        language: Annotated[str, Form()] = "auto",
        model: Annotated[str, Form()] = "turbo",
        isolate_vocals: Annotated[bool, Form()] = True,
    ):
        youtube = youtube.strip()
        project_id = project_id.strip()
        sources = sum([audio is not None, bool(youtube), bool(project_id)])
        if sources == 0:
            raise HTTPException(400, "Hãy chọn file âm thanh, dán link YouTube, hoặc chọn một Project có sẵn.")
        if sources > 1:
            raise HTTPException(400, "Chỉ chọn một nguồn âm thanh tại một thời điểm.")
        if mode == "align" and not lyrics_text.strip():
            raise HTTPException(400, "Vui lòng nhập hoặc dán lời bài hát khi ở chế độ Căn chỉnh lời (Lyrics Alignment).")

        uploaded_path: Path | None = None
        with tempfile.TemporaryDirectory(prefix="songscription-in-") as temp_dir:
            try:
                if project_id:
                    proj = open_project(project_id)
                    if proj.audio_path is None or not proj.audio_path.is_file():
                        raise HTTPException(400, "Project này không có tệp âm thanh gốc để trích xuất lời.")
                    source_audio = proj.audio_path
                    resolved_title = title.strip() or proj.data["title"]
                    source_kind = "project"
                elif youtube:
                    try:
                        source_audio, video_title = download_youtube_audio(youtube, Path(temp_dir))
                    except YouTubeImportError as exc:
                        raise HTTPException(exc.status_code, str(exc)) from exc
                    resolved_title = title.strip() or video_title
                    source_kind = "youtube"
                else:
                    assert audio is not None
                    uploaded_path = save_upload(audio, MEDIA_EXTS)
                    source_audio = uploaded_path
                    resolved_title = title.strip() or Path(uploaded_path.name).stem
                    source_kind = "upload"

                record = songscriptions.create(
                    resolved_title,
                    source_audio,
                    source_kind=source_kind,
                    project_id=project_id or None,
                    mode=mode,
                    lyrics_text=lyrics_text,
                    requested_language=language,
                    model_id=model,
                    isolate_vocals=isolate_vocals,
                )
            finally:
                if uploaded_path is not None and uploaded_path.is_relative_to(root / ".uploads"):
                    shutil.rmtree(uploaded_path.parent, ignore_errors=True)

        runner.submit("songscription", "", [record["id"]])
        return songscription_view(songscriptions.get(record["id"]))

    @app.get("/api/songscriptions/{sid}")
    def get_songscription(sid: str):
        return songscription_view(find_songscription(sid))

    @app.post("/api/songscriptions/{sid}/retry")
    def retry_songscription(sid: str):
        find_songscription(sid)
        if any(j.kind == "songscription" and j.takes[0] == sid for j in runner.active_for("")):
            raise HTTPException(409, "Tác vụ đang chạy.")
        songscriptions.update(
            sid,
            state="queued",
            mode="extract",
            isolate_vocals=engine.tsumugi is not None,
            error=None,
        )
        runner.submit("songscription", "", [sid])
        return songscription_view(songscriptions.get(sid))

    @app.post("/api/songscriptions/{sid}/align")
    def align_existing_songscription(sid: str, body: SongscriptionAlignIn):
        find_songscription(sid)
        if not body.lyrics_text.strip():
            raise HTTPException(400, "Vui lòng nhập hoặc dán lời bài hát để căn chỉnh.")
        if any(j.kind == "songscription" and j.takes[0] == sid for j in runner.active_for("")):
            raise HTTPException(409, "Tác vụ đang chạy.")
        songscriptions.set_input_lyrics_text(sid, body.lyrics_text)
        songscriptions.update(
            sid,
            state="queued",
            mode="align",
            error=None,
            requested_language=body.language or "auto",
            model=body.model or "turbo",
            isolate_vocals=bool(body.isolate_vocals),
        )
        runner.submit("songscription", "", [sid])
        return songscription_view(songscriptions.get(sid))

    @app.patch("/api/songscriptions/{sid}")
    def update_songscription(sid: str, body: SongscriptionPatch):
        find_songscription(sid)
        if body.segments is not None:
            saved = songscriptions.save_edited_segments(sid, body.segments, title=body.title)
            return songscription_view(saved)
        updates: dict = {}
        if body.title is not None and body.title.strip():
            updates["title"] = body.title.strip()
            updates["updated"] = now()
        return songscription_view(songscriptions.update(sid, **updates))

    @app.post("/api/songscriptions/{sid}/retranscribe-flagged")
    def retranscribe_flagged_songscription(sid: str, body: SongscriptionRetranscribeIn | None = None):
        find_songscription(sid)
        if any(j.kind == "songscription" and j.takes[0] == sid for j in runner.active_for("")):
            raise HTTPException(409, "Phiên Songscription này đang chạy tác vụ khác, vui lòng đợi hoàn tất.")
        payload = body or SongscriptionRetranscribeIn()
        try:
            out = songscriptions.retranscribe_flagged(
                sid,
                segment_ids=payload.segment_ids,
                model_id=payload.model,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(404, str(e)) from e
        out["item"] = songscription_view(out["item"])
        return out

    @app.get("/api/songscriptions/{sid}/audio")
    def get_songscription_audio(sid: str):
        find_songscription(sid)
        audio_file = songscriptions.audio_path(sid)
        if audio_file is None or not audio_file.is_file():
            raise HTTPException(404, "Không tìm thấy tệp âm thanh.")
        return FileResponse(audio_file)

    @app.delete("/api/songscriptions/{sid}", status_code=204)
    def delete_songscription(sid: str):
        find_songscription(sid)
        if any(j.kind == "songscription" and j.takes[0] == sid for j in runner.active_for("")):
            raise HTTPException(409, "Tác vụ đang chạy, hãy bấm dừng trước khi xóa.")
        songscriptions.delete(sid)

    # ------------------------------------------------------------ ジョブ
    @app.get("/api/jobs")
    def list_jobs(project: str | None = None):
        return [j.to_dict() for j in runner.jobs(project)]

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        job = runner.cancel(job_id)
        if job is None:
            raise HTTPException(404, f"ジョブがありません: {job_id}")
        return job.to_dict()

    # ------------------------------------------------------------ 画面
    @app.exception_handler(ValueError)
    def value_error(_, exc: ValueError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/{path:path}", include_in_schema=False)
    def web(path: str):
        """music-frontend/ のビルドを配る。ファイルでないパス (/p/<id> など画面の中の行き先) には index.html を返す"""
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404)
        if not (STATIC_DIR / "index.html").exists():
            return RedirectResponse("/docs")
        file = (STATIC_DIR / path).resolve()
        if path and file.is_file() and file.is_relative_to(STATIC_DIR.resolve()):
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    return app


def _attachment(name: str) -> str:
    return f"attachment; filename*=UTF-8''{quote(name)}"
