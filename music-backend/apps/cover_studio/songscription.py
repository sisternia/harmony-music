"""Songscription Store & Job Bridge — powered by the unified Harmony Auralis AI Lyrics Service.

Supports standalone audio uploads, YouTube links, and direct Project audio reuse (without duplicating
large audio files when referencing an existing Project), in both Automatic Extraction (Mode 1)
and User-Provided Lyrics Alignment (Mode 2).
"""

from __future__ import annotations

import shutil
import threading
import uuid
from collections.abc import Callable
from pathlib import Path

from .engine import Engine
from .lyrics_service import (
    LYRICS_MODELS,
    LyricsCache,
    ProjectLyricsStore,
    build_full_text,
    execute_lyrics_pipeline,
    language_label,
    retranscribe_flagged_segments,
    sanitize_edited_segments,
)
from .project import Project, file_sha1, now, read_json, safe_name, write_json

SONGSCRIPTION_MODELS = LYRICS_MODELS


class SongscriptionStore:
    """Persist standalone & project-linked Songscription sessions at `<projects_root>/.songscriptions/<sid>/`."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.dir = self.root / ".songscriptions"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cache = LyricsCache(self.root)
        self._lock = threading.RLock()

    def _item_dir(self, sid: str) -> Path:
        d = self.dir / sid
        if not sid or Path(sid).name != sid or not (d / "item.json").is_file():
            raise KeyError(sid)
        return d

    def _review_stats(self, d: Path) -> dict:
        """Derive real review counts from saved transcript.json without fabricating values."""
        transcript_path = d / "transcript.json"
        if not transcript_path.is_file():
            return {"review_needs_count": 0, "review_done_count": 0}
        try:
            tr = read_json(transcript_path)
            segs = tr.get("segments") or []
            needs = 0
            done = 0
            for seg in segs:
                rs = seg.get("review_status")
                if rs in ("verified", "edited", "skipped"):
                    done += 1
                elif rs == "needs_review" or bool(seg.get("low_confidence")):
                    needs += 1
                else:
                    done += 1
            return {"review_needs_count": needs, "review_done_count": done}
        except Exception:
            return {"review_needs_count": 0, "review_done_count": 0}

    def list(self) -> list[dict]:
        with self._lock:
            items: list[dict] = []
            if self.dir.is_dir():
                for d in self.dir.iterdir():
                    meta = d / "item.json"
                    if meta.is_file():
                        try:
                            data = read_json(meta)
                            data.update(self._review_stats(d))
                            items.append(data)
                        except Exception:
                            continue
            return sorted(items, key=lambda x: x.get("updated") or x.get("completed") or x.get("created", ""), reverse=True)

    def get(self, sid: str) -> dict:
        with self._lock:
            d = self._item_dir(sid)
            data = read_json(d / "item.json")
            data.update(self._review_stats(d))
            transcript_path = d / "transcript.json"
            if transcript_path.is_file():
                try:
                    data["transcript"] = read_json(transcript_path)
                except Exception:
                    data["transcript"] = None
            return data

    def audio_path(self, sid: str) -> Path | None:
        d = self._item_dir(sid)
        data = read_json(d / "item.json")
        # Reuse project audio directly without copying if linked to an existing project
        pid = data.get("project_id")
        if pid and Path(pid).name == pid:
            p_json = self.root / pid / "project.json"
            if p_json.is_file():
                try:
                    proj = Project(self.root / pid)
                    if proj.audio_path and proj.audio_path.is_file():
                        return proj.audio_path
                except Exception:
                    pass
        rel = data.get("audio_file")
        if rel:
            p = (d / rel).resolve()
            if p.is_relative_to(d.resolve()) and p.is_file():
                return p
        return None

    def create(
        self,
        title: str,
        source_audio: Path,
        *,
        source_kind: str,
        project_id: str | None = None,
        mode: str = "extract",
        lyrics_text: str = "",
        requested_language: str = "auto",
        model_id: str = "turbo",
        isolate_vocals: bool = True,
    ) -> dict:
        with self._lock:
            sid = uuid.uuid4().hex[:12]
            d = self.dir / sid
            d.mkdir(parents=True, exist_ok=True)
            audio_name: str | None = None
            if source_kind != "project":
                ext = source_audio.suffix.lower() or ".mp3"
                audio_name = f"source{ext}"
                shutil.copy2(source_audio, d / audio_name)
            sha1 = file_sha1(source_audio)
            if lyrics_text.strip():
                (d / "input_lyrics.txt").write_text(lyrics_text, encoding="utf-8")
            record = {
                "id": sid,
                "title": title.strip() or safe_name(source_audio.stem),
                "created": now(),
                "state": "queued",
                "mode": "align" if mode == "align" else "extract",
                "error": None,
                "source_kind": source_kind,
                "project_id": project_id,
                "audio_sha1": sha1,
                "requested_language": requested_language or "auto",
                "model": model_id,
                "isolate_vocals": bool(isolate_vocals),
                "audio_file": audio_name,
                "language": None,
                "language_name": None,
                "language_confidence": None,
                "detected_languages": [],
                "vocal_isolated": False,
                "vocal_cached": self.cache.has_vocal(sha1),
                "alignment_method": None,
                "duration": None,
                "segment_count": 0,
            }
            write_json(d / "item.json", record)
            return record

    def input_lyrics_text(self, sid: str) -> str:
        d = self._item_dir(sid)
        p = d / "input_lyrics.txt"
        return p.read_text(encoding="utf-8") if p.is_file() else ""

    def set_input_lyrics_text(self, sid: str, text: str) -> None:
        d = self._item_dir(sid)
        (d / "input_lyrics.txt").write_text(text, encoding="utf-8")

    def update(self, sid: str, transcript: dict | None = None, **fields) -> dict:
        with self._lock:
            d = self._item_dir(sid)
            data = read_json(d / "item.json")
            data.update(fields)
            write_json(d / "item.json", data)
            if transcript is not None:
                write_json(d / "transcript.json", transcript)
                data["transcript"] = transcript
            return data

    def save_edited_segments(self, sid: str, raw_segments: list[dict], title: str | None = None) -> dict:
        item = self.get(sid)
        transcript = item.get("transcript") or {}
        fallback_lang = str(transcript.get("language") or item.get("language") or "en")
        cleaned = sanitize_edited_segments(raw_segments, fallback_lang=fallback_lang)
        new_transcript = {
            **transcript,
            "language": fallback_lang,
            "language_name": transcript.get("language_name") or language_label(fallback_lang),
            "segments": cleaned,
            "full_text": build_full_text(cleaned, include_sections=False),
            "structured_text": build_full_text(cleaned, include_sections=True),
        }
        updates: dict = {"segment_count": len(cleaned), "updated": now()}
        if title is not None and title.strip():
            updates["title"] = title.strip()
        saved = self.update(sid, transcript=new_transcript, **updates)

        # Sync back to linked Project if applicable
        pid = saved.get("project_id")
        if pid and Path(pid).name == pid and (self.root / pid / "project.json").is_file():
            try:
                proj = Project(self.root / pid)
                ProjectLyricsStore(proj).update(
                    transcript=new_transcript,
                    state="done",
                    segment_count=len(cleaned),
                )
            except Exception:
                pass
        return saved

    def retranscribe_flagged(
        self,
        sid: str,
        *,
        segment_ids: list[int] | None = None,
        model_id: str | None = None,
    ) -> dict:
        """Precision multi-pass re-transcription for flagged ('Cần kiểm tra') or specified line IDs."""
        item = self.get(sid)
        transcript = item.get("transcript")
        if not transcript or not isinstance(transcript.get("segments"), list):
            raise ValueError("Phiên Songscription này chưa có lời bài hát để đọc lại.")
        audio_src = self.audio_path(sid)
        if audio_src is None or not audio_src.is_file():
            raise FileNotFoundError("Không tìm thấy tệp âm thanh nguồn để đọc lại lời bài hát.")

        lang = str(transcript.get("language") or item.get("language") or item.get("requested_language") or "auto")
        eff_model = model_id or str(item.get("model") or "turbo")
        isolate = bool(item.get("isolate_vocals", True))

        updated_segs, results = retranscribe_flagged_segments(
            cache=self.cache,
            source_audio=audio_src,
            audio_sha1=item.get("audio_sha1"),
            segments=transcript["segments"],
            target_ids=segment_ids,
            language=lang,
            model_id=eff_model,
            isolate_vocals=isolate,
        )

        if results:
            saved = self.save_edited_segments(sid, updated_segs)
        else:
            saved = self.get(sid)

        improved_count = sum(1 for r in results if r.get("improved"))
        auto_resolved_count = sum(1 for r in results if r.get("auto_resolved"))
        return {
            "item": saved,
            "retranscribed_count": len(results),
            "improved_count": improved_count,
            "auto_resolved_count": auto_resolved_count,
            "results": results,
        }

    def delete(self, sid: str) -> None:
        with self._lock:
            d = self._item_dir(sid)
            shutil.rmtree(d, ignore_errors=True)

    def recover(self) -> None:
        with self._lock:
            if not self.dir.is_dir():
                return
            for d in self.dir.iterdir():
                meta = d / "item.json"
                if not meta.is_file():
                    continue
                try:
                    data = read_json(meta)
                    if data.get("state") in ("queued", "running"):
                        data["state"] = "cancelled"
                        data["error"] = "Máy chủ khởi động lại nên tác vụ đã dừng."
                        write_json(meta, data)
                except Exception:
                    continue


def run_songscription_job(
    store: SongscriptionStore,
    engine: Engine,
    sid: str,
    chords: list[tuple[float, str]] | None,
    progress: Callable[[float | None, str], None],
    should_stop: Callable[[], bool],
    log: Callable[[str], None],
) -> None:
    item = store.get(sid)
    audio_src = store.audio_path(sid)
    if audio_src is None or not audio_src.is_file():
        raise FileNotFoundError("Không tìm thấy tệp âm thanh nguồn để xử lý lời bài hát.")

    store.update(sid, state="running", error=None)
    mode = "align" if item.get("mode") == "align" else "extract"
    lyrics_text = store.input_lyrics_text(sid) if mode == "align" else ""

    transcript = execute_lyrics_pipeline(
        engine=engine,
        cache=store.cache,
        source_audio=audio_src,
        audio_sha1=item.get("audio_sha1"),
        mode=mode,
        user_lyrics_text=lyrics_text,
        requested_language=str(item.get("requested_language") or "auto"),
        model_id=str(item.get("model") or "turbo"),
        isolate_vocals=bool(item.get("isolate_vocals", True)),
        chords=chords,
        progress=progress,
        should_stop=should_stop,
        log=log,
    )

    summary_fields = {
        "state": "done",
        "error": None,
        "completed": now(),
        "language": transcript["language"],
        "language_name": transcript["language_name"],
        "language_confidence": transcript["language_confidence"],
        "detected_languages": transcript["detected_languages"],
        "vocal_isolated": transcript["vocal_isolated"],
        "vocal_cached": transcript["vocal_cached"],
        "alignment_method": transcript["alignment_method"],
        "duration": transcript["duration"],
        "segment_count": len(transcript["segments"]),
    }
    store.update(sid, transcript=transcript, **summary_fields)
    progress(1.0, "completed")

    # Sync to ProjectLyricsStore when linked to a project
    pid = item.get("project_id")
    if pid and Path(pid).name == pid and (store.root / pid / "project.json").is_file():
        try:
            proj = Project(store.root / pid)
            ProjectLyricsStore(proj).update(
                transcript=transcript,
                mode=mode,
                model=item.get("model") or "turbo",
                requested_language=item.get("requested_language") or "auto",
                isolate_vocals=bool(item.get("isolate_vocals", True)),
                **summary_fields,
            )
        except Exception:
            pass
