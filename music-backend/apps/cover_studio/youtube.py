"""Download one YouTube video's audio for a Cover Studio project."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

MAX_DURATION_SECONDS = 20 * 60
MAX_AUDIO_BYTES = 100 * 1024 * 1024
VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")


class YouTubeImportError(Exception):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def extract_youtube_id(value: str | None) -> str | None:
    """Return the 11-character YouTube video ID if `value` is a valid YouTube URL, else None."""
    if not value or not value.strip():
        return None
    try:
        normalized = normalize_youtube_url(value)
    except YouTubeImportError:
        return None
    vid = parse_qs(urlsplit(normalized).query).get("v", [""])[0]
    return vid if VIDEO_ID.fullmatch(vid) else None


def normalize_youtube_url(value: str) -> str:
    """Accept video URLs only, then discard playlist and tracking parameters."""
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname or ""
        port = parsed.port
    except ValueError as exc:
        raise YouTubeImportError("Link YouTube không hợp lệ.") from exc

    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password or port:
        raise YouTubeImportError("Link YouTube không hợp lệ.")

    if host in ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"):
        parts = parsed.path.strip("/").split("/")
        if parsed.path == "/watch":
            video_id = parse_qs(parsed.query).get("v", [""])[0]
        elif len(parts) == 2 and parts[0] in ("shorts", "live", "embed"):
            video_id = parts[1]
        else:
            video_id = ""
    elif host in ("youtu.be", "www.youtu.be"):
        video_id = parsed.path.strip("/")
    else:
        video_id = ""

    if not VIDEO_ID.fullmatch(video_id):
        raise YouTubeImportError("Hãy dán link của một video YouTube hợp lệ.")
    return f"https://www.youtube.com/watch?v={video_id}"


def _ffmpeg_dir() -> str | None:
    bundled = Path(__file__).resolve().parents[3] / ".tsumugi" / "ffmpeg-shared" / "bin"
    if (bundled / "ffmpeg.exe").is_file() and (bundled / "ffprobe.exe").is_file():
        return str(bundled)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg and ffprobe:
        return str(Path(ffmpeg).parent)
    return None


def download_youtube_audio(value: str, directory: Path) -> tuple[Path, str]:
    url = normalize_youtube_url(value)
    ffmpeg_dir = _ffmpeg_dir()
    if not ffmpeg_dir:
        raise YouTubeImportError("Cần FFmpeg và FFprobe để nhập link YouTube.", 503)

    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise YouTubeImportError("Backend chưa cài yt-dlp[default].", 503) from exc

    def check_video(info: dict, *, incomplete: bool) -> str | None:
        if incomplete:
            return None
        if info.get("is_live") or info.get("live_status") == "is_live":
            return "Không hỗ trợ video đang phát trực tiếp."
        duration = info.get("duration")
        if duration is None or duration > MAX_DURATION_SECONDS:
            return "Chỉ hỗ trợ video có thời lượng tối đa 20 phút."
        return None

    directory.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(directory / "source.%(ext)s"),
        "noplaylist": True,
        "max_filesize": MAX_AUDIO_BYTES,
        "match_filter": check_video,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}],
        "ffmpeg_location": ffmpeg_dir,
        "js_runtimes": {"deno": {}, "node": {}},
        "socket_timeout": 20,
        "retries": 2,
        "fragment_retries": 2,
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "nopart": True,
    }
    try:
        with YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=True)
    except DownloadError as exc:
        raise YouTubeImportError(f"Không tải được âm thanh YouTube: {exc}", 400) from exc

    audio = directory / "source.mp3"
    if not info or not audio.is_file():
        raise YouTubeImportError("YouTube không trả về file âm thanh.", 400)
    if audio.stat().st_size > MAX_AUDIO_BYTES:
        raise YouTubeImportError("File âm thanh phải nhỏ hơn 100 MB.", 413)
    return audio, str(info.get("title") or "YouTube video")[:200]
