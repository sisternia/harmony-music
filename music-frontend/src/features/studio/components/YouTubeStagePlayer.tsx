import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { SongscriptionSegment, SongscriptionWord } from "../api";
import { getWordPlaybackState } from "../utils/lyricsPlayback";

type YTPlayerInstance = {
  playVideo: () => void;
  pauseVideo: () => void;
  seekTo: (seconds: number, allowSeekAhead?: boolean) => void;
  mute: () => void;
  unMute: () => void;
  isMuted?: () => boolean;
  getVolume?: () => number;
  setVolume?: (volume: number) => void;
  setPlaybackRate: (rate: number) => void;
  getPlaybackRate?: () => number;
  getCurrentTime: () => number;
  getDuration?: () => number;
  getPlayerState: () => number;
  destroy: () => void;
};

type YTNamespace = {
  Player: new (
    element: HTMLElement,
    options: {
      videoId: string;
      playerVars?: Record<string, number | string>;
      events?: {
        onReady?: (event: { target: YTPlayerInstance }) => void;
        onStateChange?: (event: { target: YTPlayerInstance; data: number }) => void;
        onPlaybackRateChange?: (event: { target: YTPlayerInstance; data: number }) => void;
      };
    },
  ) => YTPlayerInstance;
};

declare global {
  interface Window {
    YT?: YTNamespace;
    onYouTubeIframeAPIReady?: () => void;
  }
}

let ytApiPromise: Promise<YTNamespace> | null = null;

function loadYouTubeIframeApi(): Promise<YTNamespace> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("No window"));
  }
  if (window.YT && typeof window.YT.Player === "function") {
    return Promise.resolve(window.YT);
  }
  if (ytApiPromise) return ytApiPromise;

  ytApiPromise = new Promise<YTNamespace>((resolve, reject) => {
    const prevReady = window.onYouTubeIframeAPIReady;
    window.onYouTubeIframeAPIReady = () => {
      prevReady?.();
      if (window.YT) resolve(window.YT);
    };
    const existing = document.querySelector<HTMLScriptElement>('script[src="https://www.youtube.com/iframe_api"]');
    if (!existing) {
      const script = document.createElement("script");
      script.src = "https://www.youtube.com/iframe_api";
      script.async = true;
      script.onerror = () => {
        ytApiPromise = null;
        reject(new Error("Không thể tải YouTube IFrame API"));
      };
      document.head.appendChild(script);
    }
  });
  return ytApiPromise;
}

export function extractYoutubeVideoId(urlOrId?: string | null): string | null {
  if (!urlOrId) return null;
  const trimmed = urlOrId.trim();
  if (/^[A-Za-z0-9_-]{11}$/.test(trimmed)) return trimmed;
  try {
    const parsed = new URL(trimmed);
    const host = parsed.hostname.toLowerCase();
    if (host.includes("youtube.com")) {
      const v = parsed.searchParams.get("v");
      if (v && /^[A-Za-z0-9_-]{11}$/.test(v)) return v;
      const parts = parsed.pathname.split("/").filter(Boolean);
      if (parts.length >= 2 && ["shorts", "live", "embed"].includes(parts[0]) && /^[A-Za-z0-9_-]{11}$/.test(parts[1])) {
        return parts[1];
      }
    } else if (host === "youtu.be" || host === "www.youtu.be") {
      const id = parsed.pathname.replace(/^\/+/, "").split("/")[0];
      if (/^[A-Za-z0-9_-]{11}$/.test(id)) return id;
    }
  } catch {
    return null;
  }
  return null;
}

export type YouTubeStagePlayerHandle = {
  play: () => void;
  pause: () => void;
  togglePlay: () => void;
  seekTo: (seconds: number, autoPlay?: boolean) => void;
  setPlaybackRate: (rate: number) => void;
  getCurrentTime: () => number;
};

export type YouTubeStagePlayerProps = {
  videoId: string;
  title: string;
  /** True when listening to separated Vocal / Instrumental / Solo stem so YouTube original audio stays muted */
  muteYouTube: boolean;
  /** Authoritative playback position when muteYouTube is true (local stem mode) */
  externalPosition: number;
  /** Authoritative playing state when muteYouTube is true (local stem mode) */
  externalPlaying: boolean;
  /** True when player is kept alive in the background while user is viewing another in-app screen */
  isBackground?: boolean;
  /** Shared volume (0..1) mirrored into YouTube when returning to Bản gốc */
  externalVolume?: number;
  /** Shared mute state mirrored into YouTube when returning to Bản gốc */
  externalMuted?: boolean;
  playbackRate: number;
  showChords: boolean;
  showWordHighlight: boolean;
  activeSegments: SongscriptionSegment[];
  karaokeWordsBySegId: Map<number, SongscriptionWord[] | undefined>;
  syncTime: number;
  onYouTubeTimeUpdate: (timeSec: number) => void;
  onYouTubePlayingChange: (isPlaying: boolean, timeSec: number) => void;
  onYouTubeDurationChange?: (durationSec: number) => void;
  onYouTubeRateChange?: (rate: number) => void;
  /** Fired when user manually scrubs YouTube's native timeline while in Vocal/Instrumental mode */
  onYouTubeManualSeek?: (timeSec: number) => void;
  /** Fired in Bản gốc mode when user adjusts YouTube's native volume slider or mute button */
  onYouTubeVolumeChange?: (volume01: number, muted: boolean) => void;
};

export const YouTubeStagePlayer = forwardRef<YouTubeStagePlayerHandle, YouTubeStagePlayerProps>(
  function YouTubeStagePlayer(
    {
      videoId,
      title,
      muteYouTube,
      externalPosition,
      externalPlaying,
      isBackground = false,
      externalVolume = 1,
      externalMuted = false,
      playbackRate,
      showChords,
      showWordHighlight,
      activeSegments,
      karaokeWordsBySegId,
      syncTime,
      onYouTubeTimeUpdate,
      onYouTubePlayingChange,
      onYouTubeDurationChange,
      onYouTubeRateChange,
      onYouTubeManualSeek,
      onYouTubeVolumeChange,
    },
    ref,
  ) {
    const hostRef = useRef<HTMLDivElement>(null);
    const playerRef = useRef<YTPlayerInstance | null>(null);
    const readyRef = useRef(false);
    const ytPlayingRef = useRef(false);
    const lastSyncRef = useRef(0);
    const suppressExternalUntilRef = useRef(0);
    const lastPolledYtTimeRef = useRef(0);
    const lastYtVolRef = useRef(Math.round(externalVolume * 100));
    const lastYtMutedRef = useRef(externalMuted);
    const mutePendingUntilRef = useRef(0);
    const lastVisibilityChangeMsRef = useRef(0);
    const prevExternalPosRef = useRef(externalPosition);
    const [ytReady, setYtReady] = useState(false);
    const [ytError, setYtError] = useState(false);

    // Latest callback refs to avoid recreating the YouTube IFrame on prop updates
    const muteRef = useRef(muteYouTube);
    const extPosRef = useRef(externalPosition);
    const extPlayingRef = useRef(externalPlaying);
    const isBackgroundRef = useRef(isBackground);
    const extVolRef = useRef(externalVolume);
    const extMutedRef = useRef(externalMuted);
    const onTimeRef = useRef(onYouTubeTimeUpdate);
    const onPlayRef = useRef(onYouTubePlayingChange);
    const onDurRef = useRef(onYouTubeDurationChange);
    const onRateRef = useRef(onYouTubeRateChange);
    const onManualSeekRef = useRef(onYouTubeManualSeek);
    const onVolRef = useRef(onYouTubeVolumeChange);

    useEffect(() => {
      muteRef.current = muteYouTube;
    }, [muteYouTube]);
    useEffect(() => {
      extPosRef.current = externalPosition;
    }, [externalPosition]);
    useEffect(() => {
      extPlayingRef.current = externalPlaying;
    }, [externalPlaying]);
    useEffect(() => {
      isBackgroundRef.current = isBackground;
    }, [isBackground]);
    useEffect(() => {
      extVolRef.current = externalVolume;
    }, [externalVolume]);
    useEffect(() => {
      extMutedRef.current = externalMuted;
    }, [externalMuted]);
    useEffect(() => {
      onTimeRef.current = onYouTubeTimeUpdate;
    }, [onYouTubeTimeUpdate]);
    useEffect(() => {
      onPlayRef.current = onYouTubePlayingChange;
    }, [onYouTubePlayingChange]);
    useEffect(() => {
      onDurRef.current = onYouTubeDurationChange;
    }, [onYouTubeDurationChange]);
    useEffect(() => {
      onRateRef.current = onYouTubeRateChange;
    }, [onYouTubeRateChange]);
    useEffect(() => {
      onManualSeekRef.current = onYouTubeManualSeek;
    }, [onYouTubeManualSeek]);
    useEffect(() => {
      onVolRef.current = onYouTubeVolumeChange;
    }, [onYouTubeVolumeChange]);

    useImperativeHandle(
      ref,
      () => ({
        play: () => {
          const p = playerRef.current;
          if (!p || !readyRef.current) return;
          try {
            suppressExternalUntilRef.current = performance.now() + 450;
            p.playVideo();
          } catch {
            // Ignore transient player states
          }
        },
        pause: () => {
          const p = playerRef.current;
          if (!p || !readyRef.current) return;
          try {
            suppressExternalUntilRef.current = performance.now() + 450;
            p.pauseVideo();
          } catch {
            // Ignore
          }
        },
        togglePlay: () => {
          const p = playerRef.current;
          if (!p || !readyRef.current) return;
          try {
            suppressExternalUntilRef.current = performance.now() + 450;
            const st = p.getPlayerState?.();
            if (st === 1 || ytPlayingRef.current) {
              p.pauseVideo();
            } else {
              p.playVideo();
            }
          } catch {
            // Ignore
          }
        },
        seekTo: (seconds: number, autoPlay = false) => {
          const p = playerRef.current;
          if (!p || !readyRef.current) return;
          try {
            const clamped = Math.max(0, seconds);
            suppressExternalUntilRef.current = performance.now() + 550;
            lastPolledYtTimeRef.current = clamped;
            p.seekTo(clamped, true);
            if (autoPlay) {
              p.playVideo();
            }
          } catch {
            // Ignore
          }
        },
        setPlaybackRate: (rate: number) => {
          const p = playerRef.current;
          if (!p || !readyRef.current) return;
          try {
            p.setPlaybackRate(rate);
          } catch {
            // Ignore
          }
        },
        getCurrentTime: () => {
          const p = playerRef.current;
          if (!p || !readyRef.current) return 0;
          try {
            const t = Number(p.getCurrentTime?.() ?? 0);
            return Number.isFinite(t) ? t : 0;
          } catch {
            return 0;
          }
        },
      }),
      [],
    );

    // Mount 100% native YouTube IFrame Player (`controls: 1`) with all built-in YouTube controls
    useEffect(() => {
      let cancelled = false;
      readyRef.current = false;

      const mountEl = hostRef.current;
      if (!mountEl || !videoId) return;

      mountEl.innerHTML = "";
      const targetDiv = document.createElement("div");
      mountEl.appendChild(targetDiv);

      loadYouTubeIframeApi()
        .then(YT => {
          if (cancelled || !hostRef.current) return;
          const instance = new YT.Player(targetDiv, {
            videoId,
            playerVars: {
              autoplay: 0,
              controls: 1,
              disablekb: 0,
              fs: 1,
              modestbranding: 1,
              rel: 0,
              playsinline: 1,
              iv_load_policy: 3,
              origin: window.location.origin,
            },
            events: {
              onReady: ev => {
                if (cancelled) {
                  ev.target.destroy();
                  return;
                }
                playerRef.current = ev.target;
                readyRef.current = true;
                if (muteRef.current) {
                  mutePendingUntilRef.current = performance.now() + 450;
                  ev.target.mute();
                } else {
                  const targetVol = Math.round(Math.min(1, Math.max(0, extVolRef.current)) * 100);
                  ev.target.setVolume?.(targetVol);
                  if (extMutedRef.current || extVolRef.current <= 0.001) {
                    ev.target.mute();
                  } else {
                    ev.target.unMute();
                  }
                }
                const d = Number(ev.target.getDuration?.() ?? 0);
                if (Number.isFinite(d) && d > 0) {
                  onDurRef.current?.(d);
                }
                setYtReady(true);
              },
              onStateChange: ev => {
                if (cancelled) return;
                const state = ev.data;
                const cur = Number(ev.target.getCurrentTime?.() ?? 0);
                const safeT = Number.isFinite(cur) ? cur : 0;
                const dur = Number(ev.target.getDuration?.() ?? 0);
                if (Number.isFinite(dur) && dur > 0) {
                  onDurRef.current?.(dur);
                }
                if (state === 1) {
                  ytPlayingRef.current = true;
                  lastPolledYtTimeRef.current = safeT;
                  onPlayRef.current(true, safeT);
                } else if (state === 2 || state === 0) {
                  ytPlayingRef.current = false;
                  lastPolledYtTimeRef.current = safeT;
                  // Ignore Chrome background-tab auto-pause on muted YouTube iframe so Vocal/Instrumental keeps playing
                  if (muteRef.current) {
                    const nowMs = performance.now();
                    const hiddenOrBg =
                      (typeof document !== "undefined" && document.hidden) ||
                      isBackgroundRef.current ||
                      nowMs - lastVisibilityChangeMsRef.current < 900 ||
                      nowMs < suppressExternalUntilRef.current;
                    if (hiddenOrBg) return;
                  }
                  onPlayRef.current(false, safeT);
                } else if (state === 3) {
                  lastPolledYtTimeRef.current = safeT;
                  if (muteRef.current) {
                    if (performance.now() > suppressExternalUntilRef.current) {
                      onManualSeekRef.current?.(safeT);
                    }
                  } else {
                    onTimeRef.current(safeT);
                  }
                }
              },
              onPlaybackRateChange: ev => {
                if (cancelled) return;
                const r = Number(ev.data);
                if (Number.isFinite(r) && r > 0) {
                  onRateRef.current?.(r);
                }
              },
            },
          });
          playerRef.current = instance;
        })
        .catch(() => {
          if (!cancelled) setYtError(true);
        });

      return () => {
        cancelled = true;
        readyRef.current = false;
        ytPlayingRef.current = false;
        try {
          playerRef.current?.destroy();
        } catch {
          // Ignore cleanup errors
        }
        playerRef.current = null;
      };
    }, [videoId]);

    // Toggle YouTube audio mute cleanly when switching between "Bản gốc" and "Kênh vocal / Instrumental / Solo"
    useEffect(() => {
      const p = playerRef.current;
      if (!p || !readyRef.current) return;
      try {
        if (muteYouTube) {
          if (!p.isMuted?.()) {
            mutePendingUntilRef.current = performance.now() + 450;
            p.mute();
          }
        } else {
          const targetVol = Math.round(Math.min(1, Math.max(0, extVolRef.current)) * 100);
          const curVol = Number(p.getVolume?.() ?? targetVol);
          if (Math.abs(curVol - targetVol) >= 1) {
            p.setVolume?.(targetVol);
          }
          lastYtVolRef.current = targetVol;
          lastYtMutedRef.current = extMutedRef.current;
          const currentlyMuted = Boolean(p.isMuted?.());
          if (extMutedRef.current && !currentlyMuted) {
            p.mute();
          } else if (!extMutedRef.current && currentlyMuted) {
            p.unMute();
          }
        }
      } catch {
        // Ignore transient states
      }
    }, [muteYouTube, ytReady]);

    // Sync playback rate when changed externally
    useEffect(() => {
      const p = playerRef.current;
      if (!p || !readyRef.current) return;
      try {
        if (p.getPlaybackRate && Math.abs(p.getPlaybackRate() - playbackRate) < 0.01) return;
        p.setPlaybackRate(playbackRate);
      } catch {
        // Ignore
      }
    }, [playbackRate, ytReady]);

    // 60fps time clock + passive Bản gốc volume sync + silent guard in stem mode (never spams mute/seek)
    useEffect(() => {
      if (!ytReady) return;
      let rafId = 0;

      const tick = () => {
        const p = playerRef.current;
        if (p && readyRef.current) {
          try {
            if (!muteRef.current) {
              if (ytPlayingRef.current) {
                const t = Number(p.getCurrentTime?.() ?? 0);
                if (Number.isFinite(t)) {
                  lastPolledYtTimeRef.current = t;
                  onTimeRef.current(t);
                }
              }
              const rawVol = p.getVolume?.();
              const rawMuted = p.isMuted?.();
              if (typeof rawVol === "number" && Number.isFinite(rawVol) && typeof rawMuted === "boolean") {
                if (Math.abs(rawVol - lastYtVolRef.current) >= 1 || rawMuted !== lastYtMutedRef.current) {
                  lastYtVolRef.current = rawVol;
                  lastYtMutedRef.current = rawMuted;
                  onVolRef.current?.(Math.min(1, Math.max(0, rawVol / 100)), rawMuted);
                }
              }
            } else {
              // In separated Vocal / Instrumental mode, ensure YouTube original audio stays silent if user clicks YouTube's speaker
              const rawMuted = p.isMuted?.();
              const nowMs = performance.now();
              if (rawMuted === false && nowMs > mutePendingUntilRef.current) {
                mutePendingUntilRef.current = nowMs + 450;
                p.mute();
              }
            }
          } catch {
            // Ignore transient YT player states
          }
        }
        rafId = requestAnimationFrame(tick);
      };
      rafId = requestAnimationFrame(tick);

      const scrubCheckId = window.setInterval(() => {
        const p = playerRef.current;
        if (!p || !readyRef.current) return;
        try {
          const st = p.getPlayerState?.();
          const t = Number(p.getCurrentTime?.() ?? 0);
          if (!Number.isFinite(t)) return;
          if (!muteRef.current) {
            if (st === 1 && !ytPlayingRef.current) {
              ytPlayingRef.current = true;
              lastPolledYtTimeRef.current = t;
              onPlayRef.current(true, t);
              return;
            }
            if ((st === 2 || st === 0) && ytPlayingRef.current) {
              ytPlayingRef.current = false;
              lastPolledYtTimeRef.current = t;
              onPlayRef.current(false, t);
              return;
            }
            if (!ytPlayingRef.current && Math.abs(t - lastPolledYtTimeRef.current) > 0.12) {
              lastPolledYtTimeRef.current = t;
              onTimeRef.current(t);
            }
          } else if (performance.now() > suppressExternalUntilRef.current) {
            // Detect manual scrubbing on YouTube's native red progress bar while listening to Vocal/Instrumental
            const jumpedFromLast = Math.abs(t - lastPolledYtTimeRef.current) > 0.85;
            const farFromExt = Math.abs(t - extPosRef.current) > 0.85;
            if (jumpedFromLast && farFromExt) {
              lastPolledYtTimeRef.current = t;
              onManualSeekRef.current?.(t);
            } else {
              lastPolledYtTimeRef.current = t;
            }
          }
        } catch {
          // Ignore
        }
      }, 200);

      return () => {
        cancelAnimationFrame(rafId);
        window.clearInterval(scrubCheckId);
      };
    }, [ytReady]);

    // Resync muted video frame cleanly when returning from another browser tab
    useEffect(() => {
      if (typeof document === "undefined") return;
      const onVisibilityChange = () => {
        lastVisibilityChangeMsRef.current = performance.now();
        if (document.visibilityState !== "visible" || isBackgroundRef.current) return;
        const p = playerRef.current;
        if (!p || !readyRef.current || !muteRef.current || !extPlayingRef.current) return;
        try {
          const nowMs = performance.now();
          suppressExternalUntilRef.current = nowMs + 750;
          if (!p.isMuted?.()) {
            mutePendingUntilRef.current = nowMs + 450;
            p.mute();
          }
          const curYt = Number(p.getCurrentTime?.() ?? 0);
          if (Math.abs(curYt - extPosRef.current) > 0.8) {
            p.seekTo(Math.max(0, extPosRef.current), true);
            lastSyncRef.current = nowMs;
          }
          if (p.getPlayerState?.() !== 1) {
            p.playVideo();
          }
        } catch {
          // Ignore
        }
      };
      document.addEventListener("visibilitychange", onVisibilityChange);
      return () => document.removeEventListener("visibilitychange", onVisibilityChange);
    }, []);

    // Keep muted YouTube video play/pause locked to local <audio> without resetting YouTube's 3s auto-hide timer
    useEffect(() => {
      if (!muteYouTube || !ytReady || isBackground) return;
      if (typeof document !== "undefined" && document.hidden) return;
      const p = playerRef.current;
      if (!p || !readyRef.current) return;
      const nowMs = performance.now();
      if (nowMs < suppressExternalUntilRef.current) return;
      try {
        if (!p.isMuted?.() && nowMs > mutePendingUntilRef.current) {
          mutePendingUntilRef.current = nowMs + 450;
          p.mute();
        }
        const st = p.getPlayerState?.();
        if (externalPlaying && st !== 1) {
          suppressExternalUntilRef.current = nowMs + 550;
          const curYt = Number(p.getCurrentTime?.() ?? 0);
          if (Math.abs(curYt - extPosRef.current) > 0.8) {
            p.seekTo(Math.max(0, extPosRef.current), true);
            lastSyncRef.current = nowMs;
          }
          p.playVideo();
        } else if (!externalPlaying && st === 1) {
          suppressExternalUntilRef.current = nowMs + 450;
          p.pauseVideo();
        }
      } catch {
        // Ignore
      }
    }, [muteYouTube, externalPlaying, ytReady, isBackground]);

    // Sync muted YouTube timeline ONLY on explicit user seeks or genuine large desync (>2.5s)
    // so YouTube's native player HUD always auto-hides smoothly after 3s when idle
    useEffect(() => {
      const prevPos = prevExternalPosRef.current;
      prevExternalPosRef.current = externalPosition;
      if (!muteYouTube || !ytReady || isBackground) return;
      if (typeof document !== "undefined" && document.hidden) return;
      const p = playerRef.current;
      if (!p || !readyRef.current) return;
      const nowMs = performance.now();
      if (nowMs < suppressExternalUntilRef.current) return;

      const isExplicitJump = Math.abs(externalPosition - prevPos) > 0.65;
      if (!isExplicitJump) {
        if (!externalPlaying || nowMs - lastSyncRef.current < 6000) return;
      } else if (nowMs - lastSyncRef.current < 180) {
        return;
      }

      try {
        const curYt = Number(p.getCurrentTime?.() ?? 0);
        const drift = Math.abs(curYt - externalPosition);
        const threshold = isExplicitJump ? 0.45 : 2.5;
        if (drift > threshold) {
          suppressExternalUntilRef.current = nowMs + 600;
          lastPolledYtTimeRef.current = externalPosition;
          p.seekTo(externalPosition, true);
          lastSyncRef.current = nowMs;
        }
      } catch {
        // Ignore
      }
    }, [muteYouTube, externalPosition, externalPlaying, ytReady, isBackground]);

    const leadLine = activeSegments[0] ?? null;

    return (
      <div className="ss-yt-stage">
        {/* 16:9 Native YouTube Video Frame (100% native YouTube controls) */}
        <div className="ss-yt-video-box">
          {ytError ? (
            <iframe
              className="ss-yt-iframe-fallback"
              src={`https://www.youtube.com/embed/${videoId}?rel=0&modestbranding=1`}
              title={title}
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          ) : (
            <div ref={hostRef} className="ss-yt-iframe-host" />
          )}
        </div>

        {/* Live Subtitle / Karaoke Strip */}
        <div className="ss-yt-subtitle-bar" aria-live="polite">
          {leadLine ? (
            <div className="ss-yt-subtitle-stack">
              {activeSegments.slice(0, 2).map(seg => {
                const words = karaokeWordsBySegId.get(seg.id) ?? seg.words ?? [];
                const hasKaraoke = showWordHighlight && words.length > 0;
                return (
                  <div key={seg.id} className="ss-yt-subtitle-line">
                    {showChords && seg.chord && <span className="ss-yt-sub-chord">{seg.chord}</span>}
                    <p className="ss-yt-sub-text">
                      {hasKaraoke
                        ? words.map((w, wi) => {
                            const nextWordStart = wi + 1 < words.length ? words[wi + 1].start : null;
                            const ws = getWordPlaybackState(w, syncTime, nextWordStart);
                            return (
                              <span
                                key={wi}
                                className={`ss-yt-word ${ws === "active" ? "active" : ws === "sung" ? "passed" : ""}`}
                              >
                                {w.word}
                              </span>
                            );
                          })
                        : seg.text}
                    </p>
                  </div>
                );
              })}
            </div>
          ) : (
            <p className="ss-yt-sub-idle">♪ Nhấn phát trên video YouTube hoặc chọn một câu bên phải để nghe đồng bộ</p>
          )}
        </div>
      </div>
    );
  },
);
