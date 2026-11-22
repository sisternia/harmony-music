import type { CSSProperties, MouseEvent } from "react";
import { createRoot } from "react-dom/client";
import { colors } from "@/theme/colors";
import Icon from "../ui/Icon";
import "./navbar-feedback.css";

const activeFeedback = new Set<() => void>();
const notes = [
  { x: -37, y: -27, rotation: -24, size: 15 },
  { x: -10, y: -45, rotation: 14, size: 17 },
  { x: 29, y: -34, rotation: 27, size: 14 },
  { x: 42, y: 7, rotation: -16, size: 16 },
  { x: -28, y: 25, rotation: 18, size: 13 },
];

// Own a short-lived body layer so feedback can finish after the route unmounts.
// Capture click once for mouse, touch, Enter and Space without delaying navigation.
export function playNavbarFeedback(event: MouseEvent<HTMLElement>) {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  const target = event.target instanceof Element ? event.target.closest("button") : null;
  if (!(target instanceof HTMLButtonElement) || target.disabled || !event.currentTarget.contains(target)) return;
  const rect = target.getBoundingClientRect();
  if (!rect.width || !rect.height) return;

  const x = event.detail === 0 ? rect.left + rect.width / 2 : event.clientX;
  const y = event.detail === 0 ? rect.top + rect.height / 2 : event.clientY;
  const localX = x - rect.left;
  const localY = y - rect.top;
  const diameter = Math.hypot(Math.max(localX, rect.width - localX), Math.max(localY, rect.height - localY)) * 2;

  // Keep rapid repeated clicks inexpensive, including across route changes.
  if (activeFeedback.size >= 3) activeFeedback.values().next().value?.();
  const host = document.createElement("div");
  host.className = "navbar-click-feedback";
  host.setAttribute("aria-hidden", "true");
  document.body.appendChild(host);
  const root = createRoot(host);
  const dispose = () => {
    window.clearTimeout(timeout);
    root.unmount();
    host.remove();
    activeFeedback.delete(dispose);
  };
  const timeout = window.setTimeout(dispose, 900);
  activeFeedback.add(dispose);
  host.addEventListener("animationend", animation => {
    if (animation.target instanceof HTMLElement && animation.target.hasAttribute("data-feedback-last")) dispose();
  });

  root.render(<>
    <div className="navbar-click-feedback__clip" style={{
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height,
      borderRadius: getComputedStyle(target).borderRadius,
    }}>
      <span className="navbar-click-feedback__ripple" style={{ left: localX, top: localY, width: diameter, height: diameter }} />
    </div>
    {notes.map((note, index) => <span
      key={index}
      className="navbar-click-feedback__note"
      data-feedback-last={index === notes.length - 1 ? "" : undefined}
      style={{
        left: x - note.size / 2,
        top: y - note.size / 2,
        "--note-x": `${note.x}px`,
        "--note-y": `${note.y}px`,
        "--note-rotation": `${note.rotation}deg`,
        animationDelay: `${index * 18}ms`,
      } as CSSProperties}
    >
      <Icon name={index % 2 ? "musicNotes" : "musicNote"} color={index % 2 ? colors.primary : colors.secondary} size={note.size} />
    </span>)}
  </>);
}
