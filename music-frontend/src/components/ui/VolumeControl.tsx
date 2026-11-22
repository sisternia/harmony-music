import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { createPortal } from 'react-dom';
import { colors } from '@/theme/colors';
import Icon from './Icon';
import './volume-control.css';

type Props = {
  value: number;
  onChange: (value: number) => void;
  placement?: 'top' | 'bottom';
  label?: string;
  sliderLabel?: string;
  className?: string;
  buttonClassName?: string;
  onOpen?: () => void;
};

export default function VolumeControl({
  value,
  onChange,
  placement = 'bottom',
  label = 'Điều chỉnh âm lượng',
  sliderLabel = 'Âm lượng',
  className = '',
  buttonClassName = '',
  onOpen,
}: Props) {
  const [open, setOpen] = useState(false);
  const trigger = useRef<HTMLButtonElement>(null);
  const popup = useRef<HTMLDivElement>(null);
  const slider = useRef<HTMLInputElement>(null);
  const popupId = useId();
  const level = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  const percent = Math.round(level * 100);

  // Position outside the toolbar's stacking/overflow context so the Piano
  // keyboard cannot cover the slider, including on small touch screens.
  useLayoutEffect(() => {
    if (!open) return;
    const position = () => {
      const button = trigger.current;
      const panel = popup.current;
      if (!button || !panel) return;
      const rect = button.getBoundingClientRect();
      const gap = 8;
      const panelWidth = panel.offsetWidth;
      const panelHeight = panel.offsetHeight;
      const left = Math.max(gap, Math.min(rect.left, window.innerWidth - panelWidth - gap));
      const above = rect.top - panelHeight - gap;
      const below = rect.bottom + gap;
      const top = placement === 'top'
        ? above >= gap ? above : below
        : below + panelHeight <= window.innerHeight - gap ? below : above;
      panel.style.left = `${left}px`;
      panel.style.top = `${Math.max(gap, Math.min(top, window.innerHeight - panelHeight - gap))}px`;
    };
    position();
    slider.current?.focus({ preventScroll: true });
    window.addEventListener('resize', position);
    window.addEventListener('scroll', position, true);
    return () => {
      window.removeEventListener('resize', position);
      window.removeEventListener('scroll', position, true);
    };
  }, [open, placement]);

  useEffect(() => {
    if (!open) return;
    const isInside = (target: EventTarget | null) => target instanceof Node
      && (trigger.current?.contains(target) || popup.current?.contains(target));
    const closeOutside = (event: PointerEvent) => {
      if (!isInside(event.target)) setOpen(false);
    };
    const closeOnFocusOut = (event: FocusEvent) => {
      if (!isInside(event.target)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      trigger.current?.focus({ preventScroll: true });
    };
    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('focusin', closeOnFocusOut);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOutside);
      document.removeEventListener('focusin', closeOnFocusOut);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [open]);

  return (
    <div className={`ui-volume-control ${className}`}>
      <button
        ref={trigger}
        type="button"
        className={`ui-volume-trigger ${buttonClassName}`}
        aria-label={label}
        aria-expanded={open}
        aria-controls={open ? popupId : undefined}
        aria-haspopup="dialog"
        title={`${sliderLabel}: ${percent}%`}
        onClick={() => {
          if (!open) onOpen?.();
          setOpen(current => !current);
        }}
      >
        <Icon
          name={level === 0 ? 'volumeMute' : level < 0.34 ? 'volumeLow' : level < 0.67 ? 'volumeMedium' : 'volumeHigh'}
          color={colors.text}
          size={21}
        />
      </button>
      {open && createPortal(
        <div ref={popup} className="ui-volume-popover" id={popupId} role="dialog" aria-label={sliderLabel}>
          <div className="ui-volume-caption">
            <span>{sliderLabel}</span>
            <output htmlFor={`${popupId}-slider`}>{percent}%</output>
          </div>
          <input
            ref={slider}
            id={`${popupId}-slider`}
            className="ui-volume-slider"
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={level}
            style={{ '--fill': `${percent}%` } as CSSProperties}
            aria-label={sliderLabel}
            aria-valuetext={`${percent}%`}
            onChange={event => {
              const next = Number(event.currentTarget.value);
              if (Number.isFinite(next)) onChange(Math.max(0, Math.min(1, next)));
            }}
          />
        </div>,
        document.body,
      )}
    </div>
  );
}
