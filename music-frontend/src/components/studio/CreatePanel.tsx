// Clean, minimalist Piano Cover generator panel (simple presets up front, advanced ML knobs tucked away)
import { useState } from "react";
import type { Config, ContinueFrom, Params, Take } from "../../features/studio/api";
import { useT } from "../../i18n";
import { CHANNEL_CFG, GROUPS, LENGTHS, formatTime, formatValue, type SliderDef } from "../../features/studio/utils/params";
import { Icon } from "../ui/StudioIcon";
import { StyleSection } from "./StyleSection";

export type Draft = {
  params: Params;
  model: string | null;
  count: number;
  seedLocked: boolean;
  seed: number | null;
};

type StylePreset = {
  id: "faithful" | "balanced" | "expressive";
  label: string;
  desc: string;
  patch: Partial<Params>;
};

const COVER_PRESETS: StylePreset[] = [
  {
    id: "faithful",
    label: "Sát bản gốc",
    desc: "Bám chặt giai điệu & nhịp gốc",
    patch: { source_cfg: 2.5, onset_bias: 5.0, dynamics: 1.0, density: 1.0, fill: 0.0, span: 0.0 },
  },
  {
    id: "balanced",
    label: "Cân bằng",
    desc: "Tự nhiên, mượt mà, dễ nghe",
    patch: { source_cfg: 2.0, onset_bias: 4.0, dynamics: 1.0, density: 1.0, fill: 0.0, span: 0.0 },
  },
  {
    id: "expressive",
    label: "Biểu cảm",
    desc: "Phối dày, sắc thái mạnh & giàu âm",
    patch: { source_cfg: 1.65, onset_bias: 3.0, dynamics: 1.4, density: 1.35, fill: 0.8, span: 0.5 },
  },
];

type Props = {
  config: Config;
  draft: Draft;
  onChange: (draft: Draft) => void;
  continueFrom: ContinueFrom | null;
  continueTake: Take | null;
  onClearContinue: () => void;
  onGenerate: () => void;
  onClose?: () => void;
  generating: boolean;
  disabled: string | null;
};

export function CreatePanel(props: Props) {
  const { config, draft, onChange, disabled } = props;
  const t = useT();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const defaults = config.defaults;
  const params = draft.params;
  const set = (patch: Partial<Params>) => onChange({ ...draft, params: { ...params, ...patch } });
  const model = config.models.find((m) => m.id === draft.model) ?? config.models[0];
  const maxChannel = model?.num_channels ? model.num_channels - 1 : undefined;

  const activePreset =
    COVER_PRESETS.find((pr) =>
      Object.entries(pr.patch).every(([k, v]) => Math.abs(Number(params[k as keyof Params]) - Number(v)) < 0.05),
    )?.id ?? null;

  const slider = (def: SliderDef, disabledSlider = false) => {
    const value = params[def.key];
    const isDefault = Math.abs(value - defaults[def.key]) < 1e-9;
    const text = t.params[def.key];
    return (
      <div className={`slider ${disabledSlider ? "disabled" : ""}`} key={def.key}>
        <div className="slider-head">
          <label htmlFor={`p-${def.key}`}>{text.label}</label>
          <span className={`slider-value mono ${isDefault ? "" : "changed"}`}>{formatValue(value, def.step)}</span>
        </div>
        <input
          id={`p-${def.key}`}
          type="range"
          min={def.min}
          max={def.max}
          step={def.step}
          value={value}
          disabled={disabledSlider}
          onChange={(e) => set({ [def.key]: Number(e.target.value) } as Partial<Params>)}
          onDoubleClick={() => set({ [def.key]: defaults[def.key] } as Partial<Params>)}
          style={{ "--fill": `${((value - def.min) / (def.max - def.min)) * 100}%` } as React.CSSProperties}
        />
      </div>
    );
  };

  return (
    <aside className="create create-clean">
      <div className="create-scroll">
        <div className="panel-title">
          <span>
            <Icon name="sparkle" size={16} /> Tạo bản phối Piano
          </span>
          {props.onClose && (
            <button type="button" className="icon-btn" onClick={props.onClose} aria-label="Đóng bảng tạo phối" title="Đóng">
              <Icon name="close" size={16} />
            </button>
          )}
        </div>

        {props.continueFrom && (
          <div className="continue-card">
            <Icon name="scissors" />
            <div>
              <b>
                Viết tiếp từ {props.continueTake ? takeLabel(props.continueTake, t) : props.continueFrom.take} ({formatTime(props.continueFrom.seconds)})
              </b>
            </div>
            <button className="icon-btn" onClick={props.onClearContinue} title="Hủy">
              <Icon name="close" size={15} />
            </button>
          </div>
        )}

        <div className="field">
          <label>Phong cách phối</label>
          <div className="create-preset-list">
            {COVER_PRESETS.map((pr) => (
              <button
                key={pr.id}
                type="button"
                className={`create-preset-btn ${activePreset === pr.id ? "on" : ""}`}
                onClick={() => set(pr.patch)}
              >
                <strong>{pr.label}</strong>
                <small>{pr.desc}</small>
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label>Độ dài bản tạo</label>
          <div className="segmented">
            {LENGTHS.filter((v) => v !== 90).map((value) => (
              <button
                key={String(value)}
                type="button"
                className={params.seconds === value ? "on" : ""}
                onClick={() => set({ seconds: value })}
              >
                {value == null ? "Cả bài" : `${value}s`}
              </button>
            ))}
          </div>
        </div>

        <div className="field">
          <label>Số phiên bản tạo cùng lúc</label>
          <div className="segmented small">
            {[1, 2, 3].map((n) => (
              <button
                key={n}
                type="button"
                className={draft.count === n ? "on" : ""}
                onClick={() => onChange({ ...draft, count: n })}
              >
                {n} bản
              </button>
            ))}
          </div>
        </div>

        <div className="create-advanced-box">
          <button
            type="button"
            className="create-advanced-toggle"
            aria-expanded={advancedOpen}
            onClick={() => setAdvancedOpen((v) => !v)}
          >
            <span>Tùy chỉnh nâng cao (AI)</span>
            <span>{advancedOpen ? "▴" : "▾"}</span>
          </button>

          {advancedOpen && (
            <div className="create-advanced-body">
              <div className="field">
                <label htmlFor="model">Mô hình AI</label>
                <select id="model" value={model?.id ?? ""} onChange={(e) => onChange({ ...draft, model: e.target.value })}>
                  {config.models.map((m) => (
                    <option value={m.id} key={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
              </div>

              {GROUPS.map((group) => (
                <section className="group" key={group.id}>
                  <div className="group-title">
                    {t.groups[group.id].title}
                    {group.id === "arrangement" && (
                      <label className="switch">
                        <input
                          type="checkbox"
                          checked={params.arrangement}
                          onChange={(e) => set({ arrangement: e.target.checked })}
                        />
                        <span />
                      </label>
                    )}
                  </div>
                  {group.sliders.map((def) => slider(def, group.id === "arrangement" && !params.arrangement))}
                </section>
              ))}

              <section className="group">
                <div className="field inline">
                  <label htmlFor="channel">Performer #</label>
                  <input
                    id="channel"
                    type="number"
                    min={0}
                    max={maxChannel}
                    value={params.channel}
                    disabled={!!params.style}
                    onChange={(e) => set({ channel: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
                  />
                </div>
                <StyleSection model={model} selected={params.style} onSelect={(style) => set({ style })} />
                {slider(CHANNEL_CFG, !params.channel && !params.style)}

                <label className="seed">
                  <input
                    type="checkbox"
                    checked={draft.seedLocked}
                    onChange={(e) =>
                      onChange({
                        ...draft,
                        seedLocked: e.target.checked,
                        seed: draft.seed ?? Math.floor(Math.random() * 2 ** 31),
                      })
                    }
                  />
                  Khóa Seed
                  <input
                    className="mono"
                    type="number"
                    value={draft.seed ?? ""}
                    placeholder="Ngẫu nhiên"
                    disabled={!draft.seedLocked}
                    onChange={(e) => onChange({ ...draft, seed: e.target.value === "" ? null : Number(e.target.value) })}
                  />
                </label>
              </section>

              <button
                type="button"
                className="link"
                onClick={() => onChange({ ...draft, params: { ...defaults } })}
              >
                Khôi phục thông số mặc định
              </button>
            </div>
          )}
        </div>
      </div>

      <div className="create-footer">
        <button className="generate" disabled={!!disabled || props.generating} onClick={props.onGenerate}>
          <Icon name="sparkle" size={17} />
          {props.continueFrom ? "Viết tiếp từ vị trí này" : "Tạo bản phối Piano"}
        </button>
        {disabled && <div className="footer-note">{disabled}</div>}
      </div>
    </aside>
  );
}

export function takeLabel(take: Take, t: ReturnType<typeof useT>): string {
  return take.name || t.take(take.number);
}
