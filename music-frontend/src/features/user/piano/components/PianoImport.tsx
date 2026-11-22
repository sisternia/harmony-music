import { useEffect, useRef, useState } from 'react';
import { pianoRequest, type PianoSong, type TranscriptionJob } from '../utils/transcription';
import { parseMidi } from '../utils/parseMidi';
import { sourceProjectMidi, takeProjectMidi } from '../utils/projectMidiSong';
import { api, type ProjectDetail, type ProjectSummary } from '../../../studio/api';

type Props = { open: boolean; embedded?: boolean; onClose: () => void; onImported: (song: PianoSong) => void };

export default function PianoImport({ open, embedded = false, onClose, onImported }: Props) {
  const [mode, setMode] = useState<'audio' | 'midi'>('audio');
  const [file, setFile] = useState<File | null>(null);
  const [youtube, setYoutube] = useState('');
  const [job, setJob] = useState<TranscriptionJob | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState('');
  const [midiBusy, setMidiBusy] = useState(false);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedProject, setSelectedProject] = useState('');
  const [projectDetail, setProjectDetail] = useState<ProjectDetail | null>(null);
  const [projectLoading, setProjectLoading] = useState(false);
  const [midiChoice, setMidiChoice] = useState('');
  const projectRequest = useRef(0);
  const fileInput = useRef<HTMLInputElement>(null);
  const imported = useRef(onImported);
  const close = useRef(onClose);
  useEffect(() => { imported.current = onImported; close.current = onClose; }, [onImported, onClose]);
  const busy = submitting || !!job && !['completed', 'failed', 'cancelled'].includes(job.status);

  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    void api.projects().then(items => {
      if (!cancelled) setProjects(items.filter(item => item.has_source || item.takes > 0));
    }).catch(reason => {
      if (!cancelled) setError(reason instanceof Error ? reason.message : 'Không tải được Projects.');
    });
    return () => { cancelled = true; };
  }, [open]);

  const selectProject = (projectId: string) => {
    const request = ++projectRequest.current;
    setSelectedProject(projectId);
    setProjectDetail(null);
    setMidiChoice('');
    setError('');
    if (!projectId) { setProjectLoading(false); return; }
    setProjectLoading(true);
    void api.project(projectId).then(detail => {
      if (request !== projectRequest.current) return;
      setProjectDetail(detail);
      setMidiChoice(detail.source ? 'source' : detail.takes.find(take => take.state === 'done')?.id || '');
    }).catch(reason => {
      if (request === projectRequest.current) setError(reason instanceof Error ? reason.message : 'Không đọc được Project.');
    }).finally(() => { if (request === projectRequest.current) setProjectLoading(false); });
  };

  const openProjectMidi = async () => {
    if (!projectDetail || !midiChoice || midiBusy || busy) return;
    setMidiBusy(true);
    setError('');
    try {
      let song: PianoSong;
      if (midiChoice === 'source') song = sourceProjectMidi(projectDetail.title, await api.sourceView(projectDetail.id));
      else {
        const take = projectDetail.takes.find(item => item.id === midiChoice);
        if (!take || take.state !== 'done') throw new Error('Take này chưa có MIDI khả dụng.');
        const [cover, source, musicXml] = await Promise.all([
          api.takeView(projectDetail.id, take.id),
          projectDetail.source ? api.sourceView(projectDetail.id).catch(() => null) : Promise.resolve(null),
          take.score?.state === 'done' ? api.scoreXml(projectDetail.id, take.id).catch(() => undefined) : Promise.resolve(undefined),
        ]);
        song = takeProjectMidi(`${projectDetail.title} · ${take.name || `Take ${take.number}`}`, cover, source, musicXml);
      }
      imported.current(song);
      close.current();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Không mở được MIDI từ Project.');
    } finally { setMidiBusy(false); }
  };

  useEffect(() => {
    if (!job?.id || ['completed', 'failed', 'cancelled'].includes(job.status)) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const state = await pianoRequest<TranscriptionJob>(`/jobs/${job.id}`, { signal: controller.signal });
        if (state.status === 'completed') {
          const result = await pianoRequest<Omit<PianoSong, 'id'>>(`/jobs/${job.id}/result`, { signal: controller.signal });
          imported.current({ ...result, id: job.id });
          setJob(state);
          close.current();
        } else if (['failed', 'cancelled'].includes(state.status)) {
          setJob(state);
          if (state.status === 'failed') setError(state.message);
        } else {
          setJob(state);
          timer = setTimeout(poll, 1500);
        }
      } catch (reason) {
        if (controller.signal.aborted) return;
        setError(reason instanceof Error ? reason.message : 'Không đọc được tiến trình.');
        // A temporary lost connection does not abandon an expensive running job.
        timer = setTimeout(poll, 4000);
      }
    };
    void poll();
    return () => { controller.abort(); clearTimeout(timer); };
  }, [job?.id, job?.status]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (busy) return;
    setError('');
    const source = youtube.trim() ? 'youtube' : 'file';
    if (source === 'file' && !file) { setError('Bạn chưa chọn file hoặc nhập link YouTube.'); return; }
    if (source === 'youtube' && !youtube.trim()) { setError('Bạn chưa nhập link YouTube.'); return; }
    setSubmitting(true);
    try {
      const data = new FormData();
      if (source === 'file' && file) data.append('file', file);
      else data.append('youtube', youtube.trim());
      setJob(await pianoRequest<TranscriptionJob>('/jobs', { method: 'POST', body: data }));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Không gửi được bản nhạc.');
    } finally { setSubmitting(false); }
  };

  useEffect(() => {
    if (!open) return;
    const onEscape = (event: KeyboardEvent) => { if (event.key === 'Escape') close.current(); };
    window.addEventListener('keydown', onEscape);
    return () => window.removeEventListener('keydown', onEscape);
  }, [open]);

  if (!open) return null;
  const chooseFile = (next: File | null) => {
    if (next && mode === 'midi' && !/\.midi?$/i.test(next.name)) {
      setError('Hãy chọn tệp .mid hoặc .midi.');
      return;
    }
    if (next && /\.midi?$/i.test(next.name)) {
      if (mode !== 'midi') { setError('Hãy chọn tab MIDI để nhập tệp này.'); return; }
      setMidiBusy(true);
      setError('');
      void parseMidi(next).then(song => { imported.current(song); close.current(); }).catch(reason => setError(reason instanceof Error ? reason.message : 'Không đọc được MIDI.')).finally(() => setMidiBusy(false));
      return;
    }
    setFile(next);
    if (next) { setYoutube(''); setError(''); }
  };
  return (
    <div className={embedded ? 'transcription-entry' : 'piano-import-backdrop'} onClick={event => { if (!embedded && event.target === event.currentTarget) onClose(); }}>
      <div className={embedded ? 'piano-import-shell' : 'piano-import-modal'}>
      {!embedded && <button className="piano-import-dismiss" type="button" aria-label="Đóng" onClick={onClose}>×</button>}
      <section className="piano-import" role={embedded ? undefined : 'dialog'} aria-modal={embedded ? undefined : true} aria-labelledby="piano-import-title">
        <span className="piano-import-eyebrow">HARMONY AURALIS · PIANO AI</span>
        <h2 id="piano-import-title">{mode === 'midi' ? 'Mở MIDI trong Piano' : 'Biến bản nhạc thành piano'}</h2>
        <p>{mode === 'midi' ? 'Chọn tệp MIDI từ máy hoặc một bản MIDI đã có trong Projects để nghe và tập theo.' : 'Chọn một bản thu hoặc video YouTube. AI sẽ chuyển thành bản nhạc piano để bạn nghe và tập theo.'}</p>
        <div className="piano-import-tabs" role="tablist" aria-label="Nguồn bản nhạc">
          <button id="piano-audio-tab" type="button" role="tab" aria-controls="piano-audio-panel" aria-selected={mode === 'audio'} onClick={() => { setMode('audio'); setError(''); }}>Âm thanh / YouTube</button>
          <button id="piano-midi-tab" type="button" role="tab" aria-controls="piano-midi-panel" aria-selected={mode === 'midi'} onClick={() => { setMode('midi'); setError(''); }}>MIDI</button>
        </div>
        {mode === 'midi' && <div id="piano-midi-panel" role="tabpanel" aria-labelledby="piano-midi-tab" className="piano-midi-panel">
        <label className="piano-upload-zone" onDragOver={event => event.preventDefault()} onDrop={event => {
          event.preventDefault();
          if (!midiBusy && !busy) chooseFile(event.dataTransfer.files[0] ?? null);
        }}>
          <svg className="upload-cloud" width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M14 34H10a8 8 0 0 1-1-16 14 14 0 0 1 27-2 9 9 0 0 1 2 18h-5M24 39V20m-7 7 7-7 7 7" /></svg>
          <strong>Nhập tệp MIDI từ máy</strong>
          <span className="piano-drop-hint">Hoặc kéo và thả tệp vào đây</span>
          <span>Định dạng .mid hoặc .midi</span>
          <input type="file" accept=".mid,.midi" aria-label="Chọn tệp MIDI từ máy" disabled={midiBusy || busy} onChange={event => { chooseFile(event.target.files?.[0] ?? null); event.target.value = ''; }} />
        </label>
        <div className="piano-midi-project">
          <div>
            <select aria-label="Chọn Project có MIDI" disabled={midiBusy || busy} onChange={event => selectProject(event.target.value)} value={selectedProject}>
              <option value="">{projects.length ? 'Chọn Project' : 'Chưa có Project chứa MIDI'}</option>
              {projects.map(project => <option key={project.id} value={project.id}>{project.title}</option>)}
            </select>
          </div>
          {projectLoading && <span role="status">Đang đọc Project…</span>}
          {projectDetail && <label>Chọn bản MIDI
            <select value={midiChoice} onChange={event => setMidiChoice(event.target.value)} disabled={midiBusy}>
              {!projectDetail.source && <option value="">Chọn Take</option>}
              {projectDetail.source && <option value="source">Source MIDI</option>}
              {projectDetail.takes.filter(take => take.state === 'done').map(take => <option key={take.id} value={take.id}>{take.name || `Take ${take.number}`}</option>)}
            </select>
          </label>}
          {projectDetail && <button type="button" className="piano-primary" disabled={!midiChoice || midiBusy || busy} onClick={() => void openProjectMidi()}>Mở MIDI từ Project</button>}
        </div>
        {midiBusy && <p role="status">Đang đọc MIDI…</p>}
        {error && <p className="piano-import-error" role="alert">{error}</p>}
        </div>}
        {mode === 'audio' && <form id="piano-audio-panel" role="tabpanel" aria-labelledby="piano-audio-tab" onSubmit={submit}>
            <label className="piano-upload-zone" onDragOver={event => event.preventDefault()} onDrop={event => {
              event.preventDefault();
              if (!busy) chooseFile(event.dataTransfer.files[0] ?? null);
            }}>
              <svg className="upload-cloud" width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M14 34H10a8 8 0 0 1-1-16 14 14 0 0 1 27-2 9 9 0 0 1 2 18h-5M24 39V20m-7 7 7-7 7 7" /></svg>
              <strong>{file ? file.name : 'Nhập file âm thanh hoặc bản nhạc'}</strong>
              <span className="piano-drop-hint">Hoặc kéo và thả file vào đây</span>
              <span>Âm thanh hoặc video · tối đa 100 MB</span>
              <input ref={fileInput} className="piano-file-input" aria-label="Chọn file âm thanh hoặc video" type="file" accept=".mp3,.wav,.flac,.m4a,.aac,.ogg,.mp4,.webm" disabled={busy} onChange={event => chooseFile(event.target.files?.[0] ?? null)} />
            </label>
            <label className="piano-youtube-label piano-youtube-inline">
              <svg className="piano-youtube-icon" viewBox="0 0 28 20" aria-hidden="true"><rect width="28" height="20" rx="5" fill="currentColor" /><path d="m11 5 8 5-8 5z" fill="var(--on-accent)" /></svg>
              <input type="url" aria-label="Link YouTube" placeholder="Dán link YouTube tại đây" value={youtube} onChange={event => {
                setYoutube(event.target.value);
                if (event.target.value.trim()) { setFile(null); if (fileInput.current) fileInput.current.value = ''; }
              }} disabled={busy} />
            </label>
          {job && <div className="piano-job-status" role="status"><progress max={100} value={job.progress} /><span>{job.message}</span></div>}
          {error && <p className="piano-import-error" role="alert">{error}</p>}
          <p className="piano-import-note">Tối đa 20 phút mỗi bài. Chuyển nhạc có thể mất vài phút; phổ AI có thể cần chỉnh lại một số nốt.</p>
          <div className="piano-import-actions">
            {busy && job ? <button className="piano-import-cancel" type="button" onClick={async () => {
              try { await pianoRequest(`/jobs/${job.id}/cancel`, { method: 'POST' }); }
              catch (reason) { setError(reason instanceof Error ? reason.message : 'Không hủy được tác vụ.'); }
            }}>Hủy xử lý</button> : null}
            <button type="submit" className="piano-primary" disabled={busy}>{busy ? 'Đang chuyển nhạc…' : 'Chuyển thành piano'}</button>
          </div>
        </form>}
      </section>
      </div>
    </div>
  );
}
