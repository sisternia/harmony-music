import { useEffect, useRef, useState, type FormEvent } from 'react';
import '../../../styles/theme.css';
import { AuthApiError, authApi, saveAuthSession, type AuthUserSession } from '../api';
import { type ContactKind } from '../components/fields';
import { LoginScreen } from './LoginScreen';
import { RegisterScreen } from './RegisterScreen';
import { RecoverScreen } from './RecoverScreen';
import { VerifyScreen } from './VerifyScreen';
import { ResetScreen } from './ResetScreen';
import '../auth.css';

type AuthStep = 'login' | 'register' | 'recover' | 'verify' | 'reset';
type Props = {
  open: boolean;
  onClose: () => void;
  logo: string;
  initialStep?: 'login' | 'register';
  onAuthenticated?: (session: AuthUserSession) => void;
};

const headings: Record<AuthStep, { title: string; subtitle: string }> = {
  login: { title: 'Chào mừng trở lại', subtitle: 'Đăng nhập để tiếp tục hành trình âm nhạc của bạn.' },
  register: { title: 'Tạo tài khoản', subtitle: 'Bắt đầu lưu lại những bản nhạc của riêng bạn.' },
  recover: { title: 'Quên mật khẩu?', subtitle: 'Nhập email tài khoản để nhận mã xác thực 6 chữ số.' },
  verify: { title: 'Xác thực tài khoản', subtitle: 'Nhập mã xác thực gồm 6 chữ số đã gửi về email của bạn.' },
  reset: { title: 'Đổi mật khẩu', subtitle: 'Chọn mật khẩu mới mạnh mẽ cho tài khoản của bạn.' },
};

function validContact(value: string, kind: ContactKind) {
  return kind === 'email'
    ? /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value.trim())
    : /^\+?[\d\s().-]{8,}$/.test(value.trim()) && value.replace(/\D/g, '').length >= 8;
}

function validateStrongPassword(pw: string): string | null {
  if (pw.length < 8) return 'Mật khẩu cần ít nhất 8 ký tự.';
  if (!/[A-Z]/.test(pw)) return 'Mật khẩu cần có ít nhất 1 chữ in hoa (A–Z).';
  if (!/[a-z]/.test(pw)) return 'Mật khẩu cần có ít nhất 1 chữ thường (a–z).';
  if (!/[0-9]/.test(pw)) return 'Mật khẩu cần có ít nhất 1 chữ số (0–9).';
  if (!/[!@#$%^&*()_+=[\]{};:'"<,.>/?|\\]/.test(pw)) {
    return 'Mật khẩu cần có ít nhất 1 ký tự đặc biệt (!@#$%^&*...).';
  }
  return null;
}

export default function AuthDialog({ open, onClose, logo, initialStep = 'login', onAuthenticated }: Props) {
  const [step, setStep] = useState<AuthStep>(initialStep);
  const [source, setSource] = useState<'register' | 'recover'>('register');
  const [username, setUsername] = useState('');
  const [identity, setIdentity] = useState('');
  const [contact, setContact] = useState('');
  const [contactKind, setContactKind] = useState<ContactKind>('email');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [code, setCode] = useState<string[]>(Array(6).fill(''));
  const [verifiedCode, setVerifiedCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const firstField = useRef<HTMLInputElement>(null);
  const opener = useRef<HTMLElement | null>(null);
  const closeRef = useRef(onClose);

  useEffect(() => { closeRef.current = onClose; }, [onClose]);

  const changeStep = (next: AuthStep, keepNotice = '') => {
    setStep(next);
    setError('');
    setNotice(keepNotice);
    setPassword('');
    setConfirmation('');
    if (next !== 'reset') {
      setCode(Array(6).fill(''));
    }
  };

  const changeContactKind = (next: ContactKind) => {
    setContactKind(next);
    setContact('');
    setError('');
  };

  useEffect(() => {
    if (!open) return;
    opener.current = document.activeElement as HTMLElement;
    const escape = (event: globalThis.KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) closeRef.current();
    };
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('keydown', escape);
      opener.current?.focus();
    };
  }, [open, busy]);

  useEffect(() => {
    if (!open) return;
    const focus = requestAnimationFrame(() => firstField.current?.focus());
    return () => cancelAnimationFrame(focus);
  }, [open, step]);

  const handleResendCode = async () => {
    if (busy || !contact.trim()) return;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const res = await authApi.sendVerifyCode(contact);
      const devHint = res.verify_code ? ` (Mã dev: ${res.verify_code})` : '';
      setNotice(`Đã gửi lại mã xác thực 6 chữ số đến ${contact.trim()}.${devHint}`);
      setCode(Array(6).fill(''));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không thể gửi lại mã xác thực.');
    } finally {
      setBusy(false);
    }
  };

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy) return;
    setError('');
    setNotice('');

    if (step === 'login') {
      const emailVal = identity.trim();
      if (!emailVal || !password) {
        setError('Vui lòng nhập email và mật khẩu.');
        return;
      }
      if (!validContact(emailVal, 'email')) {
        setError('Vui lòng nhập đúng định dạng email tài khoản.');
        return;
      }
      setBusy(true);
      try {
        const res = await authApi.login(emailVal, password);
        const session: AuthUserSession = {
          user_id: res.user_id,
          email: res.email,
          access: res.access,
          refresh: res.refresh,
        };
        saveAuthSession(session);
        onAuthenticated?.(session);
        onClose();
      } catch (err) {
        if (err instanceof AuthApiError && err.status === 403 && err.rawCode?.toLowerCase().includes('not verified')) {
          setContact(emailVal);
          setContactKind('email');
          setSource('register');
          try {
            const sent = await authApi.sendVerifyCode(emailVal);
            const devHint = sent.verify_code ? ` (Mã dev: ${sent.verify_code})` : '';
            changeStep('verify', `Tài khoản chưa xác thực. Đã gửi mã mới tới ${emailVal}.${devHint}`);
          } catch {
            changeStep('verify', 'Tài khoản chưa xác thực. Vui lòng nhập mã 6 số hoặc bấm gửi lại mã.');
          }
        } else {
          setError(err instanceof Error ? err.message : 'Đăng nhập thất bại.');
        }
      } finally {
        setBusy(false);
      }
      return;
    }

    if (step === 'register') {
      if (contactKind === 'phone') {
        setError('Hệ thống tài khoản hiện xác thực qua Email. Vui lòng chọn tab Email.');
        return;
      }
      if (!validContact(contact, 'email')) {
        setError('Email chưa hợp lệ.');
        return;
      }
      const pwErr = validateStrongPassword(password);
      if (pwErr) {
        setError(pwErr);
        return;
      }
      if (password !== confirmation) {
        setError('Mật khẩu xác nhận không khớp.');
        return;
      }
      setBusy(true);
      try {
        const res = await authApi.register(contact, password, confirmation);
        setSource('register');
        const devHint = res.dev_verify_code ? ` (Mã dev: ${res.dev_verify_code})` : '';
        changeStep('verify', `Đăng ký thành công! Đã gửi mã 6 số tới ${res.email}.${devHint}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Đăng ký thất bại.');
      } finally {
        setBusy(false);
      }
      return;
    }

    if (step === 'recover') {
      if (contactKind === 'phone') {
        setError('Khôi phục mật khẩu hiện hỗ trợ qua Email. Vui lòng chọn tab Email.');
        return;
      }
      if (!validContact(contact, 'email')) {
        setError('Email chưa hợp lệ.');
        return;
      }
      setBusy(true);
      try {
        const res = await authApi.sendVerifyCode(contact);
        setSource('recover');
        const devHint = res.verify_code ? ` (Mã dev: ${res.verify_code})` : '';
        changeStep('verify', `Đã gửi mã xác thực 6 số tới ${res.email}.${devHint}`);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Không thể gửi mã xác thực.');
      } finally {
        setBusy(false);
      }
      return;
    }

    if (step === 'verify') {
      const joined = code.join('');
      if (joined.length !== 6) {
        setError('Vui lòng nhập đủ 6 chữ số.');
        return;
      }
      setBusy(true);
      try {
        if (source === 'recover') {
          const res = await authApi.validateVerifyCode(contact, joined);
          if (!res.is_valid) {
            setError(res.is_expired ? 'Mã xác thực đã hết hạn. Hãy bấm gửi lại mã mới.' : 'Mã xác thực không hợp lệ.');
            return;
          }
          setVerifiedCode(joined);
          changeStep('reset', 'Mã xác thực hợp lệ. Hãy nhập mật khẩu mới của bạn.');
        } else {
          await authApi.confirmVerifyCode(contact, joined);
          setIdentity(contact.trim());
          changeStep('login', 'Xác thực tài khoản thành công! Bạn có thể đăng nhập ngay bây giờ.');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Xác thực mã thất bại.');
      } finally {
        setBusy(false);
      }
      return;
    }

    // step === 'reset'
    const pwErr = validateStrongPassword(password);
    if (pwErr) {
      setError(pwErr);
      return;
    }
    if (password !== confirmation) {
      setError('Mật khẩu xác nhận không khớp.');
      return;
    }
    setBusy(true);
    try {
      await authApi.resetPassword(contact, verifiedCode || code.join(''), password, confirmation);
      setIdentity(contact.trim());
      setVerifiedCode('');
      changeStep('login', 'Đổi mật khẩu thành công! Hãy đăng nhập bằng mật khẩu mới.');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Không thể đổi mật khẩu.');
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;

  return <div className="auth-backdrop" onMouseDown={event => { if (event.target === event.currentTarget && !busy) onClose(); }}>
    <section className="auth-dialog" role="dialog" aria-modal="true" aria-labelledby="auth-title" aria-describedby="auth-subtitle">
      <button className="auth-close" type="button" onClick={onClose} aria-label="Đóng" disabled={busy}>×</button>
      <div className="auth-brand">
        <div className="auth-brand-inner">
          <img className="auth-logo" src={logo} alt="Harmony Auralis" />
          <span>Không gian âm nhạc của bạn</span>
        </div>
      </div>
      <div className="auth-content">
        <span className="auth-kicker">HARMONY AURALIS</span>
        <h2 id="auth-title">{headings[step].title}</h2>
        <p id="auth-subtitle" className="auth-subtitle">{headings[step].subtitle}</p>
        {step === 'login' && <LoginScreen identity={identity} password={password} busy={busy} setIdentity={setIdentity} setPassword={setPassword} onSubmit={e => void submit(e)} onForgot={() => changeStep('recover')} firstField={firstField} />}
        {step === 'register' && <RegisterScreen username={username} contact={contact} contactKind={contactKind} password={password} confirmation={confirmation} busy={busy} setUsername={setUsername} setContact={setContact} setContactKind={changeContactKind} setPassword={setPassword} setConfirmation={setConfirmation} onSubmit={e => void submit(e)} firstField={firstField} />}
        {step === 'recover' && <RecoverScreen contact={contact} contactKind={contactKind} busy={busy} setContact={setContact} setContactKind={changeContactKind} onSubmit={e => void submit(e)} firstField={firstField} />}
        {step === 'verify' && <VerifyScreen contact={contact} code={code} busy={busy} setCode={setCode} onSubmit={e => void submit(e)} onResend={() => void handleResendCode()} firstField={element => { firstField.current = element; }} />}
        {step === 'reset' && <ResetScreen password={password} confirmation={confirmation} busy={busy} setPassword={setPassword} setConfirmation={setConfirmation} onSubmit={e => void submit(e)} firstField={firstField} />}
        {error && <p className="auth-message auth-error" role="alert">{error}</p>}
        {notice && <p className="auth-message auth-notice" role="status">{notice}</p>}
        <div className="auth-footer">
          {step === 'login' ? <>Chưa có tài khoản? <button type="button" onClick={() => changeStep('register')} disabled={busy}>Đăng ký</button></> :
            step === 'register' ? <>Đã có tài khoản? <button type="button" onClick={() => changeStep('login')} disabled={busy}>Đăng nhập</button></> :
            <button type="button" onClick={() => changeStep(step === 'reset' ? 'verify' : 'login')} disabled={busy}>← Quay lại {step === 'reset' ? 'xác thực' : 'đăng nhập'}</button>}
        </div>
      </div>
    </section>
  </div>;
}
