import type { FormEvent, Ref } from 'react';
import { FloatingField } from '../components/fields';
import { PasswordStrength } from '../components/PasswordStrength';

export function ResetScreen({ password, confirmation, busy, setPassword, setConfirmation, onSubmit, firstField }: {
  password: string; confirmation: string; busy?: boolean; setPassword: (value: string) => void; setConfirmation: (value: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void; firstField: Ref<HTMLInputElement>;
}) {
  return <form className="auth-form" onSubmit={onSubmit} noValidate>
    <FloatingField label="Mật khẩu mới" type="password" value={password} onChange={setPassword} inputRef={firstField} autoComplete="new-password" required />
    <PasswordStrength password={password} />
    <FloatingField label="Xác thực mật khẩu" type="password" value={confirmation} onChange={setConfirmation} autoComplete="new-password" required />
    <button className="auth-submit" type="submit" disabled={busy}>{busy ? 'Đang đổi mật khẩu…' : 'Đổi mật khẩu'}</button>
  </form>;
}
