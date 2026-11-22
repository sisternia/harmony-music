import type { FormEvent } from 'react';
import { CodeInput } from '../components/CodeInput';

export function VerifyScreen({ contact, code, busy, setCode, onSubmit, onResend, firstField }: {
  contact: string; code: string[]; busy?: boolean; setCode: (code: string[]) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onResend?: () => void;
  firstField: (element: HTMLInputElement | null) => void;
}) {
  return <form className="auth-form" onSubmit={onSubmit} noValidate>
    <p className="auth-code-caption">{contact ? `Mã xác thực cho ${contact}` : 'Nhập mã xác thực của bạn'}</p>
    <CodeInput value={code} onChange={setCode} firstField={firstField} />
    {onResend && (
      <button className="auth-text-link auth-forgot" type="button" onClick={onResend} disabled={busy}>
        Gửi lại mã xác thực
      </button>
    )}
    <button className="auth-submit" type="submit" disabled={busy}>{busy ? 'Đang xác thực…' : 'Xác thực'}</button>
  </form>;
}
