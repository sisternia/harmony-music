const DEFAULT_ACCOUNTS_API =
  (typeof process !== "undefined" && process.env?.EXPO_PUBLIC_ACCOUNTS_API_URL) ||
  "http://127.0.0.1:8000";

const AUTH_STORAGE_KEY = "harmony_auralis_auth_session";

export type AuthUserSession = {
  user_id: number;
  email: string;
  access: string;
  refresh: string;
};

export type LoginResponse = AuthUserSession & {
  message: string;
};

export type RegisterResponse = {
  message: string;
  user_id: number;
  email: string;
  verify_id: number;
  dev_verify_code?: string;
};

export type SendVerifyCodeResponse = {
  message: string;
  verify_id: number;
  user_id: number;
  email: string;
  verify_code?: string;
  verify_status: string;
  expire_minutes: number;
};

export type ConfirmVerifyCodeResponse = {
  message: string;
  verify_status: string;
  verify_time?: string | null;
};

export type ValidateVerifyCodeResponse = {
  message: string;
  verify_status: string;
  is_valid: boolean;
  is_expired: boolean;
  expire_minutes: number;
};

export type ResetPasswordResponse = {
  message: string;
  email: string;
  user_id: number;
};

let accountsBase = DEFAULT_ACCOUNTS_API.replace(/\/$/, "");

export function configureAccountsApiBase(base: string) {
  if (base) accountsBase = base.replace(/\/$/, "");
}

function accountsUrl(path: string) {
  return `${accountsBase}${path}`;
}

function translateBackendError(raw: string): string {
  const lower = raw.toLowerCase();
  if (lower.includes("email already exists")) return "Email này đã được đăng ký.";
  if (lower.includes("email address is invalid")) return "Địa chỉ email không hợp lệ.";
  if (lower.includes("account not found")) return "Không tìm thấy tài khoản với email này.";
  if (lower.includes("account is not verified")) return "Tài khoản chưa được xác thực email. Vui lòng nhập mã 6 số để xác thực.";
  if (lower.includes("invalid email or password")) return "Email hoặc mật khẩu không đúng.";
  if (lower.includes("passwords do not match")) return "Mật khẩu xác nhận không khớp.";
  if (lower.includes("at least 8 characters")) return "Mật khẩu cần ít nhất 8 ký tự.";
  if (lower.includes("uppercase letter")) return "Mật khẩu cần có ít nhất 1 chữ in hoa (A–Z).";
  if (lower.includes("lowercase letter")) return "Mật khẩu cần có ít nhất 1 chữ thường (a–z).";
  if (lower.includes("one number")) return "Mật khẩu cần có ít nhất 1 chữ số (0–9).";
  if (lower.includes("special character")) return "Mật khẩu cần có ít nhất 1 ký tự đặc biệt (!@#$%^&*...).";
  if (lower.includes("verification code not found")) return "Mã xác thực không đúng.";
  if (lower.includes("expired")) return "Mã xác thực đã hết hạn. Hãy gửi lại mã mới.";
  if (lower.includes("already verified")) return "Tài khoản này đã được xác thực trước đó.";
  if (lower.includes("enter a valid email")) return "Vui lòng nhập đúng định dạng email.";
  return raw;
}

function extractErrorMessage(body: unknown, fallback: string): string {
  if (!body || typeof body !== "object") return fallback;
  const obj = body as Record<string, unknown>;
  if (obj.errors && typeof obj.errors === "object") {
    const errs = obj.errors as Record<string, unknown>;
    for (const val of Object.values(errs)) {
      if (Array.isArray(val) && val.length > 0 && typeof val[0] === "string") {
        return translateBackendError(val[0]);
      }
      if (typeof val === "string") {
        return translateBackendError(val);
      }
    }
  }
  if (typeof obj.message === "string" && obj.message.trim()) {
    return translateBackendError(obj.message);
  }
  if (typeof obj.detail === "string" && obj.detail.trim()) {
    return translateBackendError(obj.detail);
  }
  return fallback;
}

export class AuthApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public rawCode?: string,
  ) {
    super(message);
  }
}

async function postJson<T>(path: string, payload: Record<string, unknown>): Promise<T> {
  let res: Response;
  try {
    res = await fetch(accountsUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new AuthApiError(
      0,
      "Không thể kết nối đến máy chủ tài khoản (Django :8000). Hãy chắc chắn server backend đang chạy.",
    );
  }

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }

  if (!res.ok) {
    const rawMsg =
      body && typeof body === "object" && typeof (body as Record<string, unknown>).message === "string"
        ? String((body as Record<string, unknown>).message)
        : undefined;
    throw new AuthApiError(
      res.status,
      extractErrorMessage(body, `Lỗi máy chủ (${res.status})`),
      rawMsg,
    );
  }
  return body as T;
}

export const authApi = {
  login: (email: string, password: string) =>
    postJson<LoginResponse>("/api/accounts/auth/login/", { email: email.trim(), password }),

  register: (email: string, password: string, confirm_password: string) =>
    postJson<RegisterResponse>("/api/accounts/auth/register/", {
      email: email.trim(),
      password,
      confirm_password,
    }),

  sendVerifyCode: (email: string) =>
    postJson<SendVerifyCodeResponse>("/api/accounts/auth/send-verify-code/", {
      email: email.trim(),
    }),

  confirmVerifyCode: (email: string, verify_code: string) =>
    postJson<ConfirmVerifyCodeResponse>("/api/accounts/auth/confirm-verify-code/", {
      email: email.trim(),
      verify_code: verify_code.trim(),
    }),

  validateVerifyCode: (email: string, verify_code: string) =>
    postJson<ValidateVerifyCodeResponse>("/api/accounts/auth/validate-verify-code/", {
      email: email.trim(),
      verify_code: verify_code.trim(),
    }),

  resetPassword: (
    email: string,
    verify_code: string,
    new_password: string,
    confirm_new_password: string,
  ) =>
    postJson<ResetPasswordResponse>("/api/accounts/auth/reset-password/", {
      email: email.trim(),
      verify_code: verify_code.trim(),
      new_password,
      confirm_new_password,
    }),
};

// Local session persistence + custom event so Navbar/Library updates immediately
const AUTH_CHANGE_EVENT = "harmony-auralis-auth-changed";

export function loadAuthSession(): AuthUserSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<AuthUserSession>;
    if (parsed && typeof parsed.email === "string" && typeof parsed.access === "string") {
      return parsed as AuthUserSession;
    }
    return null;
  } catch {
    return null;
  }
}

export function saveAuthSession(session: AuthUserSession) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(session));
    window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
  } catch {
    // ignore storage quota errors
  }
}

export function clearAuthSession() {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    window.dispatchEvent(new Event(AUTH_CHANGE_EVENT));
  } catch {
    // ignore
  }
}

export function subscribeAuthSession(listener: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  const handler = () => listener();
  window.addEventListener(AUTH_CHANGE_EVENT, handler);
  window.addEventListener("storage", handler);
  return () => {
    window.removeEventListener(AUTH_CHANGE_EVENT, handler);
    window.removeEventListener("storage", handler);
  };
}

