'use dom';

import AuthDialog from '../../features/auth/screen/AuthDialog';

export default function AuthDOM({ mode }: {
  mode: 'login' | 'register';
  dom?: import('expo/dom').DOMProps;
}) {
  return (
    <div className="auth-page">
      <AuthDialog
        key={mode}
        open
        initialStep={mode}
        logo="/images/harmony-auralis-logo.png"
        onClose={() => { window.location.href = '/'; }}
      />
    </div>
  );
}
