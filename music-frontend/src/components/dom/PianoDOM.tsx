'use dom';

import PianoKeyboard from '../../features/user/piano/screen/PianoKeyboard';

export default function PianoDOM({ dom: _dom }: { dom?: import('expo/dom').DOMProps }) {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <PianoKeyboard />
    </div>
  );
}
