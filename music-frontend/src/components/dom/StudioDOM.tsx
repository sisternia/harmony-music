'use dom';

import ExpoStudio from "../../features/studio/screen/ExpoStudio";

export default function StudioDOM({
  apiBase,
  path,
}: {
  apiBase: string;
  path: string;
  dom?: import("expo/dom").DOMProps;
}) {
  return (
    <div style={{ width: "100%", height: "100%" }}>
      <ExpoStudio apiBase={apiBase} path={path} />
    </div>
  );
}
