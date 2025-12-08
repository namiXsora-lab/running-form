// src/main.jsx
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";                  // ← ランニング用（従来どおり）
import PitchCompare from "./PitchCompare.tsx"; // ← 追加：投球フォーム比較

function Root() {
  const isPitch = window.location.hash === "#pitch";
  // ハッシュが変わったら即時反映
  const [, force] = React.useReducer(x => x + 1, 0);
  React.useEffect(() => {
    const onHash = () => force();
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  return isPitch ? <PitchCompare /> : <App />;
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
