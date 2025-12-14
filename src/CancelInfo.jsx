export default function CancelInfo() {
  const back = () => {
    window.location.href = window.location.pathname;
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(180deg, rgba(109,191,242,0.12), rgba(255,255,255,1))",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 20,
      }}
    >
      <div
        style={{
          width: "min(520px, 100%)",
          background: "#fff",
          borderRadius: 24,
          padding: "28px 26px",
          boxShadow: "0 12px 30px rgba(0,0,0,0.10)",
          textAlign: "center",
        }}
      >
        <div style={{ fontSize: 26, marginBottom: 10 }}>☁️</div>

        <h1 style={{ fontSize: 20, margin: "0 0 10px", color: "#2A6EBB" }}>
          お手続きはキャンセルされました
        </h1>

        <p style={{ fontSize: 14, lineHeight: 1.9, margin: "0 0 18px", color: "#333" }}>
          大丈夫です。いつでも、あなたのタイミングでOK。
          <br />
          また準備ができたら、ここから再開できます。
        </p>

        <button
          onClick={back}
          style={{
            width: "100%",
            padding: "12px 14px",
            borderRadius: 999,
            border: "none",
            background: "#3ba9ff",
            color: "#fff",
            fontWeight: "bold",
            cursor: "pointer",
            fontSize: 14,
          }}
        >
          フォーム比較画面に戻る
        </button>
      </div>
    </div>
  );
}
