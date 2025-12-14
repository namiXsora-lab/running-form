// src/SuccessThanks.jsx

export default function SuccessThanks() {
  const goNext = () => {
    // ✅ クエリ（status / session_id）を消して、通常画面へ戻す
    // いちばん確実に状態をリセットできるので、フェーズ1ではこれでOK
    window.location.href = window.location.pathname;
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "linear-gradient(180deg, rgba(109,191,242,0.18), rgba(255,255,255,1))",
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
        <div style={{ fontSize: 28, marginBottom: 10 }}>🌤️</div>

        <h1 style={{ fontSize: 22, margin: "0 0 10px", color: "#2A6EBB" }}>
          ご登録ありがとうございます
        </h1>

        <p
          style={{
            fontSize: 14,
            lineHeight: 1.9,
            margin: "0 0 18px",
            color: "#333",
          }}
        >
          あなたのフォームを、もっと優しく・ていねいに見つめる時間を
          <br />
          SORA LABと一緒に育てていけたらうれしいです。
          <br />
          ひと息つきながら、少しずつ一緒に進みましょう。
          <br />
          <span style={{ color: "#2A6EBB", fontWeight: "bold" }}>
            ここから、あなたのペースで始められます。
          </span>
        </p>

        {/* ✅ 事実情報（不安を消す）をまとめて補足欄へ */}
        <div
          style={{
            background: "rgba(109,191,242,0.12)",
            borderRadius: 14,
            padding: "10px 12px",
            fontSize: 12,
            color: "#333",
            margin: "0 0 14px",
            lineHeight: 1.7,
          }}
        >
          決済は正常に完了しています。
          <br />
          登録完了メールが届くまで、少し時間がかかることがあります。
          <br />
          <br />
          ご安心ください。定期購入はいつでも解約できます。
          <br />
          解約後も、次回更新日まではご利用いただけます。
        </div>

        <button
          onClick={goNext}
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
          フォーム比較をはじめる
        </button>

        <div style={{ marginTop: 12, fontSize: 12, color: "#666" }}>
          うまく表示されないときは、ページを更新してね。
        </div>
      </div>
    </div>
  );
}
