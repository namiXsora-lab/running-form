import { useEffect } from "react";

// checkout.js の goToCheckout() を使う想定（名前が違ったら後で合わせよう）
import { goToCheckout } from "./checkout";

export default function StartSubscribe() {
  useEffect(() => {
    goToCheckout();
  }, []);

  return (
    <div style={{ padding: 24 }}>
      <h2>決済ページへ移動しています…</h2>
      <p>うまく移動しない場合は、少し待ってからもう一度お試しください。</p>
    </div>
  );
}
