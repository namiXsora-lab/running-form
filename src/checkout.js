// src/checkout.js

const CHECKOUT_URL =
  "https://iap7rjj3eccm26idib5qbjhfhu0tcfdd.lambda-url.ap-northeast-1.on.aws/";

export async function goToCheckout() {
  try {
    // ★ とりあえず GET、ヘッダーも付けない → preflight が飛ばない
    const res = await fetch(CHECKOUT_URL, {
      method: "GET",
    });

    const text = await res.text();
    console.log("Lambda raw response:", res.status, text);

    let data = {};
    try {
      data = JSON.parse(text);
    } catch (parseErr) {
      console.error("JSONパースエラー:", parseErr);
      alert("サーバーからの応答が不正です（JSONではありません）。");
      return;
    }

    if (!res.ok) {
      console.error("Lambda側のエラー:", data);
      alert("サーバー側でエラーが発生しました（Stripe設定か環境変数を確認してください）。");
      return;
    }

    if (data.url) {
      window.location.href = data.url; // Stripe決済画面へ
    } else {
      console.error("url がレスポンスに含まれていません:", data);
      alert("決済ページを開けませんでした。");
    }
  } catch (e) {
    console.error("goToCheckout 内での想定外エラー:", e);
    alert("決済処理でエラーが発生しました。詳細はコンソールを確認してください。");
  }
}
