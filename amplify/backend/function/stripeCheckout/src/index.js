// amplify/backend/function/stripeCheckout/src/index.js

// ▼ 環境変数（Lambdaの「環境変数」と名前を合わせる）
const STRIPE_SECRET_KEY = process.env.STRIPE_SECRET_KEY;
const PRICE_ID = process.env.STRIPE_PRICE_ID;
const SUCCESS_URL =
  process.env.CHECKOUT_SUCCESS_URL || "https://sora-lab-app.com/success";
const CANCEL_URL =
  process.env.CHECKOUT_CANCEL_URL || "https://sora-lab-app.com/";

// Stripe 初期化
const stripe = require("stripe")(STRIPE_SECRET_KEY);

// ▼ CORSヘッダー（開発中なのでゆるめ設定）
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Methods": "OPTIONS,GET,POST",
};

// ▼ メインハンドラ
exports.handler = async (event) => {
  console.log("Incoming event:", JSON.stringify(event));

  // 1) プリフライト(OPTIONS)対応
  const method = event.requestContext?.http?.method;
  if (method === "OPTIONS") {
    return {
      statusCode: 200,
      headers: CORS_HEADERS,
      body: "",
    };
  }

  // 2) 本処理（Stripe Checkoutセッション作成）
  try {
    // フロントから何か送る場合はここで body を読む（今回は未使用）
    const body = event.body ? JSON.parse(event.body) : {};
    console.log("Request body:", body);

    // サブスク用Checkoutセッション作成
    const session = await stripe.checkout.sessions.create({
      mode: "subscription",
      line_items: [
        {
          price: PRICE_ID,
          quantity: 1,
        },
      ],
      success_url: SUCCESS_URL,
      cancel_url: CANCEL_URL,
    });

    console.log("Created session:", session.id);

    return {
      statusCode: 200,
      headers: {
        ...CORS_HEADERS,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ url: session.url }),
    };
  } catch (error) {
    console.error("Stripe checkout error:", error);

    return {
      statusCode: 500,
      headers: {
        ...CORS_HEADERS,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: "Failed to create checkout session",
        error: error.message,
      }),
    };
  }
};
