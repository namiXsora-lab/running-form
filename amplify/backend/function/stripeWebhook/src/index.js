// amplify/backend/function/stripeWebhook/src/index.js

const AWS = require("aws-sdk");
const ddb = new AWS.DynamoDB.DocumentClient();

const TABLE_NAME = process.env.USER_ACCESS_TABLE;

// メインハンドラー
exports.handler = async (event) => {
  console.log("Raw event from API Gateway / Function URL:", JSON.stringify(event));

  // Stripe からのリクエストボディを取得
  const body = event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64").toString("utf8")
    : event.body || "";

  let stripeEvent;
  try {
    stripeEvent = JSON.parse(body);
  } catch (e) {
    console.error("JSON parse error:", e);
    return response(400, "Invalid JSON");
  }

  const type = stripeEvent.type;
  console.log("Stripe event type:", type);

  // 1) Checkout完了 → とりあえず active で登録
  if (type === "checkout.session.completed") {
    const session = stripeEvent.data.object;
    const subscriptionId = session.subscription || null;
    const customerId = session.customer || null;

    // フロントから渡す予定の Cognito の sub
    const cognitoSub = session.client_reference_id || null;

    if (!cognitoSub) {
      console.log("No cognitoSub (client_reference_id) in session. skip.");
      return response(200, "no cognitoSub");
    }

    const expiresAt =
      session.expires_at != null
        ? new Date(session.expires_at * 1000).toISOString()
        : null;

    await upsertEntitlement({
      cognitoSub,
      status: "active",
      customerId,
      subscriptionId,
      currentPeriodEnd: expiresAt,
    });

    return response(200, "checkout handled");
  }

  // 2) サブスク更新・解約 → 状態を更新
  if (type === "customer.subscription.updated" || type === "customer.subscription.deleted") {
    const sub = stripeEvent.data.object;
    const status = mapStripeStatus(sub.status);
    const customerId = sub.customer || null;
    const subscriptionId = sub.id || null;

    // ここには、あとで Stripe 側の metadata.cognito_sub を入れる想定
    const cognitoSub =
      (sub.metadata && sub.metadata.cognito_sub) || null;

    if (!cognitoSub) {
      console.log("No cognitoSub on subscription.metadata. skip.");
      return response(200, "no cognitoSub on subscription");
    }

    const currentPeriodEnd =
      sub.current_period_end != null
        ? new Date(sub.current_period_end * 1000).toISOString()
        : null;

    await upsertEntitlement({
      cognitoSub,
      status,
      customerId,
      subscriptionId,
      currentPeriodEnd,
    });

    return response(200, "subscription handled");
  }

  // その他のイベントは一旦無視
  return response(200, "ignored");
};

// Stripeのステータス → 自分のステータスにマッピング
function mapStripeStatus(s) {
  if (s === "active" || s === "trialing") return "active";
  if (s === "past_due" || s === "unpaid") return "past_due";
  return "canceled";
}

// DynamoDB に会員情報を書き込む
async function upsertEntitlement({
  cognitoSub,
  status,
  customerId,
  subscriptionId,
  currentPeriodEnd,
}) {
  const pk = `USER#${cognitoSub}`;
  const sk = "ENTITLEMENT#ALL_APPS";

  const now = new Date().toISOString();

  const params = {
    TableName: TABLE_NAME,
    Item: {
      pk,
      sk,
      status,
      stripe_customer_id: customerId,
      stripe_subscription_id: subscriptionId,
      current_period_end: currentPeriodEnd,
      source: "stripe",
      updated_at: now,
    },
  };

  console.log("DynamoDB put:", params);
  await ddb.put(params).promise();
}

function response(statusCode, body) {
  return {
    statusCode,
    headers: { "Content-Type": "text/plain" },
    body: typeof body === "string" ? body : JSON.stringify(body),
  };
}
