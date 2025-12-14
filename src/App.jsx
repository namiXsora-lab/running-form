import { useEffect } from "react";
import SuccessThanks from "./SuccessThanks";
import CancelInfo from "./CancelInfo";
import MainFormApp from "./MainFormApp";
import StartSubscribe from "./StartSubscribe";
import { goToCheckout } from "./checkout";

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const status = params.get("status");
  const start = params.get("start");

  // 🔽 先に副作用（決済開始）を書く
  useEffect(() => {
    if (start === "subscribe") {
      goToCheckout(); // Stripeへ飛ばす
    }
  }, [start]);

  // 🔽 その後で画面分岐
  if (start === "subscribe") return <StartSubscribe />;
  if (status === "success") return <SuccessThanks />;
  if (status === "cancel") return <CancelInfo />;

  return <MainFormApp />;
}
