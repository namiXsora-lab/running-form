import SuccessThanks from "./SuccessThanks";
import CancelInfo from "./CancelInfo";
import MainFormApp from "./MainFormApp";
import StartSubscribe from "./StartSubscribe";

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const status = params.get("status");
  const start = params.get("start");

  if (start === "subscribe") return <StartSubscribe />;
  if (status === "success") return <SuccessThanks />;
  if (status === "cancel") return <CancelInfo />;
  return <MainFormApp />;
}
