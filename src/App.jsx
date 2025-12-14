import SuccessThanks from "./SuccessThanks";
import CancelInfo from "./CancelInfo";
import MainFormApp from "./MainFormApp";

export default function App() {
  const params = new URLSearchParams(window.location.search);
  const status = params.get("status");

  if (status === "success") return <SuccessThanks />;
  if (status === "cancel") return <CancelInfo />;
  return <MainFormApp />;
}
