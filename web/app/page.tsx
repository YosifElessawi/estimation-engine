import { EstimationApp } from "@/components/EstimationApp";
import { PublicApp } from "@/components/PublicApp";

// NEXT_PUBLIC_EDITION=public builds the public table (npm run build:public): a name, then a game
// against three B-Tig bots. Checked here, not imported, so the build drops the other app.
export default function Home() {
  return process.env.NEXT_PUBLIC_EDITION === "public" ? <PublicApp /> : <EstimationApp />;
}
