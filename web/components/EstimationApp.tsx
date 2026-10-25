"use client";

import { useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { BotView, GameView } from "@/lib/types";

import { GameScreen } from "./GameScreen";
import { Setup } from "./Setup";

/** The whole app: pick seats, then watch or play the game. */
export function EstimationApp() {
  const [bots, setBots] = useState<BotView[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [game, setGame] = useState<GameView | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = (attempt: number) => {
      api
        .bots()
        .then((found) => {
          if (!cancelled) {
            setBots(found);
            setLoadError(null);
          }
        })
        .catch((error: unknown) => {
          if (cancelled) return;
          setLoadError(error instanceof Error ? error.message : String(error));
          if (attempt < 20) setTimeout(() => load(attempt + 1), 1500);
        });
    };
    load(0);
    return () => {
      cancelled = true;
    };
  }, []);

  if (game) {
    return <GameScreen key={game.id} initial={game} onNewGame={() => setGame(null)} onOpen={setGame} />;
  }
  return <Setup bots={bots} loadError={loadError} onStart={setGame} />;
}
