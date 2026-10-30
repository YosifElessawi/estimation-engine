// The full table's lobby calls: bots, new games from seats, past games. Kept apart from api.ts so
// the public page (which has no lobby) never carries them.

import { call } from "./api";
import type { BotView, GameView, SavedGame } from "./types";

export const lobbyApi = {
  bots: () => call<BotView[]>("/bots"),
  newGame: (seats: string[], seed: number | null) =>
    call<GameView>("/games", { method: "POST", body: JSON.stringify({ seats, seed }) }),
  history: () => call<SavedGame[]>("/history"),
  reviewSaved: (id: string) => call<GameView>(`/history/${id}/review`, { method: "POST" }),
};
