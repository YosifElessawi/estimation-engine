// Calls to the FastAPI server, proxied by Next at /api.

import type { BotView, FramesView, GameView, SeekTarget } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
      else if (Array.isArray(body.detail)) message = JSON.stringify(body.detail);
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export const api = {
  bots: () => call<BotView[]>("/bots"),
  newGame: (seats: string[], seed: number | null) =>
    call<GameView>("/games", { method: "POST", body: JSON.stringify({ seats, seed }) }),
  game: (id: string) => call<GameView>(`/games/${id}`),
  frames: (id: string, start: number, count: number) =>
    call<FramesView>(`/games/${id}/frames?start=${start}&count=${count}`),
  seek: (id: string, start: number, direction: "next" | "prev", target: SeekTarget) =>
    call<{ index: number }>(
      `/games/${id}/seek?start=${start}&direction=${direction}&target=${target}`,
    ),
  move: (id: string, action: number, frame: number) =>
    call<FramesView>(`/games/${id}/moves`, {
      method: "POST",
      body: JSON.stringify({ action, frame }),
    }),
  review: (id: string) => call<GameView>(`/games/${id}/review`, { method: "POST" }),
};
