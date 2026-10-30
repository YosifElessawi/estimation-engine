// Calls to the FastAPI server, proxied by Next at /api (the public table serves the page and
// /api itself).

import type { CreatedGame, FramesView, GameView, SeekTarget } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

// The public table's game token, sent with every call about that game (server/public).
let gameToken: { id: string; token: string } | null = null;

export async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (gameToken && path.startsWith(`/games/${gameToken.id}`)) headers["X-Game-Token"] = gameToken.token;
  const response = await fetch(`/api${path}`, {
    ...init,
    headers: { ...headers, ...init?.headers },
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

/** The calls the game screen makes, at either table. The lobby's own are in lobbyApi.ts. */
export const api = {
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
  // The public table
  setToken: (id: string, token: string) => {
    gameToken = { id, token };
  },
  startPublic: (name: string, visitor: string) =>
    call<CreatedGame>("/games", { method: "POST", body: JSON.stringify({ name, visitor }) }),
  leave: (id: string) => call<{ left: boolean }>(`/games/${id}/leave`, { method: "POST" }),
};
