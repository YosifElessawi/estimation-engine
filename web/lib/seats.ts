import type { GameView } from "./types";

export type Position = "bottom" | "right" | "top" | "left";

const POSITIONS: Position[] = ["bottom", "right", "top", "left"];

/** Where ``seat`` sits for a viewer at ``bottom``. Play goes counterclockwise, so the next seat
 * sits to the right (TBL-2). */
export function positionOf(seat: number, bottom: number): Position {
  return POSITIONS[(seat - bottom + 4) % 4] ?? "bottom";
}

export function seatName(game: GameView, seat: number): string {
  const info = game.seats[seat];
  if (!info) return `Seat ${seat}`;
  return info.human && game.mode === "play" ? "You" : info.label;
}

export const formatElo = (elo: number | null): string =>
  elo === null ? "unrated" : elo.toLocaleString("en-US");

export const signed = (n: number): string => (n > 0 ? `+${n}` : `${n}`);
