"use client";

import { useState } from "react";

import type { SavedGame } from "@/lib/types";

const ORDINALS = ["1st", "2nd", "3rd", "4th"];

function when(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

function outcome(game: SavedGame): string {
  const name = (seat: number) => (seat === game.human_seat ? "You" : (game.seats[seat]?.label ?? `Seat ${seat}`));
  if (game.human_seat !== null) {
    const mine = game.scores[game.human_seat] ?? 0;
    if (game.winners.includes(game.human_seat)) return game.winners.length > 1 ? `You tied for 1st, ${mine}` : `You won, ${mine}`;
    const place = game.scores.filter((score) => score > mine).length;
    return `You finished ${ORDINALS[place] ?? `${place + 1}th`}, ${mine}`;
  }
  const best = game.scores[game.winners[0] ?? 0] ?? 0;
  return `${game.winners.map(name).join(" and ")} won, ${best}`;
}

interface Props {
  games: SavedGame[];
  onSameDeals: (game: SavedGame) => void;
  onReview: (game: SavedGame) => void;
}

/** Finished games saved on the server, with their seeds: deal the same cards again, or review
 * the game with every hand shown. */
export function PastGames({ games, onSameDeals, onReview }: Props) {
  const [all, setAll] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  if (games.length === 0) return null;
  const shown = all ? games : games.slice(0, 5);

  const copy = async (game: SavedGame) => {
    try {
      await navigator.clipboard.writeText(String(game.seed));
      setCopied(game.id);
    } catch {
      setCopied(null);
    }
  };

  return (
    <section className="panel" aria-label="Past games">
      <h2 className="panel__title">Past games</h2>
      <ol className="past">
        {shown.map((game) => {
          const won = game.human_seat !== null && game.winners.includes(game.human_seat);
          return (
            <li key={game.id} className="past__row">
              <div className="past__top">
                <span className={`tag ${game.mode === "play" ? (won ? "tag--accent" : "tag--soft") : "tag--muted"}`}>
                  {game.mode === "play" ? "Played" : "Watched"}
                </span>
                <span className="past__outcome">{outcome(game)}</span>
                <span className="past__when">{when(game.finished_at)}</span>
              </div>
              <p className="past__seats">
                {game.seats.map((seat, i) => (
                  <span key={i}>
                    {i === game.human_seat ? "You" : seat.label} <b>{game.scores[i]}</b>
                  </span>
                ))}
              </p>
              <div className="past__actions">
                <button type="button" className="past__seed" title="Copy the seed" onClick={() => void copy(game)}>
                  seed <span className="mono">{game.seed}</span>
                  <span className="past__copied">{copied === game.id ? "copied" : "copy"}</span>
                </button>
                <span className="spacer" />
                <button type="button" className="btn btn--ghost btn--small" onClick={() => onSameDeals(game)}>
                  Same deals
                </button>
                <button type="button" className="btn btn--ghost btn--small" onClick={() => onReview(game)}>
                  Review
                </button>
              </div>
            </li>
          );
        })}
      </ol>
      {games.length > 5 && (
        <button type="button" className="link" onClick={() => setAll(!all)}>
          {all ? "Show fewer" : `Show all ${games.length}`}
        </button>
      )}
    </section>
  );
}
