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
  const shown = all ? games : games.slice(0, 6);

  const copy = async (game: SavedGame) => {
    try {
      await navigator.clipboard.writeText(String(game.seed));
      setCopied(game.id);
    } catch {
      setCopied(null);
    }
  };

  return (
    <section className="past" aria-label="Past games">
      <h2 className="panel-title">Past games</h2>
      <ol className="past__list">
        {shown.map((game) => (
          <li key={game.id} className="past__row">
            <div className="past__head">
              <span className={`badge ${game.mode === "play" ? "badge--teal" : "badge--muted"}`}>
                {game.mode === "play" ? "Played" : "Watched"}
              </span>
              <span className="past__when">{when(game.finished_at)}</span>
              <span className="past__outcome">{outcome(game)}</span>
            </div>
            <p className="past__seats small">
              {game.seats.map((seat, i) => (
                <span key={i}>
                  {i === game.human_seat ? "You" : seat.label} <b className="mono">{game.scores[i]}</b>
                </span>
              ))}
            </p>
            <div className="past__foot">
              <button
                type="button"
                className="past__seed mono"
                title="Copy the seed"
                onClick={() => void copy(game)}
              >
                seed {game.seed}
                <span className="past__copied">{copied === game.id ? "copied" : "copy"}</span>
              </button>
              <span className="topbar__spacer" />
              <button type="button" className="button button--ghost" onClick={() => onSameDeals(game)}>
                Same deals
              </button>
              <button type="button" className="button button--ghost" onClick={() => onReview(game)}>
                Review
              </button>
            </div>
          </li>
        ))}
      </ol>
      {games.length > 6 && (
        <button type="button" className="link small" onClick={() => setAll(!all)}>
          {all ? "show fewer" : `show all ${games.length}`}
        </button>
      )}
    </section>
  );
}
