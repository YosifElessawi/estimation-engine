"use client";

import type { ReactNode } from "react";

import { trumpName } from "@/lib/cards";
import { seatName, signed } from "@/lib/seats";
import type { Frame, GameView, RoundResultView } from "@/lib/types";

import { TrumpGlyph } from "./ActionText";
import { Avatar } from "./Table";

type SeatResult = RoundResultView["seats"][number];

function breakdown(seat: SeatResult): { label: string; points: number }[] {
  const parts = [
    { label: seat.won ? "Base" : "Missed", points: seat.base_points },
    { label: "Dash Call", points: seat.dash_points },
    { label: seat.caller ? "Caller" : "With", points: seat.call_points },
    { label: "Risk", points: seat.risk_points },
    { label: seat.won ? "Only winner" : "Only loser", points: seat.only_points },
    { label: "Super call", points: seat.super_points },
  ];
  return parts.filter((part) => part.points !== 0);
}

function roleOf(seat: SeatResult): string {
  const roles = [];
  if (seat.caller) roles.push("Caller");
  if (seat.with_player) roles.push("With");
  if (seat.dash) roles.push("Dash");
  if (seat.super_call) roles.push("Super call");
  if (seat.last) roles.push(seat.risks ? `Last, ${seat.risks} Risk${seat.risks > 1 ? "s" : ""}` : "Last");
  return roles.join(" · ");
}

function headline(game: GameView, winners: number[]): string {
  const names = winners.map((seat) => seatName(game, seat));
  if (names.length > 1) return `${names.join(" and ")} share the win`;
  if (winners[0] === game.human_seat && game.mode === "play") return "You win!";
  return `${names[0] ?? "Nobody"} wins`;
}

export function RoundResult({
  frame,
  game,
  bottom,
  actions,
}: {
  frame: Frame;
  game: GameView;
  bottom: number;
  actions: ReactNode;
}) {
  const result = frame.result;
  if (!result) return null;
  const order = [0, 1, 2, 3].map((i) => (bottom + i) % 4);
  const final = frame.kind === "game_end";
  const winners = frame.winners ?? [];
  const ranked = [0, 1, 2, 3].sort((a, b) => (frame.scores[b] ?? 0) - (frame.scores[a] ?? 0));
  const human = (seat: number) => game.seats[seat]?.human ?? false;
  return (
    <div className="veil" role="dialog" aria-label={final ? "Game over" : `Round ${result.round_no} result`}>
      <div className={`result ${final ? "result--final" : ""}`}>
        {final && (
          <header className="result__final">
            <span className="eyebrow">Game over</span>
            <h2>{headline(game, winners)}</h2>
            <ol className="podium">
              {ranked.map((seat, place) => (
                <li key={seat} className={winners.includes(seat) ? "is-winner" : ""}>
                  <span className="podium__place">{place + 1}</span>
                  <Avatar seat={seat} human={human(seat)} name={seatName(game, seat)} />
                  <span className="podium__name">{seatName(game, seat)}</span>
                  <span className="podium__score">{frame.scores[seat]}</span>
                </li>
              ))}
            </ol>
          </header>
        )}
        <header className="result__head">
          <span className="eyebrow">{final ? `Last round, round ${result.round_no}` : `Round ${result.round_no}`}</span>
          <h3>
            {result.trump !== null ? (
              <>
                <TrumpGlyph trump={result.trump} /> {trumpName(result.trump)}
              </>
            ) : (
              "No trump"
            )}
            {!result.skipped && result.total !== null && result.over !== null && (
              <span className="result__over">
                {result.over ? `Over by ${result.total - 13}` : `Under by ${13 - result.total}`}
              </span>
            )}
          </h3>
          <p className="result__facts">
            {result.skipped && "Skipped: no Caller, everyone scores 0. "}
            {!result.skipped && result.multiplier !== 1 && `Scored x${result.multiplier}. `}
            {result.all_lose && "Everyone lost. "}
            {result.trump_changed &&
              (result.replayed ? "Trump changed, so the round is replayed next." : "Trump changed (replay limit reached).")}
          </p>
        </header>
        {!result.skipped && (
          <ol className="result__rows">
            {order.map((seat) => {
              const row = result.seats[seat];
              if (!row) return null;
              return (
                <li key={seat} className={`result__row ${row.won ? "is-win" : "is-loss"}`}>
                  <Avatar seat={seat} human={human(seat)} name={seatName(game, seat)} />
                  <span className="result__who">
                    <span className="result__name">{seatName(game, seat)}</span>
                    <span className="result__role">{roleOf(row)}</span>
                  </span>
                  <span className="result__tricks">
                    <b>{row.tricks}</b> of {row.estimate}
                    <span className="result__mark" aria-label={row.won ? "made it" : "missed"}>
                      {row.won ? "✓" : "✗"}
                    </span>
                  </span>
                  <span className="result__chips">
                    {breakdown(row).map((part) => (
                      <span key={part.label} className={`chip ${part.points > 0 ? "chip--plus" : "chip--minus"}`}>
                        {part.label} {signed(part.points)}
                      </span>
                    ))}
                    {row.multiplier !== 1 && <span className="chip chip--mult">x{row.multiplier}</span>}
                  </span>
                  <span className={`result__score ${row.score > 0 ? "is-plus" : row.score < 0 ? "is-minus" : ""}`}>
                    {signed(row.score)}
                  </span>
                  <span className="result__total" title="Total">
                    {frame.scores[seat]}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
        {result.next_carry > 1 && !final && <p className="result__carry">Next round counts x{result.next_carry}</p>}
        <div className="result__actions">{actions}</div>
      </div>
    </div>
  );
}
