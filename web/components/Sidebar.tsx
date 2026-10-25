"use client";

import { useState } from "react";

import { seatName, signed } from "@/lib/seats";
import type { Frame, GameView } from "@/lib/types";

import { ActionText, TrumpGlyph } from "./ActionText";

export function Scoreboard({ frame, game }: { frame: Frame; game: GameView }) {
  const best = Math.max(...frame.scores);
  const ranked = [0, 1, 2, 3].sort((a, b) => (frame.scores[b] ?? 0) - (frame.scores[a] ?? 0));
  return (
    <section className="panel">
      <h2 className="panel-title">Scores</h2>
      <ol className="scores">
        {ranked.map((seat) => (
          <li key={seat} className={`scores__row ${frame.scores[seat] === best ? "is-leader" : ""}`}>
            <span className="scores__seat mono">{seat}</span>
            <span className="scores__name">{seatName(game, seat)}</span>
            <span className="scores__total mono">{frame.scores[seat]}</span>
          </li>
        ))}
      </ol>
      {frame.history.length > 0 && (
        <div className="history">
          <table>
            <thead>
              <tr>
                <th>Rd</th>
                <th />
                {[0, 1, 2, 3].map((seat) => (
                  <th key={seat} className="num" title={seatName(game, seat)}>
                    S{seat}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...frame.history].reverse().map((round) => (
                <tr key={round.round_no}>
                  <td className="mono">{round.round_no}</td>
                  <td>
                    {round.skipped ? (
                      <span className="muted">skip</span>
                    ) : round.trump !== null ? (
                      <TrumpGlyph trump={round.trump} />
                    ) : null}
                    {round.multiplier > 1 && <span className="history__mult">x{round.multiplier}</span>}
                  </td>
                  {round.scores.map((score, seat) => (
                    <td key={seat} className={`num mono ${score > 0 ? "plus" : score < 0 ? "minus" : "muted"}`}>
                      {score === 0 ? "0" : signed(score)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export function RoundLog({ frame, game }: { frame: Frame; game: GameView }) {
  const events = frame.table.events;
  return (
    <section className="panel">
      <h2 className="panel-title">Round {frame.table.round_no} calls</h2>
      {events.length === 0 ? (
        <p className="muted small">No calls yet.</p>
      ) : (
        <ol className="log">
          {events.map((event, i) => (
            <li key={i} className={`log__row log__row--${event.action.kind}`}>
              <span className="log__who">{seatName(game, event.seat)}</span>
              <span className="log__what">
                <ActionText action={event.action} />
              </span>
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}

export function WhyPanel({ frame, game }: { frame: Frame; game: GameView }) {
  const [all, setAll] = useState(false);
  const why = frame.why;
  const event = frame.event;
  const explained = game.seats.filter((seat) => seat.explains).map((seat) => seat.seat);
  if (explained.length === 0) return null;
  if (!why || !event) {
    return (
      <section className="panel panel--why">
        <h2 className="panel-title">Why</h2>
        <p className="muted small">
          Step to a move by {explained.map((seat) => seatName(game, seat)).join(" or ")} to see what its nets
          predicted for each option.
        </p>
      </section>
    );
  }
  const options = all ? why.options : why.options.slice(0, 6);
  const values = why.options.map((option) => option.points);
  const low = Math.min(...values, 0);
  const high = Math.max(...values, 0);
  const span = high - low || 1;
  return (
    <section className="panel panel--why">
      <h2 className="panel-title">
        Why {seatName(game, event.seat)} chose <ActionText action={event.action} />
      </h2>
      <p className="why__net">
        <span className="badge badge--gold">{why.net}</span>
        {why.baseline !== null && (
          <span className="small">
            position worth <b className="mono">{signed(Math.round(why.baseline * 10) / 10)}</b>
          </span>
        )}
      </p>
      {why.note && <p className="small muted">{why.note}</p>}
      {options.length > 0 && (
        <ol className="why">
          {options.map((option) => {
            const left = ((Math.min(option.points, 0) - low) / span) * 100;
            const width = (Math.abs(option.points) / span) * 100;
            return (
              <li key={option.action.id} className={`why__row ${option.chosen ? "is-chosen" : ""}`}>
                <span className="why__label">
                  <ActionText action={option.action} />
                </span>
                <span className="why__bar" aria-hidden>
                  <span className="why__zero" style={{ left: `${((0 - low) / span) * 100}%` }} />
                  <span
                    className={`why__fill ${option.points >= 0 ? "is-plus" : "is-minus"}`}
                    style={{ left: `${left}%`, width: `${Math.max(width, 0.8)}%` }}
                  />
                </span>
                <span className="why__points mono">{signed(Math.round(option.points * 10) / 10)}</span>
              </li>
            );
          })}
        </ol>
      )}
      {why.options.length > 6 && (
        <button type="button" className="link small" onClick={() => setAll(!all)}>
          {all ? "show top 6" : `show all ${why.options.length} options`}
        </button>
      )}
      <p className="footnote">
        Predicted round reward in points: the bot&apos;s own score change minus a weighted average of the
        opponents&apos;, at x1.
        {why.net === "CardPlayNet" && " For a card, the position's worth plus that card's advantage."}
      </p>
    </section>
  );
}
