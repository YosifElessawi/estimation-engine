"use client";

import { useState } from "react";

import { formatElo, seatName, signed } from "@/lib/seats";
import type { Frame, GameView } from "@/lib/types";

import { ActionText, CardGlyph, TrumpGlyph } from "./ActionText";
import { Avatar } from "./Table";

export type DrawerTab = "scores" | "log" | "why";

const TITLES: Record<DrawerTab, string> = { scores: "Scores", log: "Log", why: "Why" };

/** Scores, the round's calls and the RL bots' reasons, opened from the top bar when wanted. */
export function Drawer({
  tab,
  tabs,
  frame,
  game,
  onTab,
  onClose,
}: {
  tab: DrawerTab;
  tabs: DrawerTab[];
  frame: Frame;
  game: GameView;
  onTab: (tab: DrawerTab) => void;
  onClose: () => void;
}) {
  return (
    <aside className="drawer" aria-label={TITLES[tab]}>
      <header className="drawer__head">
        <nav className="seg" aria-label="Details">
          {tabs.map((t) => (
            <button key={t} type="button" className={t === tab ? "is-on" : ""} aria-pressed={t === tab} onClick={() => onTab(t)}>
              {TITLES[t]}
            </button>
          ))}
        </nav>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
            <path d="M6 6l12 12M18 6L6 18" />
          </svg>
        </button>
      </header>
      <div className="drawer__body">
        {tab === "scores" && <Scores frame={frame} game={game} />}
        {tab === "log" && <Log frame={frame} game={game} />}
        {tab === "why" && <Why frame={frame} game={game} />}
      </div>
    </aside>
  );
}

const human = (game: GameView, seat: number) => game.seats[seat]?.human ?? false;

function Scores({ frame, game }: { frame: Frame; game: GameView }) {
  const best = Math.max(...frame.scores);
  const ranked = [0, 1, 2, 3].sort((a, b) => (frame.scores[b] ?? 0) - (frame.scores[a] ?? 0));
  return (
    <>
      <ol className="standings">
        {ranked.map((seat, place) => {
          const info = game.seats[seat];
          return (
            <li key={seat} className={frame.scores[seat] === best ? "is-leader" : ""}>
              <span className="standings__place">{place + 1}</span>
              <Avatar seat={seat} human={human(game, seat)} name={seatName(game, seat)} />
              <span className="standings__who">
                <span className="standings__name">{seatName(game, seat)}</span>
                {info && !info.human && <span className="standings__elo">Elo {formatElo(info.elo)}</span>}
              </span>
              <span className="standings__score">{frame.scores[seat]}</span>
            </li>
          );
        })}
      </ol>
      {frame.history.length > 0 && (
        <div className="rounds">
          <h3 className="drawer__title">Round by round</h3>
          <table>
            <thead>
              <tr>
                <th>Rd</th>
                <th>Trump</th>
                {[0, 1, 2, 3].map((seat) => (
                  <th key={seat} className="num" title={seatName(game, seat)}>
                    <Avatar seat={seat} human={human(game, seat)} name={seatName(game, seat)} />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...frame.history].reverse().map((round) => (
                <tr key={round.round_no}>
                  <td>{round.round_no}</td>
                  <td>
                    {round.skipped ? <span className="muted">skip</span> : round.trump !== null ? <TrumpGlyph trump={round.trump} /> : null}
                    {round.multiplier > 1 && <span className="rounds__mult">x{round.multiplier}</span>}
                  </td>
                  {round.scores.map((score, seat) => (
                    <td key={seat} className={`num ${score > 0 ? "is-plus" : score < 0 ? "is-minus" : "muted"}`}>
                      {score === 0 ? "0" : signed(score)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <p className="drawer__note">
        {game.seed === null ? "The seed stays hidden until the game ends." : <>Seed <span className="mono">{game.seed}</span>: the same seats and seed replay this game.</>}
      </p>
    </>
  );
}

function Log({ frame, game }: { frame: Frame; game: GameView }) {
  const events = frame.table.events;
  const last = frame.table.last_trick;
  return (
    <>
      <h3 className="drawer__title">Round {frame.table.round_no} calls</h3>
      {events.length === 0 ? (
        <p className="muted">No calls yet.</p>
      ) : (
        <ol className="log">
          {events.map((event, i) => (
            <li key={i} className={`log__row log__row--${event.action.kind}`}>
              <Avatar seat={event.seat} human={human(game, event.seat)} name={seatName(game, event.seat)} />
              <span className="log__who">{seatName(game, event.seat)}</span>
              <span className="log__what">
                <ActionText action={event.action} />
              </span>
            </li>
          ))}
        </ol>
      )}
      {last && (
        <>
          <h3 className="drawer__title">Last trick</h3>
          <ol className="log">
            {last.cards.map(({ seat, card }) => (
              <li key={card} className={`log__row ${seat === last.winner ? "is-winner" : ""}`}>
                <Avatar seat={seat} human={human(game, seat)} name={seatName(game, seat)} />
                <span className="log__who">{seatName(game, seat)}</span>
                <span className="log__what">
                  <CardGlyph card={card} />
                  {seat === last.winner && <span className="log__won">took it</span>}
                </span>
              </li>
            ))}
          </ol>
        </>
      )}
    </>
  );
}

function Why({ frame, game }: { frame: Frame; game: GameView }) {
  const [all, setAll] = useState(false);
  const why = frame.why;
  const event = frame.event;
  const explained = game.seats.filter((seat) => seat.explains).map((seat) => seat.seat);
  if (!why || !event) {
    return (
      <p className="muted">
        Step to a move by {explained.map((seat) => seatName(game, seat)).join(" or ")} to see what its nets predicted for
        each option.
      </p>
    );
  }
  const options = all ? why.options : why.options.slice(0, 6);
  const values = why.options.map((option) => option.points);
  const low = Math.min(...values, 0);
  const high = Math.max(...values, 0);
  const span = high - low || 1;
  return (
    <>
      <h3 className="drawer__title">
        Why {seatName(game, event.seat)} chose <ActionText action={event.action} />
      </h3>
      <p className="why__net">
        <span className="tag tag--accent">{why.net}</span>
        {why.baseline !== null && (
          <span>
            position worth <b>{signed(Math.round(why.baseline * 10) / 10)}</b>
          </span>
        )}
      </p>
      {why.note && <p className="muted">{why.note}</p>}
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
                <span className="why__points">{signed(Math.round(option.points * 10) / 10)}</span>
              </li>
            );
          })}
        </ol>
      )}
      {why.options.length > 6 && (
        <button type="button" className="link" onClick={() => setAll(!all)}>
          {all ? "Show the top 6" : `Show all ${why.options.length} options`}
        </button>
      )}
      <p className="drawer__note">
        Predicted round reward in points: the bot&apos;s own score change minus a weighted average of the
        opponents&apos;, at x1.
        {why.net === "CardPlayNet" && " For a card, the position's worth plus that card's advantage."}
      </p>
    </>
  );
}
