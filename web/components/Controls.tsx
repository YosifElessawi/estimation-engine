"use client";

import { useState } from "react";

import { SANS, suitOf, trumpName } from "@/lib/cards";
import { positionOf, seatName } from "@/lib/seats";
import type { ActionView, Frame, GameView, SeekTarget, TableView } from "@/lib/types";

import { TrumpGlyph } from "./ActionText";

export type Speed = "slow" | "normal" | "fast" | "turbo";
export const SPEED_FACTORS: Record<Speed, number> = { slow: 1.7, normal: 1, fast: 0.45, turbo: 0.12 };

const TRUMPS = [0, 1, 2, 3, SANS];
const SIMPLE = ["no_dash", "dash", "keep", "stay", "with", "pass"];

const WHERE = { bottom: "", left: "on your left", top: "across", right: "on your right" } as const;

/** The viewer's own call in this auction, when it is a bid or a With (AUC-2: they hold it). */
function heldCall(table: TableView): ActionView | null {
  const me = table.current_player;
  const call = me !== null ? (table.calls[me] ?? null) : null;
  if (!call || call.tricks === null || call.trump === null) return null;
  return call.kind === "bid" || call.kind === "with" || call.kind === "super_call" ? call : null;
}

function Bid({ tricks, trump }: { tricks: number; trump: number }) {
  return (
    <>
      {tricks} <TrumpGlyph trump={trump} />
    </>
  );
}

function simpleLabel(action: ActionView, table: TableView) {
  switch (action.kind) {
    case "no_dash":
      return "No dash";
    case "dash":
      return "Dash call";
    case "keep":
      return "Keep my bid";
    case "stay":
      return `Stay at ${action.tricks ?? ""}`;
    case "with":
      return action.tricks !== null && action.trump !== null ? (
        <>
          With <Bid tricks={action.tricks} trump={action.trump} />
        </>
      ) : (
        "With"
      );
    case "pass": {
      // AUC-2: passing keeps the call you hold
      const held = table.phase === "auction" ? heldCall(table) : null;
      return held && held.tricks !== null && held.trump !== null ? (
        <>
          Pass, keep <Bid tricks={held.tricks} trump={held.trump} />
        </>
      ) : (
        "Pass"
      );
    }
    default:
      return action.label;
  }
}

function hintOf(legal: ActionView[], table: TableView): string {
  const kinds = new Set(legal.map((action) => action.kind));
  if (kinds.has("dash") || kinds.has("no_dash")) return "A Dash call promises to take no tricks at all.";
  if (kinds.has("estimate")) {
    const numbers = legal.filter((action) => action.kind === "estimate").map((action) => action.tricks ?? 0);
    const most = Math.max(...numbers);
    const me = table.current_player;
    // EST-5: the last estimator may not bring the total to exactly 13
    let last = "";
    if (me !== null && table.last_estimator === me) {
      const others = table.estimates.reduce<number>((total, e, seat) => (seat === me ? total : total + (e ?? 0)), 0);
      const banned = 13 - others;
      if (banned >= 0 && banned <= 13) last = ` You estimate last: ${banned} would make the total 13, so it's out.`;
    }
    // EST-2: a With player who chooses picks from its own call up to the Caller's number, first
    if (me !== null && table.caller !== me && table.with_players.some((w) => w.seat === me)) {
      const held = table.with_players.find((w) => w.seat === me)?.w ?? Math.min(...numbers);
      return `You're With: pick ${held} to ${most}.${last}`;
    }
    // EST-3: nobody estimates above the Caller's number
    const cap = table.caller !== null && most < 13 ? ` Up to ${most}, the Caller's number.` : "";
    return last ? `${last.trim()}${cap}` : `How many tricks will you take?${cap}`;
  }
  const stay = legal.find((action) => action.kind === "stay");
  if (stay) return `The bid you went With was raised. Join it, stay at ${stay.tricks ?? ""}, or outbid.`;
  if (kinds.has("keep")) return "Last word on your bid: keep it or raise it.";
  if (table.top_bid) return `Beat the top bid${kinds.has("with") ? ", go With it," : ""} or pass.`;
  return "Pick how many tricks and a trump, or pass.";
}

function titleOf(legal: ActionView[]): string {
  const kinds = new Set(legal.map((action) => action.kind));
  if (kinds.has("dash") || kinds.has("no_dash")) return "Dash call?";
  if (kinds.has("estimate")) return "Your estimate";
  return "Your bid";
}

/** The bid to beat during an auction, who holds it and what the viewer holds, at the top of the
 * bidding box. */
function Standing({ table, game }: { table: TableView; game: GameView }) {
  const top = table.top_bid;
  const holder = table.top_bidder;
  if (!top || holder === null) return null;
  const me = table.current_player;
  const held = heldCall(table);
  return (
    <div className="standing">
      <span className="standing__label">Top bid</span>
      <span className="standing__bid">
        <Bid tricks={top.tricks} trump={top.trump} />
        {top.trump !== SANS && <span className="standing__trump">{trumpName(top.trump)}</span>}
      </span>
      <span className="standing__who">
        {holder === me ? "is yours" : `by ${seatName(game, holder)}`}
        {holder !== me && <span className="standing__where"> {WHERE[positionOf(holder, game.human_seat ?? 0)]}</span>}
      </span>
      {held && held.tricks !== null && held.trump !== null && holder !== me && (
        <span className="standing__mine">
          {held.kind === "with" ? "You're With at" : "You hold"} <Bid tricks={held.tricks} trump={held.trump} />
        </span>
      )}
    </div>
  );
}

/** The viewer's non-card decisions, docked above their hand: Dash Call, bids, With, Pass,
 * estimates and super calls. During an auction the bid to beat sits on top; bids are picked in
 * two short rows, tricks then trump, and the button says the exact bid. */
export function DecisionPanel({
  legal,
  table,
  game,
  busy,
  onAct,
}: {
  legal: ActionView[];
  table: TableView;
  game: GameView;
  busy: boolean;
  onAct: (action: number) => void;
}) {
  const [superMode, setSuperMode] = useState(false);
  const [tricks, setTricks] = useState<number | null>(null);
  const [trump, setTrump] = useState<number | null>(null);

  const bids = legal.filter((action) => action.kind === "bid");
  const supers = legal.filter((action) => action.kind === "super_call");
  const estimates = legal.filter((action) => action.kind === "estimate");
  const simple = SIMPLE.flatMap((kind) => legal.filter((action) => action.kind === kind));
  const pickable = superMode || bids.length === 0 ? supers : bids;
  const picking = pickable.length > 0 && (estimates.length === 0 || superMode);

  const counts = [...new Set(pickable.map((action) => action.tricks ?? 0))].sort((a, b) => a - b);
  const find = (k: number | null, t: number | null) =>
    pickable.find((action) => action.tricks === k && action.trump === t) ?? null;
  const trickOk = (k: number) => pickable.some((action) => action.tricks === k && (trump === null || action.trump === trump));
  const trumpOk = (t: number) => pickable.some((action) => action.trump === t && (tricks === null || action.tricks === tricks));
  const chosen = find(tricks, trump);

  const pickTricks = (k: number) => {
    setTricks(k);
    if (trump !== null && !find(k, trump)) setTrump(null);
  };
  const pickTrump = (t: number) => {
    setTrump(t);
    if (tricks !== null && !find(tricks, t)) setTricks(null);
  };

  return (
    <section className="decide" aria-label={titleOf(legal)}>
      <header className="decide__head">
        <h2>{superMode ? "Super call" : titleOf(legal)}</h2>
        <p>{superMode ? "A super call names more tricks in a trump of your choice." : hintOf(legal, table)}</p>
      </header>

      {table.phase === "auction" && <Standing table={table} game={game} />}

      {estimates.length > 0 && !superMode && (
        <div className="decide__row" role="group" aria-label="Estimate">
          <div className="chips chips--wrap">
            {Array.from({ length: 14 }, (_, k) => {
              const action = estimates.find((a) => a.tricks === k);
              return (
                <button
                  key={k}
                  type="button"
                  className="chip-btn"
                  disabled={!action || busy}
                  onClick={() => action && onAct(action.id)}
                  aria-label={action ? `Estimate ${k}` : `${k} is not allowed`}
                >
                  {k}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {picking && (
        <>
          <div className="decide__row" role="group" aria-label="Tricks">
            <span className="decide__label">Tricks</span>
            <div className="chips">
              {counts.map((k) => (
                <button
                  key={k}
                  type="button"
                  className={`chip-btn ${tricks === k ? "is-on" : ""}`}
                  aria-pressed={tricks === k}
                  disabled={busy || !trickOk(k)}
                  onClick={() => pickTricks(k)}
                >
                  {k}
                </button>
              ))}
            </div>
          </div>
          <div className="decide__row" role="group" aria-label="Trump">
            <span className="decide__label">Trump</span>
            <div className="chips">
              {TRUMPS.filter((t) => pickable.some((action) => action.trump === t)).map((t) => (
                <button
                  key={t}
                  type="button"
                  className={`chip-btn chip-btn--suit ${trump === t ? "is-on" : ""}`}
                  aria-pressed={trump === t}
                  aria-label={trumpName(t)}
                  disabled={busy || !trumpOk(t)}
                  onClick={() => pickTrump(t)}
                >
                  <TrumpGlyph trump={t} />
                </button>
              ))}
            </div>
          </div>
        </>
      )}

      <div className="decide__actions">
        {simple.map((action) => (
          <button
            key={action.id}
            type="button"
            className={`btn ${action.kind === "no_dash" || action.kind === "keep" ? "btn--primary" : "btn--ghost"}`}
            disabled={busy}
            onClick={() => onAct(action.id)}
          >
            {simpleLabel(action, table)}
          </button>
        ))}
        {supers.length > 0 && (bids.length > 0 || estimates.length > 0) && (
          <button
            type="button"
            className="btn btn--quiet"
            onClick={() => {
              setSuperMode(!superMode);
              setTricks(null);
              setTrump(null);
            }}
          >
            {superMode ? "Back" : "Super call…"}
          </button>
        )}
        {picking && (
          <button
            type="button"
            className="btn btn--primary btn--push"
            disabled={busy || !chosen}
            onClick={() => chosen && onAct(chosen.id)}
          >
            {chosen && chosen.trump !== null && chosen.tricks !== null ? (
              <>
                {superMode ? "Super call" : "Bid"} <Bid tricks={chosen.tricks} trump={chosen.trump} />
              </>
            ) : tricks === null ? (
              "Pick tricks"
            ) : (
              "Pick a trump"
            )}
          </button>
        )}
      </div>
    </section>
  );
}

/** One line above the viewer's hand when there is nothing to decide: whose turn it is, what the
 * viewer may play, or that they are looking back at an earlier move. */
export function StatusPill({
  frame,
  live,
  animating,
  busy,
  turnName,
  onLive,
}: {
  frame: Frame;
  live: boolean;
  animating: boolean;
  busy: boolean;
  turnName: string | null;
  onLive: () => void;
}) {
  if (!live) {
    return (
      <div className="pill pill--back">
        Looking back at an earlier move
        <button type="button" className="pill__action" onClick={onLive}>
          Back to the table
        </button>
      </div>
    );
  }
  if (frame.kind === "round_end" || frame.kind === "game_end") return null;
  const legal = frame.legal;
  if (legal?.some((action) => action.kind === "play")) {
    const led = frame.table.trick[0];
    const ledSuit = led ? suitOf(led.card) : null;
    const following =
      ledSuit !== null && legal.every((action) => action.card !== null && suitOf(action.card) === ledSuit);
    return (
      <div className="pill pill--turn">
        <b>Your turn</b>
        <span className="pill__dot" aria-hidden />
        {ledSuit === null ? (
          "you lead, any card"
        ) : following ? (
          <>
            follow <TrumpGlyph trump={ledSuit} />
          </>
        ) : (
          <>
            no <TrumpGlyph trump={ledSuit} /> left, any card
          </>
        )}
      </div>
    );
  }
  if (busy) return <div className="pill">Sending…</div>;
  if (animating && turnName) {
    return (
      <div className="pill pill--wait">
        {turnName} {turnName === "You" ? "are" : "is"} thinking
        <span className="dots" aria-hidden />
        <button type="button" className="pill__action" onClick={onLive}>
          Skip to my turn
        </button>
      </div>
    );
  }
  return null;
}

function Icon({ name }: { name: "first" | "prev-trick" | "back" | "play" | "pause" | "forward" | "next-trick" | "result" | "next" | "end" }) {
  const paths: Record<typeof name, string> = {
    first: "M6 5v14M18 6l-9 6 9 6V6z",
    "prev-trick": "M12 6l-7 6 7 6M20 6l-7 6 7 6",
    back: "M15 6l-6 6 6 6",
    play: "M8 5l11 7-11 7V5z",
    pause: "M8 5v14M16 5v14",
    forward: "M9 6l6 6-6 6",
    "next-trick": "M4 6l7 6-7 6M12 6l7 6-7 6",
    result: "M5 6l9 6-9 6V6zM19 5v14",
    next: "M4 12h13M13 6l6 6-6 6",
    end: "M5 6l7 6-7 6M12 6l7 6-7 6M20 5v14",
  };
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
      <path d={paths[name]} />
    </svg>
  );
}

export function SpeedSelect({ speed, onSpeed }: { speed: Speed; onSpeed: (speed: Speed) => void }) {
  return (
    <label className="speed">
      <span>Pace</span>
      <select value={speed} onChange={(e) => onSpeed(e.target.value as Speed)}>
        <option value="slow">Slow</option>
        <option value="normal">Normal</option>
        <option value="fast">Fast</option>
        <option value="turbo">Turbo</option>
      </select>
    </label>
  );
}

interface WatchProps {
  index: number;
  known: number;
  finished: boolean;
  autoplay: boolean;
  speed: Speed;
  busy: boolean;
  onStep: (delta: number) => void;
  onGoTo: (index: number) => void;
  onSeek: (direction: "next" | "prev", target: SeekTarget) => void;
  onAutoplay: (on: boolean) => void;
  onSpeed: (speed: Speed) => void;
}

/** Playback for a watched or reviewed game: step, jump by trick or round, play, scrub. */
export function WatchBar(props: WatchProps) {
  const { index, known, finished, autoplay, speed, busy } = props;
  const atStart = index === 0;
  const atEnd = finished && index >= known - 1;
  return (
    <div className="watchbar">
      <div className="watchbar__buttons">
        <button type="button" className="tbtn" title="Start of this round, or the round before" onClick={() => props.onSeek("prev", "deal")} disabled={atStart}>
          <Icon name="first" />
        </button>
        <button type="button" className="tbtn" title="Previous trick (Shift + ←)" onClick={() => props.onSeek("prev", "trick")} disabled={atStart}>
          <Icon name="prev-trick" />
        </button>
        <button type="button" className="tbtn" title="Step back (←)" onClick={() => props.onStep(-1)} disabled={atStart}>
          <Icon name="back" />
        </button>
        <button
          type="button"
          className={`tbtn tbtn--play ${autoplay ? "is-on" : ""}`}
          title="Play or pause (Space)"
          onClick={() => props.onAutoplay(!autoplay)}
          disabled={atEnd}
        >
          <Icon name={autoplay ? "pause" : "play"} />
        </button>
        <button type="button" className="tbtn" title="Step forward (→)" onClick={() => props.onStep(1)} disabled={atEnd}>
          <Icon name="forward" />
        </button>
        <button type="button" className="tbtn" title="Next trick (Shift + →)" onClick={() => props.onSeek("next", "trick")} disabled={atEnd}>
          <Icon name="next-trick" />
        </button>
        <button type="button" className="tbtn tbtn--text" title="This round's result" onClick={() => props.onSeek("next", "round_end")} disabled={atEnd}>
          Result
        </button>
        <button type="button" className="tbtn tbtn--text" title="Next deal" onClick={() => props.onSeek("next", "deal")} disabled={atEnd}>
          Next round
        </button>
        <button type="button" className="tbtn tbtn--text" title="Play the game to its end" onClick={() => props.onSeek("next", "end")} disabled={atEnd}>
          End
        </button>
      </div>
      <div className="watchbar__track">
        <input
          type="range"
          className="scrub"
          min={0}
          max={Math.max(known - 1, 0)}
          value={Math.min(index, Math.max(known - 1, 0))}
          onChange={(e) => props.onGoTo(Number(e.target.value))}
          aria-label="Move"
        />
        <span className="watchbar__count">
          {index + 1} / {finished ? known : `${known}…`}
        </span>
        <SpeedSelect speed={speed} onSpeed={props.onSpeed} />
        {busy && <span className="spinner" aria-label="Thinking" />}
      </div>
    </div>
  );
}
