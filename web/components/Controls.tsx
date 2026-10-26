"use client";

import { useState } from "react";

import { suitOf, suitSymbol } from "@/lib/cards";
import type { ActionView, Frame, SeekTarget } from "@/lib/types";

import { ActionText, TrumpGlyph } from "./ActionText";

export type Speed = "slow" | "normal" | "fast" | "turbo";
export const SPEED_FACTORS: Record<Speed, number> = { slow: 1.7, normal: 1, fast: 0.45, turbo: 0.12 };

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

export function WatchControls(props: WatchProps) {
  const { index, known, finished, autoplay, speed, busy } = props;
  const atEnd = finished && index >= known - 1;
  return (
    <div className="transport">
      <div className="transport__row">
        <button type="button" className="tbtn" title="Start of this round, or the round before" onClick={() => props.onSeek("prev", "deal")} disabled={index === 0}>
          ⏮ <span>Deal</span>
        </button>
        <button type="button" className="tbtn" title="Previous finished trick (Shift + ←)" onClick={() => props.onSeek("prev", "trick")} disabled={index === 0}>
          « <span>Trick</span>
        </button>
        <button type="button" className="tbtn tbtn--step" title="Step back (←)" onClick={() => props.onStep(-1)} disabled={index === 0}>
          ‹
        </button>
        <button
          type="button"
          className={`tbtn tbtn--play ${autoplay ? "is-on" : ""}`}
          title="Play / pause (Space)"
          onClick={() => props.onAutoplay(!autoplay)}
          disabled={atEnd}
        >
          {autoplay ? "❚❚" : "▶"}
        </button>
        <button type="button" className="tbtn tbtn--step" title="Step forward (→)" onClick={() => props.onStep(1)} disabled={atEnd}>
          ›
        </button>
        <button type="button" className="tbtn" title="Next finished trick (Shift + →)" onClick={() => props.onSeek("next", "trick")} disabled={atEnd}>
          <span>Trick</span> »
        </button>
        <button type="button" className="tbtn" title="This round's result" onClick={() => props.onSeek("next", "round_end")} disabled={atEnd}>
          <span>Result</span> ⏭
        </button>
        <button type="button" className="tbtn" title="Next deal" onClick={() => props.onSeek("next", "deal")} disabled={atEnd}>
          <span>Next round</span>
        </button>
        <button type="button" className="tbtn" title="Play the game to its end" onClick={() => props.onSeek("next", "end")} disabled={atEnd}>
          <span>End</span> ⇥
        </button>
      </div>
      <div className="transport__row transport__row--meta">
        <input
          type="range"
          className="scrub"
          min={0}
          max={Math.max(known - 1, 0)}
          value={Math.min(index, Math.max(known - 1, 0))}
          onChange={(e) => props.onGoTo(Number(e.target.value))}
          aria-label="Step"
        />
        <span className="transport__count mono">
          {index + 1} / {finished ? known : `${known}…`}
        </span>
        <SpeedSelect speed={speed} onSpeed={props.onSpeed} />
        {busy && <span className="spinner" aria-label="Thinking" />}
      </div>
    </div>
  );
}

const TRUMP_COLUMNS = [0, 1, 2, 3, 4];

function BidGrid({ bids, onAct }: { bids: ActionView[]; onAct: (action: number) => void }) {
  const rows = [...new Set(bids.map((bid) => bid.tricks ?? 0))].sort((a, b) => a - b);
  const byKey = new Map(bids.map((bid) => [`${bid.tricks}-${bid.trump}`, bid]));
  return (
    <div className="bidgrid" role="grid">
      <div className="bidgrid__head" role="row">
        <span />
        {TRUMP_COLUMNS.map((trump) => (
          <span key={trump} className="bidgrid__col">
            <TrumpGlyph trump={trump} />
          </span>
        ))}
      </div>
      {rows.map((tricks) => (
        <div key={tricks} className="bidgrid__row" role="row">
          <span className="bidgrid__tricks mono">{tricks}</span>
          {TRUMP_COLUMNS.map((trump) => {
            const bid = byKey.get(`${tricks}-${trump}`);
            return (
              <button
                key={trump}
                type="button"
                className="bidgrid__cell"
                disabled={!bid}
                onClick={() => bid && onAct(bid.id)}
                title={bid?.label}
              >
                {tricks}
                <TrumpGlyph trump={trump} />
              </button>
            );
          })}
        </div>
      ))}
    </div>
  );
}

interface PlayProps {
  frame: Frame;
  live: boolean;
  animating: boolean;
  busy: boolean;
  turnName: string | null;
  speed: Speed;
  onSpeed: (speed: Speed) => void;
  onLive: () => void;
  onStep: (delta: number) => void;
  canStepBack: boolean;
  canStepForward: boolean;
}

function SpeedSelect({ speed, onSpeed }: { speed: Speed; onSpeed: (speed: Speed) => void }) {
  return (
    <label className="transport__speed">
      Speed
      <select className="select select--small" value={speed} onChange={(e) => onSpeed(e.target.value as Speed)}>
        <option value="slow">Slow</option>
        <option value="normal">Normal</option>
        <option value="fast">Fast</option>
        <option value="turbo">Turbo</option>
      </select>
    </label>
  );
}

/** The status line under the table in play mode: whose turn it is, what to do, and stepping
 * back through what already happened. */
export function PlayBar({ frame, live, animating, busy, turnName, speed, onSpeed, onLive, onStep, canStepBack, canStepForward }: PlayProps) {
  const legal = live ? frame.legal : null;
  const browse = (
    <span className="playbar__browse">
      <SpeedSelect speed={speed} onSpeed={onSpeed} />
      <button type="button" className="tbtn tbtn--step" onClick={() => onStep(-1)} disabled={!canStepBack} title="Look back (←)">
        ‹
      </button>
      <button type="button" className="tbtn tbtn--step" onClick={() => onStep(1)} disabled={!canStepForward} title="Forward (→)">
        ›
      </button>
    </span>
  );

  if (!live) {
    return (
      <div className="playbar">
        {browse}
        <span className="playbar__status">Looking back at step {frame.index + 1}.</span>
        <button type="button" className="button button--primary" onClick={onLive}>
          Back to the table
        </button>
      </div>
    );
  }
  if (!legal || legal.length === 0) {
    return (
      <div className="playbar">
        {browse}
        <span className="playbar__status">
          {frame.kind === "game_end" ? (
            "Game over."
          ) : animating ? (
            <>
              {turnName ? `${turnName} to act` : "Bots playing"}
              <span className="dots" aria-hidden />
            </>
          ) : busy ? (
            "Sending your move…"
          ) : (
            "Waiting"
          )}
        </span>
        {animating && frame.kind !== "round_end" && (
          <button type="button" className="link" onClick={onLive}>
            skip to my turn
          </button>
        )}
      </div>
    );
  }
  if (legal.some((action) => action.kind === "play")) {
    const led = frame.table.trick[0];
    const ledSuit = led ? suitOf(led.card) : null;
    const following = ledSuit !== null && legal.every((action) => action.card !== null && suitOf(action.card) === ledSuit);
    return (
      <div className="playbar playbar--turn">
        {browse}
        <span className="playbar__status">
          <b>Your turn.</b>{" "}
          {led
            ? following
              ? `Follow suit: ${suitSymbol(ledSuit ?? 0)}.`
              : `You have no ${suitSymbol(ledSuit ?? 0)}: play any card.`
            : "You lead: play any card."}{" "}
          Click a card in your hand.
        </span>
      </div>
    );
  }
  return (
    <div className="playbar playbar--turn">
      {browse}
      <span className="playbar__status">
        <b>Your call.</b> Choose it in the panel on the right.
      </span>
    </div>
  );
}

/** The human's non-card decisions: Dash Call, bids, With, Pass, estimates and super calls. */
export function DecisionPanel({ legal, busy, onAct }: { legal: ActionView[]; busy: boolean; onAct: (action: number) => void }) {
  const [superOpen, setSuperOpen] = useState(false);
  const kinds = new Set(legal.map((action) => action.kind));
  const bids = legal.filter((action) => action.kind === "bid");
  const supers = legal.filter((action) => action.kind === "super_call");
  const estimates = legal.filter((action) => action.kind === "estimate");
  const others = legal.filter((action) => ["no_dash", "dash", "with", "pass", "keep", "stay"].includes(action.kind));
  const stay = legal.find((action) => action.kind === "stay");
  const legalEstimates = new Map(estimates.map((action) => [action.tricks, action]));
  return (
    <section className="panel panel--decide" aria-label="Your call">
      <h2 className="panel-title">Your call</h2>
      <p className="small muted decide__hint">
        {kinds.has("dash") || kinds.has("no_dash")
          ? "A Dash Call promises to take zero tricks."
          : estimates.length > 0
            ? "How many tricks will you take? Crossed-out numbers are not allowed here."
            : stay
              ? `The player you said With to raised. Join the new bid, stay at ${stay.tricks ?? ""}, or outbid.`
              : "Bid a number of tricks and a trump, say With, or pass."}
      </p>
      {others.length > 0 && (
        <div className="decide__row">
          {others.map((action) => (
            <button
              key={action.id}
              type="button"
              className={`button ${action.kind === "dash" ? "button--accent" : ""} ${["no_dash", "pass", "keep", "stay"].includes(action.kind) ? "button--primary" : ""}`}
              onClick={() => onAct(action.id)}
              disabled={busy}
            >
              <ActionText action={action} />
            </button>
          ))}
        </div>
      )}
      {estimates.length > 0 && (
        <div className="estimates" role="group" aria-label="Estimate">
          {Array.from({ length: 14 }, (_, k) => {
            const action = legalEstimates.get(k);
            return (
              <button
                key={k}
                type="button"
                className="estimates__chip mono"
                disabled={!action || busy}
                onClick={() => action && onAct(action.id)}
                title={action ? `Estimate ${k}` : `${k} is not allowed here`}
              >
                {k}
              </button>
            );
          })}
        </div>
      )}
      {bids.length > 0 && <BidGrid bids={bids} onAct={onAct} />}
      {supers.length > 0 && (
        <div className="supers">
          <button type="button" className="link small" onClick={() => setSuperOpen(!superOpen)}>
            {superOpen
              ? "hide super calls"
              : `or make a super call (${Math.min(...supers.map((action) => action.tricks ?? 13))}+ tricks in another trump)`}
          </button>
          {superOpen && <BidGrid bids={supers} onAct={onAct} />}
        </div>
      )}
    </section>
  );
}
