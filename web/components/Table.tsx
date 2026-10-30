"use client";

import { type CSSProperties, type ReactNode, useState } from "react";

import { SANS, bySuit, sortHand, suitOf, trickLeader } from "@/lib/cards";
import { type Position, formatElo, positionOf, seatName } from "@/lib/seats";
import type { Frame, GameView, TableView } from "@/lib/types";

import { CallText, TrumpGlyph } from "./ActionText";
import { Backs, PlayingCard } from "./PlayingCard";

// The public build (NEXT_PUBLIC_EDITION=public, see PublicApp) leaves out what only the full table
// has. Checked in this module so the build folds it away.
const PUBLIC = process.env.NEXT_PUBLIC_EDITION === "public";

interface TableProps {
  frame: Frame;
  game: GameView;
  bottom: number;
  /** Cards the viewer may play now; empty when it is not their turn to play. */
  playableCards: Set<number>;
  /** Show an empty spot in the trick where the viewer's card will land. */
  awaitingMyCard: boolean;
  onPlayCard: (card: number) => void;
  /** The viewer's decision panel or a status line, docked above the hand. */
  dock: ReactNode;
  overlay: ReactNode;
}

const SEATS = [0, 1, 2, 3];
const SLOT: Record<Position, string> = { bottom: "s", right: "e", top: "n", left: "w" };

const trumpSuitOf = (table: TableView): number | null =>
  table.trump !== null && table.trump !== SANS ? table.trump : null;

export function Table({ frame, game, bottom, playableCards, awaitingMyCard, onPlayCard, dock, overlay }: TableProps) {
  const spectator = game.mode !== "play";
  const table = frame.table;
  const hand = table.hands[bottom] ?? null;
  return (
    <div className={`stage stage--${table.phase} ${spectator ? "stage--watch" : "stage--play"}`}>
      {SEATS.map((seat) => (
        <Seat key={seat} seat={seat} position={positionOf(seat, bottom)} frame={frame} game={game} spectator={spectator} />
      ))}
      <Center frame={frame} game={game} bottom={bottom} awaitingMyCard={awaitingMyCard} />
      {hand && (
        <Hand
          cards={hand}
          playable={playableCards}
          interactive={!spectator}
          trumpSuit={trumpSuitOf(table)}
          onPlay={onPlayCard}
        />
      )}
      <div className="dock">{dock}</div>
      {overlay}
    </div>
  );
}

interface Tag {
  text: string;
  tone: "accent" | "soft" | "teal" | "rose" | "muted";
}

function tagsOf(table: TableView, seat: number): Tag[] {
  const tags: Tag[] = [];
  if (table.caller === seat) tags.push({ text: table.super_call ? "Super call" : "Caller", tone: "accent" });
  const held = table.with_players.find((w) => w.seat === seat);
  if (held) tags.push({ text: table.phase === "auction" ? `With ${held.w}` : "With", tone: "soft" });
  if (table.dash_callers.includes(seat)) tags.push({ text: "Dash", tone: "teal" });
  if (table.last_estimator === seat) {
    const risks = table.risks ?? 0;
    if (risks > 0) tags.push({ text: risks > 1 ? `${risks} Risks` : "Risk", tone: "rose" });
    else if (table.phase === "estimation") tags.push({ text: "Last", tone: "muted" });
  }
  // TBL-7: announced at the deal; it says a suit is missing, never which one
  if (table.void_announced.includes(seat)) tags.push({ text: "Void", tone: "muted" });
  return tags;
}

function BotIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" aria-hidden>
      <rect x="5" y="8" width="14" height="11" rx="3.5" />
      <path d="M12 8V5" />
      <circle cx="12" cy="4" r="1.1" fill="currentColor" />
      <circle cx="9.6" cy="13.4" r="1.2" fill="currentColor" stroke="none" />
      <circle cx="14.4" cy="13.4" r="1.2" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function Avatar({ seat, human, name, turn }: { seat: number; human: boolean; name: string; turn?: boolean }) {
  return (
    <span className={`avatar avatar--s${seat} ${human ? "avatar--human" : ""}`} aria-hidden>
      {turn && <span className="avatar__ring" />}
      {human ? name.slice(0, 1).toUpperCase() : <BotIcon />}
    </span>
  );
}

function Tally({ estimate, won }: { estimate: number | null | undefined; won: number }) {
  if (estimate === null || estimate === undefined) return null;
  const dots = Math.max(estimate, won);
  return (
    <span className={`tally ${dots > 8 ? "tally--many" : ""}`} aria-label={`${won} of ${estimate} tricks`}>
      {dots > 0 && (
        <span className="tally__dots" aria-hidden>
          {Array.from({ length: dots }, (_, i) => (
            <i key={i} className={i < won ? (i < estimate ? "is-won" : "is-over") : ""} />
          ))}
        </span>
      )}
      <span className="tally__text">
        {won}/{estimate}
      </span>
    </span>
  );
}

function Seat({
  seat,
  position,
  frame,
  game,
  spectator,
}: {
  seat: number;
  position: Position;
  frame: Frame;
  game: GameView;
  spectator: boolean;
}) {
  const table = frame.table;
  const info = game.seats[seat];
  const over = frame.kind === "round_end" || frame.kind === "game_end";
  const turn = !over && table.current_player === seat;
  const out = table.phase === "auction" && table.passed.includes(seat);
  const call = table.calls[seat] ?? null;
  const showCall = call !== null && ["dash", "auction", "estimation"].includes(table.phase);
  const fresh = frame.event?.seat === seat && frame.event.action.kind !== "play";
  const high = table.phase === "auction" && table.top_bidder === seat && call?.kind === "bid";
  const hand = table.hands[seat] ?? null;
  const name = seatName(game, seat);
  const human = info?.human ?? false;
  const isBottom = position === "bottom";
  const tags = tagsOf(table, seat);
  const won = table.tricks_won[seat] ?? 0;
  const estimate = table.estimates[seat];
  const showWon = estimate === null || estimate === undefined ? table.phase === "play" || table.phase === "done" : false;
  return (
    <section
      className={`seat seat--${position} ${turn ? "is-turn" : ""} ${out ? "is-out" : ""}`}
      aria-label={name}
    >
      <div className="plate" title={!PUBLIC && info && !human ? `${info.label} · Elo ${formatElo(info.elo)}` : undefined}>
        <Avatar seat={seat} human={human} name={name} turn={turn} />
        <div className="plate__info">
          <div className="plate__name">
            <span className="plate__label">{name}</span>
            {tags.map((tag) => (
              <span key={tag.text} className={`tag tag--${tag.tone}`}>
                {tag.text}
              </span>
            ))}
          </div>
          <div className="plate__sub">
            <span className="plate__points">{frame.scores[seat] ?? 0} pts</span>
            <Tally estimate={estimate} won={won} />
            {showWon && <span className="tally__text">won {won}</span>}
          </div>
        </div>
        {!spectator && !isBottom && <Backs count={table.hand_sizes[seat] ?? 0} />}
        {showCall && call && (
          <div key={fresh ? `call-${frame.index}` : "call"} className={`bubble ${fresh ? "is-fresh" : ""} ${high ? "is-high" : ""} ${call.kind === "pass" ? "is-pass" : ""}`}>
            <CallText action={call} />
          </div>
        )}
      </div>
      {spectator && !isBottom && hand && <MiniHand cards={hand} trumpSuit={trumpSuitOf(table)} />}
    </section>
  );
}

function MiniHand({ cards, trumpSuit }: { cards: number[]; trumpSuit: number | null }) {
  return (
    <div className="minihand" aria-label={`${cards.length} cards`}>
      {bySuit(cards)
        .filter((group) => group.cards.length > 0)
        .map((group) => (
          <span key={group.suit} className="minihand__suit">
            {group.cards.map((card) => (
              <PlayingCard key={card} card={card} size="mini" trump={group.suit === trumpSuit} />
            ))}
          </span>
        ))}
    </div>
  );
}

function Hand({
  cards,
  playable,
  interactive,
  trumpSuit,
  onPlay,
}: {
  cards: number[];
  playable: Set<number>;
  interactive: boolean;
  trumpSuit: number | null;
  onPlay: (card: number) => void;
}) {
  const sorted = sortHand(cards);
  const n = sorted.length;
  const choosing = interactive && playable.size > 0;
  return (
    <div className={`hand ${choosing ? "is-choosing" : ""}`} style={{ "--n": n } as CSSProperties} aria-label="Your hand">
      {sorted.map((card, i) => {
        const t = n > 1 ? i / (n - 1) - 0.5 : 0;
        const canPlay = choosing && playable.has(card);
        return (
          <PlayingCard
            key={card}
            card={card}
            size="lg"
            playable={canPlay}
            dimmed={choosing && !canPlay}
            trump={suitOf(card) === trumpSuit}
            style={{ "--t": t.toFixed(3) } as CSSProperties}
            onPlay={onPlay}
          />
        );
      })}
    </div>
  );
}

function Center({
  frame,
  game,
  bottom,
  awaitingMyCard,
}: {
  frame: Frame;
  game: GameView;
  bottom: number;
  awaitingMyCard: boolean;
}) {
  const table = frame.table;
  if (frame.kind === "deal") {
    return (
      <div className="center center--deal" key={`deal-${frame.index}`}>
        <span className="center__eyebrow">New deal</span>
        <span className="center__big">Round {table.round_no}</span>
        {table.fixed_trump !== null && (
          <span className="center__sub">
            Fixed trump <TrumpGlyph trump={table.fixed_trump} />
          </span>
        )}
      </div>
    );
  }
  if (table.phase === "play" || table.phase === "done") {
    return <Trick key={table.tricks_played} frame={frame} game={game} bottom={bottom} awaitingMyCard={awaitingMyCard} />;
  }
  if (table.phase === "auction") {
    if (table.top_bid && table.top_bidder !== null) {
      return (
        <div className="center">
          <span className="center__eyebrow">Highest bid</span>
          <span className="center__big">
            {table.top_bid.tricks} <TrumpGlyph trump={table.top_bid.trump} />
          </span>
          <span className="center__sub">{seatName(game, table.top_bidder)}</span>
        </div>
      );
    }
    return (
      <div className="center">
        <span className="center__eyebrow">Bidding</span>
        <span className="center__sub">No bids yet</span>
      </div>
    );
  }
  if (table.phase === "estimation") {
    const sum = table.estimates.reduce<number>((total, e) => total + (e ?? 0), 0);
    const caller = table.caller;
    return (
      <div className="center">
        <span className="center__eyebrow">Estimates so far</span>
        <span className="center__big">
          {sum}
          <small> of 13</small>
        </span>
        {caller !== null && table.trump !== null && (
          <span className="center__sub">
            {seatName(game, caller)} called {table.estimates[caller] ?? table.top_bid?.tricks}{" "}
            <TrumpGlyph trump={table.trump} />
          </span>
        )}
      </div>
    );
  }
  return (
    <div className="center">
      <span className="center__eyebrow">Dash calls</span>
      <span className="center__sub">Who promises to take no tricks?</span>
    </div>
  );
}

/** The cards on the table. While a trick is under way, "Last trick" shows the one before it
 * (the only earlier trick the public table lets a player see again). */
function Trick({
  frame,
  game,
  bottom,
  awaitingMyCard,
}: {
  frame: Frame;
  game: GameView;
  bottom: number;
  awaitingMyCard: boolean;
}) {
  const [peek, setPeek] = useState(false);
  const table = frame.table;
  const event = frame.event;
  const last = table.last_trick;
  const justTaken = table.trick.length === 0 && event?.trick_winner !== null && event?.trick_winner !== undefined;
  const canPeek = last !== null && !justTaken && table.phase === "play";
  const peeking = peek && canPeek;
  let cards = table.trick;
  let winner: number | null = null;
  if (last && (justTaken || peeking)) {
    cards = last.cards;
    winner = last.winner;
  }
  const done = winner !== null;
  const leading = done ? winner : trickLeader(cards, table.trump);
  const fresh = !peeking && event?.action.kind === "play" ? event.action.card : null;
  const winnerName = winner !== null ? seatName(game, winner) : null;
  const you = winnerName === "You";
  return (
    <div className={`trick ${peeking ? "is-peeking" : ""}`} aria-label={peeking ? "The last trick" : "Cards on the table"}>
      {awaitingMyCard && !done && <div className="trick__slot trick__slot--s trick__slot--empty" aria-hidden />}
      {cards.map(({ seat, card }, order) => {
        const slot = SLOT[positionOf(seat, bottom)];
        const state = seat === leading && cards.length > 1 ? (done ? "is-winner" : "is-leading") : "";
        return (
          <div
            key={card}
            className={`trick__slot trick__slot--${slot} ${card === fresh ? "is-fresh" : ""} ${state}`}
            style={{ zIndex: order + 1 }}
          >
            <PlayingCard card={card} size="md" />
          </div>
        );
      })}
      {done && winner !== null && (
        <div className={`trick__label trick__label--${SLOT[positionOf(winner, bottom)]}`}>
          {peeking ? `${winnerName} took the last trick` : `${winnerName} ${you ? "take" : "takes"} it`}
        </div>
      )}
      {canPeek && (
        <button type="button" className={`trick__peek ${peeking ? "is-on" : ""}`} aria-pressed={peeking} onClick={() => setPeek(!peeking)}>
          {peeking ? "Back to this trick" : "Last trick"}
        </button>
      )}
    </div>
  );
}
