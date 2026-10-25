"use client";

import { bySuit, isRed, rankLabel, sortHand, suitSymbol, trumpName } from "@/lib/cards";
import { type Position, formatElo, positionOf, seatName, shortName, signed } from "@/lib/seats";
import type { Frame, GameView, RoundResultView, TableView } from "@/lib/types";

import { ActionText, TrumpGlyph } from "./ActionText";
import { CardBack, PlayingCard } from "./PlayingCard";

interface TableProps {
  frame: Frame;
  game: GameView;
  bottom: number;
  playableCards: Set<number>;
  onPlayCard: (card: number) => void;
  overlay: React.ReactNode;
}

const SEAT_ORDER = [0, 1, 2, 3];

export function Table({ frame, game, bottom, playableCards, onPlayCard, overlay }: TableProps) {
  const spectator = game.mode !== "play";
  return (
    <div className="table">
      {SEAT_ORDER.map((seat) => {
        const position = positionOf(seat, bottom);
        return (
          <SeatPanel
            key={seat}
            seat={seat}
            position={position}
            frame={frame}
            game={game}
            ownHand={!spectator && seat === game.human_seat}
            playableCards={playableCards}
            onPlayCard={onPlayCard}
          />
        );
      })}
      <RoundPlate table={frame.table} game={game} frame={frame} />
      <div className="felt">
        <TrickArea frame={frame} bottom={bottom} />
      </div>
      {overlay}
    </div>
  );
}

function badgesOf(table: TableView, seat: number): { text: string; tone: string }[] {
  const badges: { text: string; tone: string }[] = [];
  if (table.caller === seat) badges.push({ text: "Caller", tone: "gold" });
  const withInfo = table.with_players.find((w) => w.seat === seat);
  if (withInfo && table.phase === "auction") badges.push({ text: `With ${withInfo.w}`, tone: "silver" });
  else if (withInfo) badges.push({ text: "With", tone: "silver" });
  if (table.dash_callers.includes(seat)) badges.push({ text: "Dash", tone: "teal" });
  if (table.last_estimator === seat) {
    const risks = table.risks ?? 0;
    badges.push({ text: risks > 0 ? `Last · ${risks} Risk${risks > 1 ? "s" : ""}` : "Last", tone: "rose" });
  }
  if (table.phase === "auction" && table.passed.includes(seat)) badges.push({ text: "Out", tone: "muted" });
  return badges;
}

function SeatPanel({
  seat,
  position,
  frame,
  game,
  ownHand,
  playableCards,
  onPlayCard,
}: {
  seat: number;
  position: Position;
  frame: Frame;
  game: GameView;
  ownHand: boolean;
  playableCards: Set<number>;
  onPlayCard: (card: number) => void;
}) {
  const table = frame.table;
  const info = game.seats[seat];
  const over = frame.kind === "round_end" || frame.kind === "game_end";
  const isTurn = !over && table.current_player === seat;
  const acted = frame.event?.seat === seat;
  const hand = table.hands[seat] ?? null;
  const estimate = table.estimates[seat];
  const call = table.calls[seat];
  const showCall = call && (table.phase === "dash" || table.phase === "auction" || table.phase === "estimation");
  const trumpSuit = table.trump !== null && table.trump < 4 ? table.trump : null;
  const vertical = position === "left" || position === "right";

  return (
    <section
      className={[
        "seat",
        `seat--${position}`,
        isTurn ? "is-turn" : "",
        info?.human ? "is-human" : "",
        acted ? "has-acted" : "",
      ].join(" ")}
      aria-label={`Seat ${seat}`}
    >
      <header className="seat__plate">
        <div className="seat__who">
          <span className="seat__name">{seatName(game, seat)}</span>
          {info && !info.human && <span className="seat__elo mono">{formatElo(info.elo)}</span>}
          {info?.explains && <span className="seat__rl" title="The Why panel explains this bot">why</span>}
        </div>
        <div className="seat__line">
          <span className="seat__tag">Seat {seat}</span>
          <span className="seat__score mono">{frame.scores[seat] ?? 0}</span>
        </div>
        <div className="seat__line">
          <span className="seat__tally">
            {estimate !== null && estimate !== undefined ? (
              <>
                Est <b className="mono">{estimate}</b> · Won <b className="mono">{table.tricks_won[seat]}</b>
              </>
            ) : table.phase === "play" || table.phase === "done" ? (
              <>
                Won <b className="mono">{table.tricks_won[seat]}</b>
              </>
            ) : (
              <span className="muted">no estimate yet</span>
            )}
          </span>
          <span className="seat__badges">
            {badgesOf(table, seat).map((badge) => (
              <span key={badge.text} className={`badge badge--${badge.tone}`}>
                {badge.text}
              </span>
            ))}
          </span>
        </div>
      </header>

      {showCall && call && (
        <div className={`bubble ${acted ? "bubble--fresh" : ""}`}>
          <ActionText action={call} />
        </div>
      )}
      {isTurn && !showCall && <div className="bubble bubble--turn">to act</div>}

      <div className="seat__hand">
        {hand && ownHand ? (
          <div className="fan">
            {sortHand(hand).map((card) => (
              <PlayingCard
                key={card}
                card={card}
                size="lg"
                playable={playableCards.has(card)}
                dimmed={playableCards.size > 0 && !playableCards.has(card)}
                onPlay={onPlayCard}
              />
            ))}
          </div>
        ) : hand ? (
          <SuitRows cards={hand} trumpSuit={trumpSuit} compact={vertical} />
        ) : (
          <div className="backs" aria-label={`${table.hand_sizes[seat]} cards`}>
            {Array.from({ length: table.hand_sizes[seat] ?? 0 }, (_, i) => (
              <CardBack key={i} size="sm" />
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

function SuitRows({ cards, trumpSuit, compact }: { cards: number[]; trumpSuit: number | null; compact: boolean }) {
  return (
    <div className={`suits ${compact ? "suits--compact" : ""}`}>
      {bySuit(cards).map(({ suit, cards: inSuit }) => (
        <div key={suit} className={`suits__row ${suit === trumpSuit ? "is-trump" : ""}`}>
          <span className={`suits__symbol ${isRed(suit) ? "glyph--red" : "glyph--black"}`}>{suitSymbol(suit)}</span>
          <span className="suits__ranks">
            {inSuit.length === 0 ? <span className="muted">–</span> : inSuit.map((card) => <span key={card}>{rankLabel(card)}</span>)}
          </span>
        </div>
      ))}
    </div>
  );
}

const PHASE_NAMES: Record<TableView["phase"], string> = {
  dash: "Dash Calls",
  auction: "Auction",
  estimation: "Estimation",
  play: "Play",
  done: "Round over",
};

function RoundPlate({ table, game, frame }: { table: TableView; game: GameView; frame: Frame }) {
  const fixed = table.fixed_trump;
  const caller = table.caller;
  const contract = caller !== null ? table.estimates[caller] : null;
  return (
    <div className="plate">
      <div className="plate__round">
        <span className="plate__label">Round</span>
        <span className="plate__number">{table.round_no}</span>
      </div>
      <dl className="plate__facts">
        <div>
          <dt>Type</dt>
          <dd>
            {fixed === null ? (
              "Open"
            ) : (
              <>
                Fixed <TrumpGlyph trump={fixed} />
                {table.super_call && " · super call"}
              </>
            )}
          </dd>
        </div>
        <div>
          <dt>Phase</dt>
          <dd>
            {PHASE_NAMES[table.phase]}
            {table.phase === "play" &&
              ` · trick ${frame.event?.trick_winner != null ? table.tricks_played : Math.min(table.tricks_played + 1, 13)}/13`}
          </dd>
        </div>
        <div>
          <dt>Trump</dt>
          <dd>{table.trump === null ? <span className="muted">not set</span> : <><TrumpGlyph trump={table.trump} /> {trumpName(table.trump)}</>}</dd>
        </div>
        {caller !== null && contract !== null && contract !== undefined ? (
          <div>
            <dt>Contract</dt>
            <dd>
              {contract} by {shortName(game, caller)}
            </dd>
          </div>
        ) : table.top_bid && table.top_bidder !== null ? (
          <div>
            <dt>Top bid</dt>
            <dd>
              {table.top_bid.tricks} <TrumpGlyph trump={table.top_bid.trump} /> by {shortName(game, table.top_bidder)}
            </dd>
          </div>
        ) : null}
        {table.total !== null && table.over !== null && (
          <div>
            <dt>Estimates</dt>
            <dd>
              {table.total} · {table.over ? `Over by ${table.total - 13}` : `Under by ${13 - table.total}`}
            </dd>
          </div>
        )}
        {table.carry > 1 && (
          <div>
            <dt>Carry</dt>
            <dd className="plate__carry">x{table.carry}</dd>
          </div>
        )}
      </dl>
      {frame.kind === "deal" && <div className="plate__flash">New deal</div>}
    </div>
  );
}

const SLOT: Record<Position, string> = { bottom: "s", right: "e", top: "n", left: "w" };

function TrickArea({ frame, bottom }: { frame: Frame; bottom: number }) {
  const table = frame.table;
  const event = frame.event;
  let cards = table.trick;
  let winner: number | null = null;
  if (cards.length === 0 && event?.trick_winner !== null && event?.trick_winner !== undefined && table.last_trick) {
    cards = table.last_trick.cards;
    winner = table.last_trick.winner;
  }
  const freshCard = event?.action.kind === "play" ? event.action.card : null;
  return (
    <div className="trick" aria-label="Cards on the table">
      {cards.map(({ seat, card }, order) => {
        const slot = SLOT[positionOf(seat, bottom)];
        return (
          <div key={card} className={`trick__slot trick__slot--${slot}`} style={{ zIndex: order + 1 }}>
            <PlayingCard card={card} size="md" winner={winner === seat} fresh={card === freshCard} />
          </div>
        );
      })}
      {winner !== null && <div className={`trick__winner trick__winner--${SLOT[positionOf(winner, bottom)]}`}>wins the trick</div>}
    </div>
  );
}

function breakdown(seat: RoundResultView["seats"][number]): { label: string; points: number }[] {
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

function roleOf(seat: RoundResultView["seats"][number]): string {
  const roles = [];
  if (seat.caller) roles.push("Caller");
  if (seat.with_player) roles.push("With");
  if (seat.dash) roles.push("Dash");
  if (seat.super_call) roles.push("Super");
  if (seat.last) roles.push(seat.risks ? `Last, ${seat.risks} Risk${seat.risks > 1 ? "s" : ""}` : "Last");
  return roles.join(" · ");
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
  actions: React.ReactNode;
}) {
  const result = frame.result;
  if (!result) return null;
  const order = [0, 1, 2, 3].map((i) => (bottom + i) % 4);
  const final = frame.kind === "game_end";
  const winners = frame.winners ?? [];
  return (
    <div className={`result ${final ? "result--final" : ""}`} role="dialog" aria-label="Round result">
      {final && (
        <div className="result__winners">
          <span className="eyebrow">Game over</span>
          <h2>
            {winners.map((seat) => seatName(game, seat)).join(" and ")} {winners.length > 1 ? "share the win" : winners[0] === game.human_seat && game.mode === "play" ? "win!" : "wins"}
          </h2>
          <p className="mono">{order.map((seat) => `${seatName(game, seat)} ${frame.scores[seat]}`).join("  ·  ")}</p>
        </div>
      )}
      <header className="result__head">
        <h3>
          Round {result.round_no}
          {result.trump !== null && (
            <>
              {" "}
              · <TrumpGlyph trump={result.trump} />
            </>
          )}
        </h3>
        <span className="result__facts">
          {result.skipped
            ? "Skipped: no Caller, everyone scores 0"
            : `${result.total} estimated · ${result.over ? `Over by ${(result.total ?? 13) - 13}` : `Under by ${13 - (result.total ?? 13)}`}`}
          {!result.skipped && result.multiplier !== 1 && ` · scored x${result.multiplier}`}
          {result.all_lose && " · everyone lost"}
          {result.trump_changed &&
            (result.replayed ? " · trump changed, replayed next" : " · trump changed (replay limit reached)")}
        </span>
      </header>
      {!result.skipped && (
        <table className="result__table">
          <thead>
            <tr>
              <th>Player</th>
              <th>Est</th>
              <th>Won</th>
              <th>Points</th>
              <th className="num">Score</th>
            </tr>
          </thead>
          <tbody>
            {order.map((seat) => {
              const row = result.seats[seat];
              if (!row) return null;
              return (
                <tr key={seat} className={row.won ? "is-win" : "is-loss"}>
                  <td>
                    <span className="result__name">{seatName(game, seat)}</span>
                    <span className="result__role">{roleOf(row)}</span>
                  </td>
                  <td className="mono">{row.estimate}</td>
                  <td className="mono">
                    {row.tricks} <span className="result__mark">{row.won ? "✓" : "✗"}</span>
                  </td>
                  <td>
                    {breakdown(row).map((part) => (
                      <span key={part.label} className={`chip ${part.points > 0 ? "chip--plus" : "chip--minus"}`}>
                        {part.label} {signed(part.points)}
                      </span>
                    ))}
                    {row.multiplier !== 1 && <span className="chip chip--mult">x{row.multiplier}</span>}
                  </td>
                  <td className={`num mono ${row.score > 0 ? "plus" : row.score < 0 ? "minus" : ""}`}>{signed(row.score)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {result.next_carry > 1 && <p className="result__carry">Next round counts x{result.next_carry}</p>}
      <div className="result__actions">{actions}</div>
    </div>
  );
}
