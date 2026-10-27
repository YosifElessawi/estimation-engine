import { SANS, isRed, rankLabel, suitOf, suitSymbol, trumpSymbol } from "@/lib/cards";
import type { ActionView } from "@/lib/types";

export function TrumpGlyph({ trump }: { trump: number }) {
  if (trump === SANS) return <span className="glyph glyph--sans">NT</span>;
  return <span className={`glyph ${isRed(trump) ? "glyph--red" : "glyph--black"}`}>{trumpSymbol(trump)}</span>;
}

export function CardGlyph({ card }: { card: number }) {
  const suit = suitOf(card);
  return (
    <span className={`glyph glyph--card ${isRed(suit) ? "glyph--red" : "glyph--black"}`}>
      {rankLabel(card)}
      {suitSymbol(suit)}
    </span>
  );
}

/** An action as short text with suit glyphs, for logs and the Why panel. */
export function ActionText({ action }: { action: ActionView }) {
  const { kind, tricks, trump, card } = action;
  if (kind === "play" && card !== null) return <CardGlyph card={card} />;
  if ((kind === "bid" || kind === "super_call" || kind === "with") && tricks !== null && trump !== null) {
    const word = kind === "bid" ? "Bid" : kind === "with" ? "With" : "Super call";
    return (
      <span>
        {word} {tricks} <TrumpGlyph trump={trump} />
      </span>
    );
  }
  if (kind === "estimate" && tricks !== null) return <span>Estimate {tricks}</span>;
  return <span>{action.label}</span>;
}

/** A call as the few words a player would say at the table, for the speech bubbles. */
export function CallText({ action }: { action: ActionView }) {
  const { kind, tricks, trump } = action;
  if (kind === "bid" && tricks !== null && trump !== null) {
    return (
      <span>
        {tricks} <TrumpGlyph trump={trump} />
      </span>
    );
  }
  if (kind === "super_call" && tricks !== null && trump !== null) {
    return (
      <span>
        Super {tricks} <TrumpGlyph trump={trump} />
      </span>
    );
  }
  if (kind === "estimate" && tricks !== null) return <span>Est {tricks}</span>;
  if (kind === "no_dash") return <span>No dash</span>;
  if (kind === "dash") return <span>Dash</span>;
  if (kind === "keep") return <span>Keep</span>;
  if (kind === "with") return <span>With</span>;
  return <span>{action.label}</span>;
}
