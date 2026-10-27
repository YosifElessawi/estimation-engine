import type { CSSProperties } from "react";

import { isRed, rankLabel, suitOf, suitSymbol } from "@/lib/cards";

type Size = "mini" | "md" | "lg";

interface Props {
  card: number;
  size?: Size;
  playable?: boolean;
  dimmed?: boolean;
  trump?: boolean;
  style?: CSSProperties;
  className?: string;
  onPlay?: (card: number) => void;
}

const FACES = new Set(["J", "Q", "K"]);

/** One card face. With ``onPlay`` and ``playable`` it is a button. */
export function PlayingCard({ card, size = "md", playable, dimmed, trump, style, className, onPlay }: Props) {
  const suit = suitOf(card);
  const symbol = suitSymbol(suit);
  const rank = rankLabel(card);
  const classes = [
    "card",
    `card--${size}`,
    isRed(suit) ? "is-red" : "is-black",
    FACES.has(rank) ? "is-face" : "",
    playable ? "is-playable" : "",
    dimmed ? "is-dimmed" : "",
    trump ? "is-trump" : "",
    className ?? "",
  ]
    .filter(Boolean)
    .join(" ");
  const face =
    size === "mini" ? (
      <>
        <span className="card__rank">{rank}</span>
        <span className="card__suit">{symbol}</span>
      </>
    ) : (
      <>
        <span className="card__corner">
          <span className="card__rank">{rank}</span>
          <span className="card__suit">{symbol}</span>
        </span>
        <span className="card__pip" aria-hidden>
          {FACES.has(rank) ? rank : symbol}
        </span>
        <span className="card__foot" aria-hidden>
          {symbol}
        </span>
      </>
    );
  if (onPlay && playable) {
    return (
      <button type="button" className={classes} style={style} onClick={() => onPlay(card)} aria-label={`Play ${rank}${symbol}`}>
        {face}
      </button>
    );
  }
  return (
    <span className={classes} style={style} aria-label={`${rank}${symbol}`}>
      {face}
    </span>
  );
}

/** A small fan of card backs with the number of cards a hidden hand holds. */
export function Backs({ count }: { count: number }) {
  return (
    <span className="backs" aria-label={`${count} cards`}>
      {count > 0 && (
        <span className="backs__fan" aria-hidden>
          {Array.from({ length: Math.min(count, 3) }, (_, i) => (
            <span key={i} className="backs__card" />
          ))}
        </span>
      )}
      <span className="backs__count">{count}</span>
    </span>
  );
}
