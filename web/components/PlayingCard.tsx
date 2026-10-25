import { isRed, rankLabel, suitOf, suitSymbol } from "@/lib/cards";

type Size = "sm" | "md" | "lg";

interface Props {
  card: number;
  size?: Size;
  playable?: boolean;
  dimmed?: boolean;
  winner?: boolean;
  fresh?: boolean;
  onPlay?: (card: number) => void;
}

/** One card face. With ``onPlay`` and ``playable`` it is a button. */
export function PlayingCard({ card, size = "md", playable, dimmed, winner, fresh, onPlay }: Props) {
  const suit = suitOf(card);
  const symbol = suitSymbol(suit);
  const rank = rankLabel(card);
  const classes = [
    "card",
    `card--${size}`,
    isRed(suit) ? "card--red" : "card--black",
    playable ? "card--playable" : "",
    dimmed ? "card--dimmed" : "",
    winner ? "card--winner" : "",
    fresh ? "card--fresh" : "",
  ]
    .filter(Boolean)
    .join(" ");
  const face = (
    <>
      <span className="card__corner">
        <span className="card__rank">{rank}</span>
        <span className="card__suit">{symbol}</span>
      </span>
      <span className="card__pip" aria-hidden>
        {symbol}
      </span>
      <span className="card__corner card__corner--flip" aria-hidden>
        <span className="card__rank">{rank}</span>
        <span className="card__suit">{symbol}</span>
      </span>
    </>
  );
  if (onPlay && playable) {
    return (
      <button type="button" className={classes} onClick={() => onPlay(card)} title={`Play ${rank}${symbol}`}>
        {face}
      </button>
    );
  }
  return (
    <span className={classes} aria-label={`${rank}${symbol}`}>
      {face}
    </span>
  );
}

export function CardBack({ size = "sm" }: { size?: Size }) {
  return <span className={`card card--${size} card--back`} aria-hidden />;
}
