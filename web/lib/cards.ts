// Card ids and trumps as the engine encodes them (SPEC §5.1).

export const SUIT_SYMBOLS = ["♣", "♦", "♥", "♠"] as const;
export const TRUMP_SYMBOLS = ["♣", "♦", "♥", "♠", "NT"] as const;
export const TRUMP_NAMES = ["Clubs", "Diamonds", "Hearts", "Spades", "Sans"] as const;
export const RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"] as const;
export const SANS = 4;

export const suitOf = (card: number): number => Math.floor(card / 13);
export const rankOf = (card: number): number => card % 13;
export const rankLabel = (card: number): string => RANKS[rankOf(card)] ?? "?";
export const suitSymbol = (suit: number): string => SUIT_SYMBOLS[suit] ?? "?";
export const isRed = (suit: number): boolean => suit === 1 || suit === 2;
export const cardText = (card: number): string => `${rankLabel(card)}${suitSymbol(suitOf(card))}`;

export const trumpSymbol = (trump: number): string => TRUMP_SYMBOLS[trump] ?? "?";
export const trumpName = (trump: number): string => TRUMP_NAMES[trump] ?? "?";
export const bidText = (tricks: number, trump: number): string =>
  `${tricks} ${trump === SANS ? "Sans" : trumpSymbol(trump)}`;

// Spades, Hearts, Clubs, Diamonds: black and red alternate, so neighbouring suits never look alike.
const DISPLAY_ORDER = [3, 2, 0, 1];

/** Cards grouped by suit in display order, high to low within a suit. */
export function bySuit(cards: number[]): { suit: number; cards: number[] }[] {
  return DISPLAY_ORDER.map((suit) => ({
    suit,
    cards: cards.filter((card) => suitOf(card) === suit).sort((a, b) => b - a),
  }));
}

/** A hand in display order: suits alternating colour, high to low within a suit. */
export function sortHand(cards: number[]): number[] {
  return bySuit(cards).flatMap((group) => group.cards);
}

/** The seat whose card wins ``trick`` so far (PLY-3): the highest trump if any was played, else
 * the highest card of the led suit. ``trick`` is in play order; ``trump`` is null or SANS for no
 * trump. Only public cards go in, so this never reveals a hand. */
export function trickLeader(trick: { seat: number; card: number }[], trump: number | null): number | null {
  const led = trick[0];
  if (!led) return null;
  const trumpSuit = trump !== null && trump !== SANS ? trump : null;
  let best = led; // always of the led suit or trump, so an off-suit card can never take over
  for (const played of trick.slice(1)) {
    const suit = suitOf(played.card);
    const bestSuit = suitOf(best.card);
    if (suit === trumpSuit && bestSuit !== trumpSuit) best = played;
    else if (suit === bestSuit && rankOf(played.card) > rankOf(best.card)) best = played;
  }
  return best.seat;
}
