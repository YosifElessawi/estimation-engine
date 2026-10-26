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

/** Cards grouped by suit, strongest suit first (Spades, Hearts, Diamonds, Clubs), high to low. */
export function bySuit(cards: number[]): { suit: number; cards: number[] }[] {
  return [3, 2, 1, 0].map((suit) => ({
    suit,
    cards: cards.filter((card) => suitOf(card) === suit).sort((a, b) => b - a),
  }));
}

/** A hand in play order for display: by suit (Spades first), high to low within a suit. */
export function sortHand(cards: number[]): number[] {
  return bySuit(cards).flatMap((group) => group.cards);
}
