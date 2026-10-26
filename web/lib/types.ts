// The API's JSON shapes (server/views.py, server/why.py, server/session.py, server/app.py).

export type Seat = 0 | 1 | 2 | 3;

export type ActionKind =
  | "no_dash"
  | "dash"
  | "bid"
  | "super_call"
  | "with"
  | "pass"
  | "keep"
  | "stay"
  | "estimate"
  | "play";

export interface ActionView {
  id: number;
  kind: ActionKind;
  label: string;
  tricks: number | null;
  trump: number | null;
  card: number | null;
}

export interface EventView {
  seat: number;
  action: ActionView;
  trick_winner: number | null;
}

export interface PlayedCard {
  seat: number;
  card: number;
}

export interface TableView {
  round_no: number;
  phase: "dash" | "auction" | "estimation" | "play" | "done";
  current_player: number | null;
  carry: number;
  scores: number[];
  fixed_trump: number | null;
  fixed_left: number[];
  replays: number;
  super_call: boolean;
  trump: number | null;
  dash_callers: number[];
  void_announced: number[];
  top_bid: { tricks: number; trump: number } | null;
  top_bidder: number | null;
  passed: number[];
  caller: number | null;
  with_players: { seat: number; w: number; trump: number }[];
  estimates: (number | null)[];
  last_estimator: number | null;
  total: number | null;
  over: boolean | null;
  risks: number | null;
  hands: (number[] | null)[];
  hand_sizes: number[];
  leader: number | null;
  trick: PlayedCard[];
  last_trick: { leader: number; cards: PlayedCard[]; winner: number } | null;
  tricks_played: number;
  tricks_won: number[];
  events: EventView[];
  calls: (ActionView | null)[];
}

export interface SeatResultView {
  estimate: number;
  tricks: number;
  won: boolean;
  dash: boolean;
  caller: boolean;
  with_player: boolean;
  super_call: boolean;
  last: boolean;
  risks: number;
  base_points: number;
  dash_points: number;
  call_points: number;
  risk_points: number;
  only_points: number;
  super_points: number;
  points: number;
  multiplier: number;
  score: number;
}

export interface RoundResultView {
  round_no: number;
  skipped: boolean;
  caller: number | null;
  trump: number | null;
  fixed_trump: number | null;
  trump_changed: boolean;
  total: number | null;
  over: boolean | null;
  calls: number;
  all_lose: boolean;
  carry: number;
  multiplier: number;
  next_carry: number;
  replayed: boolean;
  scores: number[];
  seats: SeatResultView[];
}

export interface RoundSummary {
  round_no: number;
  skipped: boolean;
  trump: number | null;
  fixed_trump: number | null;
  trump_changed: boolean;
  over: boolean | null;
  multiplier: number;
  scores: number[];
}

export interface OptionView {
  action: ActionView;
  points: number;
  advantage: number | null;
  chosen: boolean;
}

export interface WhyView {
  net: "CardPlayNet" | "BiddingNet" | "HeuristicBot";
  baseline: number | null;
  options: OptionView[];
  note: string | null;
}

export type FrameKind = "deal" | "action" | "round_end" | "game_end";

export interface Frame {
  index: number;
  kind: FrameKind;
  round_no: number;
  event: EventView | null;
  why: WhyView | null;
  table: TableView;
  scores: number[];
  history: RoundSummary[];
  result: RoundResultView | null;
  winners: number[] | null;
  legal: ActionView[] | null;
}

export interface BotView {
  spec: string;
  label: string;
  family: string;
  elo: number | null;
  explains: boolean;
}

export interface SeatView {
  seat: number;
  spec: string;
  label: string;
  elo: number | null;
  human: boolean;
  explains: boolean;
}

export interface GameView {
  id: string;
  mode: "watch" | "play" | "review";
  seed: number | null;
  human_seat: number | null;
  seats: SeatView[];
  known: number;
  finished: boolean;
  waiting: boolean;
}

export interface FramesView {
  frames: Frame[];
  known: number;
  finished: boolean;
}

export type SeekTarget = "trick" | "round_end" | "deal" | "end";

export interface SavedSeat {
  spec: string;
  label: string;
  elo: number | null;
}

export interface SavedGame {
  id: string;
  finished_at: string;
  mode: "play" | "watch";
  seed: number;
  seed_chosen: boolean;
  human_seat: number | null;
  seats: SavedSeat[];
  scores: number[];
  winners: number[];
  rounds: number;
}
