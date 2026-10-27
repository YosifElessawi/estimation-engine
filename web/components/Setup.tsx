"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import { type Position, formatElo, positionOf } from "@/lib/seats";
import type { BotView, GameView, SavedGame } from "@/lib/types";

import { PastGames } from "./PastGames";
import { Avatar } from "./Table";

type Mode = "watch" | "play";

// The roster's tables (estimation.arena.roster), in the order the menus list them. A bot sits
// under its first table; one the roster has no table for goes under "Other".
const GROUPS = ["Leaderboard", "RL checkpoints", "Measuring rungs"];
const OTHER = "Other";

const groupOf = (bot: BotView): string => {
  const first = bot.groups[0];
  return first && GROUPS.includes(first) ? first : OTHER;
};

const byElo = (a: BotView, b: BotView): number => (b.elo ?? -1) - (a.elo ?? -1);

function pick(bots: BotView[], names: string[], fallback: number): string {
  for (const name of names) {
    const found = bots.find((bot) => bot.label === name);
    if (found) return found.spec;
  }
  const leaders = bots.filter((bot) => groupOf(bot) === "Leaderboard").sort(byElo);
  return (leaders[fallback] ?? bots[0])?.spec ?? "heuristic";
}

/** The strongest bots of the leaderboard, one per seat. */
function defaults(bots: BotView[]): string[] {
  return [
    pick(bots, ["RL staged final", "RL M4 final"], 1),
    pick(bots, ["PIMC"], 0),
    pick(bots, ["Yosif bot"], 2),
    pick(bots, ["HeuristicBot"], 3),
  ];
}

function BotSelect({
  bots,
  value,
  onChange,
  id,
}: {
  bots: BotView[];
  value: string;
  onChange: (spec: string) => void;
  id: string;
}) {
  const groups = [...GROUPS, OTHER]
    .map((title) => ({ title, bots: bots.filter((bot) => groupOf(bot) === title).sort(byElo) }))
    .filter((group) => group.bots.length > 0);
  const known = bots.some((bot) => bot.spec === value);
  return (
    <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)}>
      {!known && <option value={value}>{value}</option>}
      {groups.map((group) => (
        <optgroup key={group.title} label={group.title}>
          {group.bots.map((bot) => (
            <option key={bot.spec} value={bot.spec} title={bot.description}>
              {bot.label} · {formatElo(bot.elo)}
            </option>
          ))}
        </optgroup>
      ))}
    </select>
  );
}

interface Props {
  bots: BotView[] | null;
  loadError: string | null;
  onStart: (game: GameView) => void;
}

const SEATS = [0, 1, 2, 3];

export function Setup({ bots, loadError, onStart }: Props) {
  const [mode, setMode] = useState<Mode>("play");
  const [chosen, setChosen] = useState<string[] | null>(null);
  const [humanSeat, setHumanSeat] = useState(0);
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [past, setPast] = useState<SavedGame[]>([]);
  const [allRatings, setAllRatings] = useState(false);

  useEffect(() => {
    if (!bots) return;
    api
      .history()
      .then(setPast)
      .catch((e: unknown) => setError(`Could not load past games: ${e instanceof Error ? e.message : String(e)}`));
  }, [bots]);

  const seats = useMemo(() => chosen ?? (bots ? defaults(bots) : null), [chosen, bots]);
  const rated = useMemo(() => (bots ?? []).filter((bot) => bot.elo !== null).sort(byElo), [bots]);
  const leaders = rated.filter((bot) => groupOf(bot) === "Leaderboard");
  const others = rated.filter((bot) => groupOf(bot) !== "Leaderboard");
  const top = Math.max(...rated.map((bot) => bot.elo ?? 0), 1);
  const viewer = mode === "play" ? humanSeat : 0;

  const setSeat = (seat: number, spec: string) => {
    if (!seats) return;
    const next = [...seats];
    next[seat] = spec;
    setChosen(next);
  };

  const sameDeals = (game: SavedGame) => {
    const current = seats ?? defaults(bots ?? []);
    setMode(game.human_seat === null ? "watch" : "play");
    if (game.human_seat !== null) setHumanSeat(game.human_seat);
    setChosen(game.seats.map((seat, i) => (i === game.human_seat ? (current[i] ?? "heuristic") : seat.spec)));
    setSeed(String(game.seed));
    setError(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const review = async (game: SavedGame) => {
    setBusy(true);
    setError(null);
    try {
      onStart(await api.reviewSaved(game.id));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  const start = async () => {
    if (!seats) return;
    const lineup = seats.map((spec, seat) => (mode === "play" && seat === humanSeat ? "human" : spec));
    const trimmed = seed.trim();
    if (trimmed !== "" && !/^\d+$/.test(trimmed)) {
      setError("The seed is a whole number, or leave it blank for a random one.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      onStart(await api.newGame(lineup, trimmed === "" ? null : Number(trimmed)));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setBusy(false);
    }
  };

  const seatCell = (seat: number) => {
    const position: Position = positionOf(seat, viewer);
    const isHuman = mode === "play" && seat === humanSeat;
    const bot = bots?.find((b) => b.spec === seats?.[seat]);
    return (
      <div key={seat} className={`pick pick--${position} ${isHuman ? "is-human" : ""}`}>
        <Avatar seat={seat} human={isHuman} name={isHuman ? "You" : (bot?.label ?? "Bot")} />
        <div className="pick__body">
          {isHuman ? (
            <span className="pick__you">You</span>
          ) : (
            bots &&
            seats && (
              <BotSelect id={`seat-${seat}`} bots={bots} value={seats[seat] ?? "heuristic"} onChange={(spec) => setSeat(seat, spec)} />
            )
          )}
          <span className="pick__meta">
            {seat === 0 && <span>First to act</span>}
            {mode === "play" && !isHuman && (
              <button type="button" className="link" onClick={() => setHumanSeat(seat)}>
                Sit here
              </button>
            )}
          </span>
        </div>
      </div>
    );
  };

  return (
    <main className="lobby">
      <header className="lobby__head">
        <h1 className="wordmark wordmark--big">
          Estimation<span className="wordmark__dot">.</span>
        </h1>
        <p className="lobby__lede">Pocket Estimation rules, four players, 18 rounds. Take a seat against three bots, or watch four of them play.</p>
      </header>

      <div className="lobby__grid">
        <section className="panel deal" aria-label="New game">
          <div className="seg seg--big" role="tablist" aria-label="Mode">
            {(["play", "watch"] as const).map((m) => (
              <button key={m} type="button" role="tab" aria-selected={mode === m} className={mode === m ? "is-on" : ""} onClick={() => setMode(m)}>
                <b>{m === "play" ? "Play" : "Watch"}</b>
                <span>{m === "play" ? "you and three bots" : "four bots, every hand shown"}</span>
              </button>
            ))}
          </div>

          {!bots ? (
            <p className="notice">{loadError ? `Waiting for the game server… (${loadError})` : "Loading the bots…"}</p>
          ) : (
            <div className="minitable">
              <div className="minitable__felt" aria-hidden />
              {SEATS.map(seatCell)}
            </div>
          )}

          <div className="deal__seed">
            <label htmlFor="seed">Seed</label>
            <input
              id="seed"
              className="input"
              inputMode="numeric"
              placeholder="random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
            <span className="deal__help">
              {mode === "play"
                ? "Fixes the deal. A random seed stays hidden until the game ends."
                : "The same seats and seed replay the same game."}
            </span>
          </div>

          {error && <p className="error">{error}</p>}

          <button type="button" className="btn btn--primary btn--wide" disabled={!seats || busy} onClick={start}>
            {busy ? "Dealing…" : mode === "watch" ? "Deal and watch" : "Deal me in"}
          </button>
        </section>

        <aside className="lobby__side">
          {rated.length > 0 && (
            <section className="panel" aria-label="Leaderboard">
              <h2 className="panel__title">Leaderboard</h2>
              <ol className="ladder">
                {leaders.map((bot, i) => (
                  <li key={bot.spec} className="ladder__row" title={bot.description}>
                    <span className="ladder__place">{i + 1}</span>
                    <span className="ladder__who">
                      <span className="ladder__name">{bot.label}</span>
                      {bot.description && <span className="ladder__desc">{bot.description}</span>}
                      <span className="ladder__bar" aria-hidden>
                        <span style={{ width: `${Math.max(4, ((bot.elo ?? 0) / top) * 100)}%` }} />
                      </span>
                    </span>
                    <span className="ladder__elo">{formatElo(bot.elo)}</span>
                  </li>
                ))}
              </ol>
              {others.length > 0 && (
                <button type="button" className="link" onClick={() => setAllRatings(!allRatings)}>
                  {allRatings ? "Hide the other ratings" : `Show ${others.length} more: RL checkpoints and measuring rungs`}
                </button>
              )}
              {allRatings && (
                <ol className="ladder ladder--more">
                  {others.map((bot) => (
                    <li key={bot.spec} className="ladder__row" title={bot.description}>
                      <span className="ladder__who">
                        <span className="ladder__name">{bot.label}</span>
                      </span>
                      <span className="ladder__elo">{formatElo(bot.elo)}</span>
                    </li>
                  ))}
                </ol>
              )}
              <p className="panel__note">Elo under the current rules, from 200 duplicate sets per pair. RandomBot is fixed at 400.</p>
            </section>
          )}
          <PastGames games={past} onSameDeals={sameDeals} onReview={(game) => void review(game)} />
        </aside>
      </div>
    </main>
  );
}
