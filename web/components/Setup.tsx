"use client";

import { useEffect, useMemo, useState } from "react";

import { api } from "@/lib/api";
import { formatElo } from "@/lib/seats";
import type { BotView, GameView, SavedGame } from "@/lib/types";

import { PastGames } from "./PastGames";

type Mode = "watch" | "play";

// The roster's tables (estimation.arena.roster), in the order the menus list them. A bot sits
// under its first table; one the roster has no table for goes under "Other".
const GROUPS = ["Leaderboard", "RL checkpoints", "Measuring rungs"];
const OTHER = "Other";

const SEAT_TITLES = ["Seat 0", "Seat 1", "Seat 2", "Seat 3"];

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

export function Setup({ bots, loadError, onStart }: Props) {
  const [mode, setMode] = useState<Mode>("watch");
  const [chosen, setChosen] = useState<string[] | null>(null);
  const [humanSeat, setHumanSeat] = useState(0);
  const [seed, setSeed] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [past, setPast] = useState<SavedGame[]>([]);

  useEffect(() => {
    if (!bots) return;
    api
      .history()
      .then(setPast)
      .catch((e: unknown) => setError(`Could not load past games: ${e instanceof Error ? e.message : String(e)}`));
  }, [bots]);

  const seats = useMemo(() => chosen ?? (bots ? defaults(bots) : null), [chosen, bots]);
  const [allRatings, setAllRatings] = useState(false);
  const rated = useMemo(() => (bots ?? []).filter((bot) => bot.elo !== null).sort(byElo), [bots]);
  const leaders = rated.filter((bot) => groupOf(bot) === "Leaderboard");
  const others = rated.filter((bot) => groupOf(bot) !== "Leaderboard");

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

  return (
    <main className="setup">
      <section className="setup__intro">
        <p className="eyebrow">Pocket Estimation rules · 4 players · 18 rounds plus replays</p>
        <h1 className="setup__title">
          Estimation<span className="setup__title-dot">.</span>
        </h1>
        <p className="setup__lede">
          Pull up a chair. Watch four bots play a whole Bola and step through every call, card and
          score, or take a seat yourself against three of them.
        </p>
        {rated.length > 0 && (
          <div className="ladder">
            <h2 className="panel-title">Leaderboard</h2>
            <ol className="ladder__list">
              {leaders.map((bot) => (
                <li key={bot.spec} className={`ladder__row ladder__row--lead ladder__row--${bot.family}`}>
                  <span>
                    <span className="ladder__name">{bot.label}</span>
                    {bot.description && <span className="ladder__desc">{bot.description}</span>}
                  </span>
                  <span className="mono">{formatElo(bot.elo)}</span>
                </li>
              ))}
            </ol>
            {others.length > 0 && (
              <button type="button" className="link small" onClick={() => setAllRatings(!allRatings)}>
                {allRatings ? "hide the other ratings" : `show ${others.length} more: RL checkpoints and measuring rungs`}
              </button>
            )}
            {allRatings && (
              <ol className="ladder__list ladder__list--more">
                {others.map((bot) => (
                  <li key={bot.spec} className={`ladder__row ladder__row--${bot.family}`}>
                    <span>{bot.label}</span>
                    <span className="mono">{formatElo(bot.elo)}</span>
                  </li>
                ))}
              </ol>
            )}
            <p className="footnote">
              Elo under the current rules, from 200 duplicate sets per pair; RandomBot fixed at 400.
            </p>
          </div>
        )}
      </section>

      <div className="setup__side">
      <section className="setup__card">
        <div className="mode-switch" role="tablist" aria-label="Mode">
          {(["watch", "play"] as const).map((m) => (
            <button
              key={m}
              type="button"
              role="tab"
              aria-selected={mode === m}
              className={`mode-switch__tab ${mode === m ? "is-active" : ""}`}
              onClick={() => setMode(m)}
            >
              <span className="mode-switch__name">{m === "watch" ? "Watch" : "Play"}</span>
              <span className="mode-switch__hint">
                {m === "watch" ? "four bots, every hand shown" : "you against three bots"}
              </span>
            </button>
          ))}
        </div>

        {!bots && (
          <p className="notice">
            {loadError ? `Waiting for the game server… (${loadError})` : "Loading the bots…"}
          </p>
        )}

        {seats && bots && (
          <div className="seat-picker">
            {SEAT_TITLES.map((title, seat) => {
              const isHuman = mode === "play" && seat === humanSeat;
              return (
                <div key={title} className={`seat-picker__row ${isHuman ? "is-human" : ""}`}>
                  <label className="seat-picker__label" htmlFor={`seat-${seat}`}>
                    {title}
                    {seat === 0 && <span className="seat-picker__first">acts first in round 1</span>}
                  </label>
                  {isHuman ? (
                    <div className="seat-picker__you">You</div>
                  ) : (
                    <BotSelect
                      id={`seat-${seat}`}
                      bots={bots}
                      value={seats[seat] ?? "heuristic"}
                      onChange={(spec) => setSeat(seat, spec)}
                    />
                  )}
                  {mode === "play" && !isHuman && (
                    <button type="button" className="link" onClick={() => setHumanSeat(seat)}>
                      sit here
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <div className="setup__seed">
          <label htmlFor="seed">Seed</label>
          <input
            id="seed"
            className="input mono"
            inputMode="numeric"
            placeholder="random"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
          <span className="footnote">
            {mode === "play"
              ? "Fixes the deal. A random seed stays hidden until the game ends."
              : "Same seats and seed replay the same game."}
          </span>
        </div>

        {error && <p className="error">{error}</p>}

        <button type="button" className="button button--primary" disabled={!seats || busy} onClick={start}>
          {busy ? "Dealing…" : mode === "watch" ? "Deal and watch" : "Deal me in"}
        </button>
      </section>
      <PastGames games={past} onSameDeals={sameDeals} onReview={(game) => void review(game)} />
      </div>
    </main>
  );
}
