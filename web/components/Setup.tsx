"use client";

import { useMemo, useState } from "react";

import { api } from "@/lib/api";
import { formatElo } from "@/lib/seats";
import type { BotView, GameView } from "@/lib/types";

type Mode = "watch" | "play";

const FAMILIES: { family: string; title: string }[] = [
  { family: "rl", title: "Self-play RL (M4)" },
  { family: "pimc", title: "PIMC search (M3)" },
  { family: "heuristic", title: "HeuristicBot (M2)" },
  { family: "random", title: "RandomBot" },
];

const SEAT_TITLES = ["Seat 0", "Seat 1", "Seat 2", "Seat 3"];

function pick(bots: BotView[], prefix: string, fallback: number): string {
  const found = bots.find((bot) => bot.label === prefix) ?? bots.find((bot) => bot.label.startsWith(prefix));
  return (found ?? bots[Math.min(fallback, bots.length - 1)])?.spec ?? "heuristic";
}

function defaults(bots: BotView[]): string[] {
  return [
    pick(bots, "RL final", bots.length - 1),
    pick(bots, "PIMC, 8 deals", bots.length - 2),
    pick(bots, "HeuristicBot", 0),
    pick(bots, "RL warm start", bots.length - 3),
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
  const groups = FAMILIES.map(({ family, title }) => ({
    title,
    bots: bots
      .filter((bot) => bot.family === family)
      .sort((a, b) => (b.elo ?? -1) - (a.elo ?? -1)),
  })).filter((group) => group.bots.length > 0);
  const known = bots.some((bot) => bot.spec === value);
  return (
    <select id={id} className="select" value={value} onChange={(e) => onChange(e.target.value)}>
      {!known && <option value={value}>{value}</option>}
      {groups.map((group) => (
        <optgroup key={group.title} label={group.title}>
          {group.bots.map((bot) => (
            <option key={bot.spec} value={bot.spec}>
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

  const seats = useMemo(() => chosen ?? (bots ? defaults(bots) : null), [chosen, bots]);
  const ladder = useMemo(
    () => (bots ?? []).filter((bot) => bot.elo !== null).sort((a, b) => (b.elo ?? 0) - (a.elo ?? 0)),
    [bots],
  );

  const setSeat = (seat: number, spec: string) => {
    if (!seats) return;
    const next = [...seats];
    next[seat] = spec;
    setChosen(next);
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
        <p className="eyebrow">Pocket Estimation rules · 4 players · 18 rounds</p>
        <h1 className="setup__title">
          Estimation<span className="setup__title-dot">.</span>
        </h1>
        <p className="setup__lede">
          Pull up a chair. Watch four bots play a whole Bola and step through every call, card and
          score, or take a seat yourself against three of them.
        </p>
        {ladder.length > 0 && (
          <div className="ladder">
            <h2 className="panel-title">Elo ladder</h2>
            <ol className="ladder__list">
              {ladder.map((bot) => (
                <li key={bot.spec} className={`ladder__row ladder__row--${bot.family}`}>
                  <span>{bot.label}</span>
                  <span className="mono">{formatElo(bot.elo)}</span>
                </li>
              ))}
            </ol>
            <p className="footnote">200 duplicate sets per neighbouring pair, RandomBot fixed at 400.</p>
          </div>
        )}
      </section>

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
    </main>
  );
}
