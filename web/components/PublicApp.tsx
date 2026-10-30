"use client";

import { type FormEvent, useEffect, useState } from "react";

import { ApiError, api } from "@/lib/api";
import type { GameView } from "@/lib/types";

import { GameScreen } from "./GameScreen";

// What this browser remembers: a random visitor id, the name last typed, and the game in play
// with its token, so a reload carries on where it left off.
const VISITOR_KEY = "btig:visitor";
const NAME_KEY = "btig:name";
const GAME_KEY = "btig:game";
const NAME_MAX = 20;

interface SavedGame {
  id: string;
  token: string;
}

function read(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null; // storage blocked: nothing remembered
  }
}

function write(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key);
    else window.localStorage.setItem(key, value);
  } catch {
    // storage blocked: the game still plays, it just is not resumed after a reload
  }
}

function newVisitorId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return Array.from({ length: 32 }, () => Math.floor(Math.random() * 16).toString(16)).join("");
}

function visitorId(): string {
  const known = read(VISITOR_KEY);
  if (known && /^[A-Za-z0-9-]{8,64}$/.test(known)) return known;
  const made = newVisitorId();
  write(VISITOR_KEY, made);
  return made;
}

function savedGame(): SavedGame | null {
  const raw = read(GAME_KEY);
  if (!raw) return null;
  try {
    const value = JSON.parse(raw) as Partial<SavedGame>;
    return typeof value.id === "string" && typeof value.token === "string" ? { id: value.id, token: value.token } : null;
  } catch {
    return null;
  }
}

type Stage = { kind: "loading" } | { kind: "welcome" } | { kind: "game"; game: GameView };

/** The public table: type a name, then play three B-Tig bots; a reload resumes the game. */
export function PublicApp() {
  const [stage, setStage] = useState<Stage>({ kind: "loading" });
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setName(read(NAME_KEY) ?? "");
    const saved = savedGame();
    if (!saved) {
      setStage({ kind: "welcome" });
      return;
    }
    api.setToken(saved.id, saved.token);
    api
      .game(saved.id)
      .then((game) => setStage({ kind: "game", game }))
      .catch((e: unknown) => {
        // A game left, finished long ago or lost to an update: start afresh.
        if (!(e instanceof ApiError) || e.status >= 500) setError(e instanceof Error ? e.message : String(e));
        write(GAME_KEY, null);
        setStage({ kind: "welcome" });
      });
  }, []);

  const start = async (event: FormEvent) => {
    event.preventDefault();
    const typed = name.trim().replace(/\s+/g, " ");
    if (!typed) {
      setError("Type your name first.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const created = await api.startPublic(typed, visitorId());
      write(NAME_KEY, typed);
      write(GAME_KEY, JSON.stringify({ id: created.game.id, token: created.token }));
      api.setToken(created.game.id, created.token);
      setStage({ kind: "game", game: created.game });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const leave = async (game: GameView, finished: boolean) => {
    if (!finished) {
      if (!window.confirm("Leave this game for a new one? You can't come back to it.")) return;
      try {
        await api.leave(game.id);
      } catch {
        // already gone: nothing to leave
      }
    }
    write(GAME_KEY, null);
    setStage({ kind: "welcome" });
  };

  if (stage.kind === "game") {
    const game = stage.game;
    return (
      <GameScreen
        key={game.id}
        initial={game}
        onNewGame={(finished) => void leave(game, finished)}
        onOpen={() => undefined}
      />
    );
  }

  return (
    <main className="welcome">
      <section className="welcome__card panel">
        <h1 className="wordmark wordmark--big">
          Estimation<span className="wordmark__dot">.</span>
        </h1>
        <p className="welcome__lede">
          You against three B-Tig bots, by the rules of the Pocket Estimation app. A game is 18 rounds.
        </p>
        {stage.kind === "loading" ? (
          <p className="muted">Loading…</p>
        ) : (
          <form className="welcome__form" onSubmit={(event) => void start(event)}>
            <label className="welcome__label" htmlFor="name">
              Your name
            </label>
            <input
              id="name"
              className="input welcome__input"
              value={name}
              maxLength={NAME_MAX}
              autoComplete="nickname"
              autoCapitalize="words"
              enterKeyHint="go"
              onChange={(event) => setName(event.target.value)}
            />
            <button type="submit" className="btn btn--primary btn--wide" disabled={busy}>
              {busy ? "Dealing…" : "Play"}
            </button>
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
          </form>
        )}
        <p className="welcome__note">Your games are saved with the name you type, to see how people play against B-Tig.</p>
      </section>
    </main>
  );
}
