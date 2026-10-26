"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { seatName } from "@/lib/seats";
import type { Frame, FramesView, GameView, SeekTarget } from "@/lib/types";

import { DecisionPanel, PlayBar, SPEED_FACTORS, type Speed, WatchControls } from "./Controls";
import { RoundLog, Scoreboard, WhyPanel } from "./Sidebar";
import { RoundResult, Table } from "./Table";

const CHUNK = 80;

/** How long a frame stays on screen while the game plays itself, in ms at normal speed. */
function dwell(frame: Frame): number {
  if (frame.kind === "deal") return 1100;
  if (frame.kind === "round_end") return 3600;
  if (frame.kind === "game_end") return 0;
  const event = frame.event;
  if (event?.action.kind === "play") return event.trick_winner !== null ? 1500 : 750;
  return 600;
}

interface Props {
  initial: GameView;
  onNewGame: () => void;
  onOpen: (game: GameView) => void;
}

export function GameScreen({ initial, onNewGame, onOpen }: Props) {
  const [game, setGame] = useState(initial);
  const spectator = game.mode !== "play";
  const bottom = game.human_seat ?? 0;

  const cache = useRef(new Map<number, Frame>());
  const knownRef = useRef(initial.known);
  const finishedRef = useRef(initial.finished);
  const [, setVersion] = useState(0);
  const [index, setIndex] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [autoplay, setAutoplay] = useState(false);
  const [follow, setFollow] = useState(true);
  const [speed, setSpeed] = useState<Speed>("normal");

  const known = knownRef.current;
  const finished = finishedRef.current;
  const frame = cache.current.get(index) ?? null;
  const last = known - 1;

  const store = useCallback((view: FramesView) => {
    for (const f of view.frames) cache.current.set(f.index, f);
    knownRef.current = view.known;
    finishedRef.current = view.finished;
    setVersion((v) => v + 1);
  }, []);

  const report = useCallback((e: unknown) => {
    setError(e instanceof Error ? e.message : String(e));
  }, []);

  const fetchFrames = useCallback(
    async (start: number, count: number) => store(await api.frames(game.id, start, count)),
    [game.id, store],
  );

  // First frames: a watched game's opening, or everything up to the human's first decision.
  useEffect(() => {
    fetchFrames(0, spectator ? CHUNK : 500).catch(report);
  }, [fetchFrames, spectator, report]);

  const goTo = useCallback(
    async (target: number) => {
      let i = Math.max(0, target);
      if (!cache.current.has(i)) {
        if (!spectator || (finishedRef.current && i >= knownRef.current)) {
          setIndex(Math.min(i, knownRef.current - 1));
          return;
        }
        setBusy(true);
        try {
          await fetchFrames(i, CHUNK);
        } catch (e) {
          report(e);
          setAutoplay(false);
          return;
        } finally {
          setBusy(false);
        }
        if (!cache.current.has(i)) i = knownRef.current - 1;
      }
      setIndex(i);
    },
    [spectator, fetchFrames, report],
  );

  // Watch: fetch ahead quietly so stepping and autoplay stay smooth.
  const prefetching = useRef(false);
  useEffect(() => {
    if (!spectator || finishedRef.current || prefetching.current) return;
    if (cache.current.has(index + 25)) return;
    let start = index + 1;
    while (cache.current.has(start)) start += 1;
    if (finishedRef.current && start >= knownRef.current) return;
    prefetching.current = true;
    fetchFrames(start, CHUNK)
      .catch(report)
      .finally(() => {
        prefetching.current = false;
      });
  }, [index, spectator, fetchFrames, report]);

  // Watch: autoplay.
  useEffect(() => {
    if (!spectator || !autoplay || !frame) return;
    if (frame.kind === "game_end") {
      setAutoplay(false);
      return;
    }
    const timer = setTimeout(() => void goTo(index + 1), dwell(frame) * SPEED_FACTORS[speed]);
    return () => clearTimeout(timer);
  }, [spectator, autoplay, frame, index, speed, goTo]);

  // Play: show the bots' moves one by one up to the human's turn. A round's result waits for
  // Continue.
  useEffect(() => {
    if (spectator || !follow || !frame || index >= last) return;
    if (frame.kind === "round_end") return;
    const timer = setTimeout(() => setIndex(index + 1), dwell(frame) * SPEED_FACTORS[speed]);
    return () => clearTimeout(timer);
  }, [spectator, follow, frame, index, last, speed]);

  const step = useCallback(
    (delta: number) => {
      if (spectator) {
        setAutoplay(false);
        void goTo(index + delta);
      } else {
        setFollow(false);
        setIndex(Math.max(0, Math.min(index + delta, last)));
      }
    },
    [spectator, goTo, index, last],
  );

  const seek = useCallback(
    async (direction: "next" | "prev", target: SeekTarget) => {
      setAutoplay(false);
      setBusy(true);
      try {
        const found = await api.seek(game.id, index, direction, target);
        if (spectator) {
          // Seeking may have made frames up to the target; fetch from there to learn how many.
          store(await api.frames(game.id, found.index, CHUNK));
        } else {
          setFollow(false);
        }
        setIndex(found.index);
      } catch (e) {
        report(e);
      } finally {
        setBusy(false);
      }
    },
    [game.id, index, spectator, store, report],
  );

  const move = useCallback(
    async (action: number) => {
      if (!frame?.legal || busy || index !== last) return;
      setBusy(true);
      setError(null);
      try {
        const view = await api.move(game.id, action, frame.index);
        store(view);
        setFollow(true);
        setIndex(frame.index + 1);
        if (view.finished) setGame(await api.game(game.id));
      } catch (e) {
        report(e);
      } finally {
        setBusy(false);
      }
    },
    [frame, busy, index, last, game.id, store, report],
  );

  const review = useCallback(async () => {
    try {
      onOpen(await api.review(game.id));
    } catch (e) {
      report(e);
    }
  }, [game.id, onOpen, report]);

  // Keyboard: arrows step (Shift jumps a trick), Space plays or pauses a watched game.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        event.preventDefault();
        const forward = event.key === "ArrowRight";
        if (event.shiftKey) void seek(forward ? "next" : "prev", "trick");
        else step(forward ? 1 : -1);
      } else if (event.key === " " && spectator) {
        event.preventDefault();
        setAutoplay((on) => !on);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [seek, step, spectator]);

  const live = !spectator && index === last;
  const playableCards = useMemo(() => {
    const cards = new Set<number>();
    if (!live || busy || !frame?.legal) return cards;
    for (const action of frame.legal) if (action.kind === "play" && action.card !== null) cards.add(action.card);
    return cards;
  }, [live, busy, frame]);

  const decisions = useMemo(
    () => (live && frame?.legal ? frame.legal.filter((action) => action.kind !== "play") : []),
    [live, frame],
  );

  const onPlayCard = useCallback(
    (card: number) => {
      const action = frame?.legal?.find((a) => a.kind === "play" && a.card === card);
      if (action) void move(action.id);
    },
    [frame, move],
  );

  if (!frame) {
    return (
      <div className="game game--loading">
        <TopBar game={game} onNewGame={onNewGame} />
        <p className="notice">{error ?? "Dealing…"}</p>
      </div>
    );
  }

  const turn = frame.table.current_player;
  const resultActions =
    frame.kind === "game_end" ? (
      <>
        {game.mode !== "review" && <span className="small muted result__saved">Saved under Past games on the start page.</span>}
        {game.mode === "play" && (
          <button type="button" className="button button--primary" onClick={review}>
            Review with every hand
          </button>
        )}
        <button type="button" className="button" onClick={onNewGame}>
          New game
        </button>
      </>
    ) : !spectator && frame.kind === "round_end" && index < last ? (
      <button type="button" className="button button--primary" onClick={() => {
        setFollow(true);
        setIndex(index + 1);
      }}>
        Continue to round {frame.round_no + 1}
      </button>
    ) : null;

  return (
    <div className={`game game--${game.mode}`}>
      <TopBar game={game} onNewGame={onNewGame} />
      <div className="game__body">
        <section className="game__main">
          <Table
            frame={frame}
            game={game}
            bottom={bottom}
            playableCards={playableCards}
            onPlayCard={onPlayCard}
            overlay={
              frame.kind === "round_end" || frame.kind === "game_end" ? (
                <RoundResult frame={frame} game={game} bottom={bottom} actions={resultActions} />
              ) : null
            }
          />
          {spectator ? (
            <WatchControls
              index={index}
              known={known}
              finished={finished}
              autoplay={autoplay}
              speed={speed}
              busy={busy}
              onStep={step}
              onGoTo={(i) => {
                setAutoplay(false);
                void goTo(i);
              }}
              onSeek={(direction, target) => void seek(direction, target)}
              onAutoplay={setAutoplay}
              onSpeed={setSpeed}
            />
          ) : (
            <PlayBar
              frame={frame}
              live={follow || live}
              animating={index < last}
              busy={busy}
              turnName={turn !== null ? seatName(game, turn) : null}
              speed={speed}
              onSpeed={setSpeed}
              onLive={() => {
                setFollow(true);
                setIndex(last);
              }}
              onStep={step}
              canStepBack={index > 0}
              canStepForward={index < last}
            />
          )}
          {error && (
            <p className="error" role="alert">
              {error}{" "}
              <button type="button" className="link" onClick={() => setError(null)}>
                dismiss
              </button>
            </p>
          )}
        </section>
        <aside className="game__side">
          {decisions.length > 0 && <DecisionPanel legal={decisions} busy={busy} onAct={(action) => void move(action)} />}
          <Scoreboard frame={frame} game={game} />
          {spectator && <WhyPanel frame={frame} game={game} />}
          <RoundLog frame={frame} game={game} />
        </aside>
      </div>
    </div>
  );
}

function TopBar({ game, onNewGame }: { game: GameView; onNewGame: () => void }) {
  const modeName = game.mode === "watch" ? "Watching" : game.mode === "play" ? "Playing" : "Reviewing";
  return (
    <header className="topbar">
      <span className="wordmark">
        Estimation<span className="wordmark__dot">.</span>
      </span>
      <span className="topbar__mode">{modeName}</span>
      <span className="topbar__seed mono" title="Same seats and seed replay the same game">
        {game.seed === null ? "seed hidden until the end" : `seed ${game.seed}`}
      </span>
      <span className="topbar__spacer" />
      <button type="button" className="button button--ghost" onClick={onNewGame}>
        New game
      </button>
    </header>
  );
}
