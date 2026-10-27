"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { api } from "@/lib/api";
import { trumpName } from "@/lib/cards";
import { seatName } from "@/lib/seats";
import type { Frame, FramesView, GameView, SeekTarget } from "@/lib/types";

import { TrumpGlyph } from "./ActionText";
import { DecisionPanel, SPEED_FACTORS, type Speed, SpeedSelect, StatusPill, WatchBar } from "./Controls";
import { Drawer, type DrawerTab } from "./Drawer";
import { RoundResult } from "./RoundResult";
import { Table } from "./Table";

const CHUNK = 80;

/** How long a frame stays on screen while the game plays itself, in ms at normal speed. */
function dwell(frame: Frame): number {
  if (frame.kind === "deal") return 1300;
  if (frame.kind === "round_end") return 3600;
  if (frame.kind === "game_end") return 0;
  const event = frame.event;
  if (event?.action.kind === "play") return event.trick_winner !== null ? 1600 : 800;
  return 750;
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
  const [drawer, setDrawer] = useState<DrawerTab | null>(null);

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

  // Keyboard: arrows step (Shift jumps a trick), Space plays or pauses a watched game, Escape
  // closes the drawer.
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
      } else if (event.key === "Escape") {
        setDrawer(null);
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

  const tabs: DrawerTab[] = game.seats.some((seat) => seat.explains) && spectator ? ["scores", "log", "why"] : ["scores", "log"];
  const toggle = (tab: DrawerTab) => setDrawer((open) => (open === tab ? null : tab));

  if (!frame) {
    return (
      <div className="game">
        <TopBar game={game} frame={null} drawer={drawer} tabs={tabs} onToggle={toggle} onNewGame={onNewGame} />
        <div className="game__loading">{error ?? "Dealing…"}</div>
      </div>
    );
  }

  const turn = frame.table.current_player;
  const resultActions =
    frame.kind === "game_end" ? (
      <>
        {game.mode !== "review" && <span className="result__saved">Saved under Past games on the start page.</span>}
        {game.mode === "play" && (
          <button type="button" className="btn btn--primary" onClick={review}>
            Review with every hand
          </button>
        )}
        <button type="button" className="btn btn--ghost" onClick={onNewGame}>
          New game
        </button>
      </>
    ) : !spectator && frame.kind === "round_end" && index < last ? (
      <button
        type="button"
        className="btn btn--primary"
        onClick={() => {
          setFollow(true);
          setIndex(index + 1);
        }}
      >
        Continue to round {frame.round_no + 1}
      </button>
    ) : null;

  const dock = spectator ? null : decisions.length > 0 ? (
    <DecisionPanel key={frame.index} legal={decisions} table={frame.table} busy={busy} onAct={(action) => void move(action)} />
  ) : (
    <StatusPill
      frame={frame}
      live={follow || live}
      animating={index < last}
      busy={busy}
      turnName={turn !== null ? seatName(game, turn) : null}
      onLive={() => {
        setFollow(true);
        setIndex(last);
      }}
    />
  );

  return (
    <div className={`game game--${game.mode} ${drawer ? "has-drawer" : ""}`}>
      <TopBar
        game={game}
        frame={frame}
        drawer={drawer}
        tabs={tabs}
        onToggle={toggle}
        onNewGame={onNewGame}
        pace={spectator ? null : <SpeedSelect speed={speed} onSpeed={setSpeed} />}
      />
      <div className="game__body">
        <main className="game__stage">
          <Table
            frame={frame}
            game={game}
            bottom={bottom}
            playableCards={playableCards}
            awaitingMyCard={playableCards.size > 0}
            onPlayCard={onPlayCard}
            dock={dock}
            overlay={
              frame.kind === "round_end" || frame.kind === "game_end" ? (
                <RoundResult frame={frame} game={game} bottom={bottom} actions={resultActions} />
              ) : null
            }
          />
          {error && (
            <div className="toast" role="alert">
              {error}
              <button type="button" className="link" onClick={() => setError(null)}>
                Dismiss
              </button>
            </div>
          )}
        </main>
        {drawer && (
          <Drawer tab={drawer} tabs={tabs} frame={frame} game={game} onTab={setDrawer} onClose={() => setDrawer(null)} />
        )}
      </div>
      {spectator && (
        <WatchBar
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
      )}
    </div>
  );
}

const TOOL_ICONS: Record<DrawerTab, string> = {
  scores: "M5 20V10M12 20V4M19 20v-7",
  log: "M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01",
  why: "M9.1 9a3 3 0 1 1 4.2 2.7c-.8.4-1.3 1.1-1.3 2v.3M12 17.5h.01",
};
const TOOL_NAMES: Record<DrawerTab, string> = { scores: "Scores", log: "Log", why: "Why" };

function TopBar({
  game,
  frame,
  drawer,
  tabs,
  onToggle,
  onNewGame,
  pace,
}: {
  game: GameView;
  frame: Frame | null;
  drawer: DrawerTab | null;
  tabs: DrawerTab[];
  onToggle: (tab: DrawerTab) => void;
  onNewGame: () => void;
  pace?: ReactNode;
}) {
  const table = frame?.table ?? null;
  return (
    <header className="bar">
      <div className="bar__left">
        <span className="wordmark">
          Estimation<span className="wordmark__dot">.</span>
        </span>
        {table && (
          <span className="bar__round">
            Round <b>{table.round_no}</b>
          </span>
        )}
        {table?.fixed_trump !== null && table?.fixed_trump !== undefined && (
          <span className="tag tag--soft">
            Fixed <TrumpGlyph trump={table.fixed_trump} />
          </span>
        )}
        {table && table.carry > 1 && <span className="tag tag--rose">Counts x{table.carry}</span>}
        {game.mode !== "play" && <span className="tag tag--muted">{game.mode === "watch" ? "Watching" : "Reviewing"}</span>}
      </div>
      {frame && <Contract frame={frame} game={game} />}
      <div className="bar__tools">
        {pace}
        {tabs.map((tab) => (
          <button
            key={tab}
            type="button"
            className={`tool ${drawer === tab ? "is-on" : ""}`}
            aria-pressed={drawer === tab}
            onClick={() => onToggle(tab)}
            title={TOOL_NAMES[tab]}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden>
              <path d={TOOL_ICONS[tab]} />
            </svg>
            <span>{TOOL_NAMES[tab]}</span>
          </button>
        ))}
        <button type="button" className="btn btn--ghost btn--small bar__leave" onClick={onNewGame} aria-label="New game">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>
            <path d="M15 4h3a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-3M10 17l-5-5 5-5M5 12h11" />
          </svg>
          <span>New game</span>
        </button>
      </div>
    </header>
  );
}

function Contract({ frame, game }: { frame: Frame; game: GameView }) {
  const table = frame.table;
  const caller = table.caller;
  const contract = caller !== null ? table.estimates[caller] : null;
  const parts: ReactNode[] = [];
  if (table.trump !== null) {
    parts.push(
      <span key="trump" className="trumpchip">
        <span className="trumpchip__suit">
          <TrumpGlyph trump={table.trump} />
        </span>
        <span className="trumpchip__name">{trumpName(table.trump)}</span>
      </span>,
    );
  }
  if (caller !== null && contract !== null && contract !== undefined) {
    parts.push(
      <span key="caller">
        <span className="contract__who">{seatName(game, caller)} called</span>
        <span className="contract__short">Contract</span> <b>{contract}</b>
      </span>,
    );
  }
  if (table.phase === "dash") parts.push(<span key="phase">Dash calls</span>);
  else if (table.phase === "auction") parts.push(<span key="phase">Bidding</span>);
  else if (table.phase === "estimation") parts.push(<span key="phase">Estimating</span>);
  else {
    if (table.total !== null && table.over !== null) {
      parts.push(
        <span key="over" className="contract__over">
          {table.over ? `Over by ${table.total - 13}` : `Under by ${13 - table.total}`}
        </span>,
      );
    }
    const trick = frame.event?.trick_winner != null ? table.tricks_played : Math.min(table.tricks_played + 1, 13);
    parts.push(
      <span key="trick">
        Trick <b>{trick}</b> of 13
      </span>,
    );
  }
  return (
    <div className="contract">
      {parts.map((part, i) => (
        <span key={i} className="contract__part">
          {i > 0 && <span className="contract__sep" aria-hidden />}
          {part}
        </span>
      ))}
    </div>
  );
}
