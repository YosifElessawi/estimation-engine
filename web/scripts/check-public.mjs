// Fails the public build (`npm run build:public`) if the page in out/ carries anything of the
// full table: the lobby, ratings, watching, reviews, the Why panel or an internal bot name.
// (The game screen's step-back and review calls stay in the bundle as dead paths; the public
// server has no such routes, which tests/server/test_public.py checks.)

import { readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const out = resolve(dirname(fileURLToPath(import.meta.url)), "..", "out");

const FORBIDDEN = [
  "Leaderboard",
  "Measuring sticks",
  "RL checkpoints",
  "Past games",
  "Review with every hand",
  "Reviewing",
  "Watching",
  "Elo ",
  "unrated",
  "CardPlayNet",
  "BiddingNet",
  "HeuristicBot",
  "hybrid",
  "pimc",
  "/history",
  "/bots",
  "replay this game",
];

function* files(folder) {
  for (const name of readdirSync(folder)) {
    const path = join(folder, name);
    if (statSync(path).isDirectory()) yield* files(path);
    else if (/\.(html|js|txt|json)$/.test(name)) yield path;
  }
}

const found = [];
for (const path of files(out)) {
  const text = readFileSync(path, "utf8");
  for (const word of FORBIDDEN) if (text.includes(word)) found.push(`${relative(out, path)}: "${word}"`);
}
if (found.length > 0) {
  console.error(`The public page carries full-table text:\n  ${found.join("\n  ")}`);
  process.exit(1);
}
console.log(`Public page checked: nothing of the full table in ${out}`);
