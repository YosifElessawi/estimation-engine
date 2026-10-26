// Starts the web table: the FastAPI server on a free local port, then `next dev` with /api
// proxied to it. Run it through portless (`portless run` in web/), which sets PORT for Next.
// Stopping either process stops both.

import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const web = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const root = resolve(web, "..");

function freePort() {
  return new Promise((done, fail) => {
    const probe = createServer();
    probe.once("error", fail);
    probe.listen(0, "127.0.0.1", () => {
      const { port } = probe.address();
      probe.close(() => done(port));
    });
  });
}

async function waitForApi(url, child, seconds) {
  const deadline = Date.now() + seconds * 1000;
  while (Date.now() < deadline) {
    if (child.exitCode !== null) throw new Error("the API server exited during startup");
    try {
      const response = await fetch(url);
      if (response.ok) return;
    } catch {
      // not listening yet
    }
    await new Promise((next) => setTimeout(next, 250));
  }
  throw new Error(`the API server did not answer ${url} within ${seconds} s`);
}

const children = [];
let stopping = false;

function stopAll(code) {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.exitCode === null) child.kill("SIGTERM");
  }
  setTimeout(() => process.exit(code), 500).unref();
}

for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) {
  process.on(signal, () => stopAll(0));
}

const apiPort = await freePort();
const apiUrl = `http://127.0.0.1:${apiPort}`;
const api = spawn("uv", ["run", "python", "-m", "server", "--port", String(apiPort)], {
  cwd: root,
  stdio: "inherit",
  env: { ...process.env, PORT: String(apiPort) },
});
children.push(api);
api.on("exit", (code) => {
  if (!stopping) console.error(`[estimation] API server exited (${code}); stopping`);
  stopAll(code ?? 1);
});

try {
  await waitForApi(`${apiUrl}/api/bots`, api, 120);
} catch (error) {
  console.error(`[estimation] ${error.message}`);
  stopAll(1);
}

if (!stopping) {
  console.log(`[estimation] API ready at ${apiUrl}`);
  const next = spawn("npx", ["next", "dev"], {
    cwd: web,
    stdio: "inherit",
    env: { ...process.env, ESTIMATION_API: apiUrl },
  });
  children.push(next);
  next.on("exit", (code) => stopAll(code ?? 0));
}
