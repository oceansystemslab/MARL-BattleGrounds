import { spawn } from "node:child_process";
import { once } from "node:events";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPOSITORY_ROOT = resolve(HERE, "../../../..");

/**
 * Start the real loopback debugger for browser integration tests.
 *
 * @param {{scenario?: string, extraArgs?: string[]}} options
 * @returns {Promise<{
 *   process: import("node:child_process").ChildProcess,
 *   url: string,
 * }>}
 */
export function startDebugger({ scenario = "arena_5v5", extraArgs = [] } = {}) {
  return new Promise((resolveUrl, reject) => {
    const child = spawn(
      "uv",
      [
        "run",
        "python",
        "-u",
        "scripts/dev/debug_renderer.py",
        "--ui",
        "browser",
        "--no-open",
        "--port",
        "0",
        "--scenario",
        scenario,
        ...extraArgs,
      ],
      {
        cwd: REPOSITORY_ROOT,
        env: process.env,
        stdio: ["ignore", "pipe", "pipe"],
      },
    );
    let settled = false;
    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Debugger startup timed out.\n${stderr}`));
    }, 60_000);

    child.once("error", (error) => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      reject(error);
    });
    child.stderr?.setEncoding("utf8");
    child.stderr?.on("data", (chunk) => {
      stderr += String(chunk);
    });
    child.stdout?.setEncoding("utf8");
    child.stdout?.on("data", (chunk) => {
      stdout += String(chunk);
      const match = stdout.match(
        /Visual debugger: (http:\/\/127\.0\.0\.1:\d+\/#token=[A-Za-z0-9_-]+)/,
      );
      if (!match || settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      resolveUrl({ process: child, url: match[1] });
    });
    child.once("exit", (code) => {
      clearTimeout(timeout);
      if (!settled) {
        settled = true;
        reject(
          new Error(`Debugger exited before startup with code ${code}.\n${stderr}`),
        );
      }
    });
  });
}

/**
 * Stop one loopback debugger process without relying on browser-close signals.
 *
 * @param {import("node:child_process").ChildProcess | null} child
 */
export async function stopDebugger(child) {
  if (!child || child.exitCode !== null) {
    return;
  }
  child.kill("SIGINT");
  await Promise.race([
    once(child, "exit"),
    new Promise((resolveDelay) => setTimeout(resolveDelay, 5_000)),
  ]);
  if (child.exitCode === null) {
    child.kill("SIGKILL");
    await once(child, "exit");
  }
}
