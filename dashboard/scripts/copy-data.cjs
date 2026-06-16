#!/usr/bin/env node
/* eslint-disable @typescript-eslint/no-require-imports */
/**
 * Copy pre-rendered analytics artefacts from `data/output/` into
 * `dashboard/public/data/` so Next.js can serve them as static files.
 *
 * Run automatically by `predev` / `prebuild`. Idempotent.
 */
const fs = require("node:fs");
const path = require("node:path");

const repoRoot = path.resolve(__dirname, "..", "..");
const src = path.join(repoRoot, "data", "output");
const dst = path.join(__dirname, "..", "public", "data");

const sources = [
  { kind: "file", rel: "analytics.json" },
  { kind: "file", rel: "analytics.db", optional: true },
  { kind: "dir", rel: "frames" },
  { kind: "dir", rel: "heatmaps" },
  { kind: "dir", rel: "annotated" },
  { kind: "dir", rel: "persons" },
  { kind: "dir", rel: "alerts" },
];

function rmrf(target) {
  if (fs.existsSync(target)) fs.rmSync(target, { recursive: true, force: true });
}

function copyFile(srcFile, dstFile) {
  fs.mkdirSync(path.dirname(dstFile), { recursive: true });
  fs.copyFileSync(srcFile, dstFile);
}

function copyDir(srcDir, dstDir) {
  fs.mkdirSync(dstDir, { recursive: true });
  for (const entry of fs.readdirSync(srcDir, { withFileTypes: true })) {
    if (entry.name === ".gitkeep") continue;
    const s = path.join(srcDir, entry.name);
    const d = path.join(dstDir, entry.name);
    if (entry.isDirectory()) copyDir(s, d);
    else copyFile(s, d);
  }
}

rmrf(dst);
fs.mkdirSync(dst, { recursive: true });

let copied = 0;
let skipped = 0;
for (const { kind, rel, optional } of sources) {
  const s = path.join(src, rel);
  const d = path.join(dst, rel);
  if (!fs.existsSync(s)) {
    if (optional) {
      console.warn(`! skipping ${rel} (optional, not present)`);
    } else {
      console.warn(`! missing ${rel} — dashboard panel for it will be empty`);
    }
    skipped++;
    continue;
  }
  if (kind === "file") copyFile(s, d);
  else copyDir(s, d);
  copied++;
}

console.log(`copy-data: ${copied} sources copied, ${skipped} skipped`);
