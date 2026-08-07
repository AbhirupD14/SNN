/**
 * Support module for scripts/record_simulation_demos.mjs.
 *
 * Keeps three concerns out of the recording script so the shot plans stay readable:
 *   1. Uvicorn server lifecycle, including the `.claude/dashboard_seed.txt` backup/restore
 *      contract and the "never kill a server we did not start" rule.
 *   2. A tiny keep-alive REST client for the dashboard's deterministic control endpoints.
 *   3. FFmpeg/FFprobe conversion + validation of the recorded WebM files.
 */

import { spawn, execFile } from 'node:child_process';
import { promises as fs } from 'node:fs';
import { createServer } from 'node:net';
import http from 'node:http';
import path from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ----------------------------------------------------------------- REST client
// One keep-alive agent for the whole run: the long-dwell fast-forward issues thousands of
// sequential POST /api/step calls and a fresh socket per call would dominate the runtime.
export class DashboardApi {
  constructor(port, host = '127.0.0.1') {
    this.port = port;
    this.host = host;
    this.agent = new http.Agent({ keepAlive: true, maxSockets: 1 });
  }

  base() { return `http://${this.host}:${this.port}`; }

  request(method, urlPath, body) {
    const payload = body === undefined ? null : JSON.stringify(body);
    const options = {
      host: this.host, port: this.port, path: urlPath, method,
      agent: this.agent,
      headers: payload ? { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) } : {},
    };
    return new Promise((resolve, reject) => {
      const req = http.request(options, (res) => {
        const chunks = [];
        res.on('data', (c) => chunks.push(c));
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8');
          if (res.statusCode >= 400) {
            reject(new Error(`${method} ${urlPath} -> HTTP ${res.statusCode}: ${text.slice(0, 400)}`));
            return;
          }
          try { resolve(text ? JSON.parse(text) : null); }
          catch { resolve(text); }
        });
      });
      req.on('error', reject);
      if (payload) req.write(payload);
      req.end();
    });
  }

  get(p) { return this.request('GET', p); }
  post(p, body) { return this.request('POST', p, body); }

  state() { return this.get('/api/state'); }
  config(overrides) { return this.post('/api/config', { overrides }); }
  pattern(name) { return this.post('/api/pattern', { name }); }
  patchPattern(row, col, name) { return this.post('/api/patch_pattern', { row, col, name }); }
  clearPatchPatterns() { return this.post('/api/patch_patterns/clear'); }
  start() { return this.post('/api/start'); }
  pause() { return this.post('/api/pause'); }
  step() { return this.post('/api/step'); }
  reset() { return this.post('/api/reset'); }
  speed(sps) { return this.post(`/api/speed/${sps}`); }
  topology() { return this.get('/api/topology'); }

  /** Advance `n` boundaries as fast as the server will accept them, with the runner paused.
   *  This is how the long-dwell video compresses its uninformative middle: the boundaries
   *  genuinely execute (the timestep readout proves it), they are just not shown at 1x. */
  async fastForward(n, onProgress) {
    const t0 = Date.now();
    for (let i = 1; i <= n; i++) {
      await this.step();
      if (onProgress && i % 100 === 0) await onProgress(i, n);
    }
    return (Date.now() - t0) / 1000;
  }

  /** Poll until `predicate(state)` holds. Throws with context on timeout -- the recording
   *  waits on explicit engine state, never on a bare sleep. */
  async waitFor(predicate, { timeout = 240000, interval = 100, what = 'state predicate' } = {}) {
    const deadline = Date.now() + timeout;
    let last = null;
    for (;;) {
      try { last = await this.state(); } catch (e) { last = { error: String(e) }; }
      if (last && !last.error && predicate(last)) return last;
      if (Date.now() > deadline) {
        throw new Error(`timed out after ${timeout}ms waiting for ${what}; ` +
          `last timestep=${last?.dynamic?.timestep} running=${last?.dynamic?.running} ` +
          `nodes=${last?.topology?.neurons?.length}`);
      }
      await sleep(interval);
    }
  }
}

// ------------------------------------------------------------- server lifecycle
export async function portInUse(port, host = '127.0.0.1') {
  return new Promise((resolve) => {
    const srv = createServer();
    srv.once('error', (e) => resolve(e.code === 'EADDRINUSE'));
    srv.once('listening', () => srv.close(() => resolve(false)));
    srv.listen(port, host);
  });
}

export async function findFreePort(start, host = '127.0.0.1') {
  for (let p = start; p < start + 200; p++) {
    if (!(await portInUse(p, host))) return p;
  }
  throw new Error(`no free port found in [${start}, ${start + 200})`);
}

/**
 * The `.claude/dashboard_seed.txt` contract.
 *
 * `backend.api` reads that file ONCE at import time to build its engine, so a recording that
 * needs a specific seed must write the file before starting its own Uvicorn. The prior
 * contents (including "file did not exist") are captured here and restored by `restore()`,
 * which the recorder calls from a `finally` block. User runtime state is never destroyed.
 */
export class SeedFile {
  constructor(repoRoot) {
    this.file = path.join(repoRoot, '.claude', 'dashboard_seed.txt');
    this.captured = false;
    this.existed = false;
    this.previous = null;
  }

  async capture() {
    try {
      this.previous = await fs.readFile(this.file, 'utf8');
      this.existed = true;
    } catch (e) {
      if (e.code !== 'ENOENT') throw e;
      this.existed = false;
      this.previous = null;
    }
    this.captured = true;
    return { existed: this.existed, previous: this.previous };
  }

  async write(seed) {
    if (!this.captured) throw new Error('SeedFile.capture() must run before write()');
    await fs.mkdir(path.dirname(this.file), { recursive: true });
    await fs.writeFile(this.file, String(seed), 'utf8');
  }

  /** Restore byte-for-byte, or remove the file if it did not exist before. */
  async restore() {
    if (!this.captured) return { restored: false, reason: 'never captured' };
    if (this.existed) {
      await fs.writeFile(this.file, this.previous, 'utf8');
      return { restored: true, existed: true, bytes: Buffer.byteLength(this.previous) };
    }
    await fs.rm(this.file, { force: true });
    return { restored: true, existed: false };
  }
}

export class UvicornServer {
  constructor({ repoRoot, port, python, logPath }) {
    this.repoRoot = repoRoot;
    this.port = port;
    this.python = python;
    this.logPath = logPath;
    this.proc = null;
    this.startedByUs = false;
  }

  async start() {
    const log = await fs.open(this.logPath, 'w');
    this.proc = spawn(
      this.python, ['-m', 'uvicorn', 'backend.api:app', '--host', '127.0.0.1',
        '--port', String(this.port), '--log-level', 'warning'],
      {
        cwd: this.repoRoot,
        env: { ...process.env, PYTHONPATH: this.repoRoot, PYTHONUNBUFFERED: '1' },
        stdio: ['ignore', log.fd, log.fd],
      });
    this.startedByUs = true;
    this.proc.on('exit', (code) => { this.exitCode = code; });
    await log.close();
    return this.proc.pid;
  }

  async waitReady(api, timeout = 60000) {
    const deadline = Date.now() + timeout;
    for (;;) {
      if (this.proc && this.exitCode != null) {
        const tail = await fs.readFile(this.logPath, 'utf8').catch(() => '');
        throw new Error(`uvicorn exited with code ${this.exitCode}\n${tail.slice(-2000)}`);
      }
      try {
        const s = await api.state();
        if (s && s.topology) return s;
      } catch { /* not up yet */ }
      if (Date.now() > deadline) {
        const tail = await fs.readFile(this.logPath, 'utf8').catch(() => '');
        throw new Error(`server did not become ready on port ${this.port}\n${tail.slice(-2000)}`);
      }
      await sleep(200);
    }
  }

  /** Stop ONLY a process this workflow started. */
  async stop() {
    if (!this.startedByUs || !this.proc || this.exitCode != null) return false;
    this.proc.kill('SIGTERM');
    for (let i = 0; i < 50 && this.exitCode == null; i++) await sleep(100);
    if (this.exitCode == null) this.proc.kill('SIGKILL');
    return true;
  }
}

// --------------------------------------------------------------- ffmpeg/ffprobe
export async function toolVersion(bin) {
  try {
    const { stdout } = await execFileAsync(bin, ['-version']);
    return stdout.split('\n')[0].trim();
  } catch (e) {
    throw new Error(`${bin} is not available on PATH: ${e.message}`);
  }
}

export async function convertToMp4(input, output, { fps = 30, width = 1920, height = 1080 } = {}) {
  await execFileAsync('ffmpeg', [
    '-y', '-i', input,
    '-vf', `fps=${fps},scale=${width}:${height}:flags=lanczos,setsar=1`,
    '-c:v', 'libx264', '-preset', 'slow', '-crf', '20',
    '-pix_fmt', 'yuv420p', '-r', String(fps),
    '-movflags', '+faststart',
    output,
  ], { maxBuffer: 32 * 1024 * 1024 });
  return output;
}

export async function probe(file) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-show_streams', '-show_format', '-of', 'json', file,
  ], { maxBuffer: 32 * 1024 * 1024 });
  const data = JSON.parse(stdout);
  const v = (data.streams || []).find((s) => s.codec_type === 'video');
  if (!v) throw new Error(`${file}: no video stream`);
  const [num, den] = String(v.r_frame_rate || '0/1').split('/').map(Number);
  return {
    file,
    codec: v.codec_name,
    pix_fmt: v.pix_fmt,
    width: v.width,
    height: v.height,
    fps: den ? num / den : 0,
    nb_frames: v.nb_frames ? Number(v.nb_frames) : null,
    duration: Number(v.duration ?? data.format?.duration ?? 0),
    size_bytes: Number(data.format?.size ?? 0),
  };
}

/** Decode one frame to prove the container actually opens. */
export async function decodeCheck(file) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-select_streams', 'v:0', '-count_packets',
    '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', file,
  ], { maxBuffer: 32 * 1024 * 1024 });
  const packets = Number(String(stdout).trim());
  if (!Number.isFinite(packets) || packets <= 0) {
    throw new Error(`${file}: decoded 0 packets`);
  }
  return packets;
}

export async function extractFrame(video, seconds, output, { width = 1920, height = 1080 } = {}) {
  await execFileAsync('ffmpeg', [
    '-y', '-ss', String(seconds), '-i', video, '-frames:v', '1',
    '-vf', `scale=${width}:${height}:flags=lanczos`,
    output,
  ], { maxBuffer: 32 * 1024 * 1024 });
  return output;
}

/**
 * Candidate click points for the 3D scene, ordered brightest-first.
 *
 * The renderer exposes no DOM handle for selecting a neuron: `_handleClick` raycasts from
 * the pointer position onto the visible neuron meshes, so the only way to select one is to
 * click where a neuron actually is. Neuron bodies are emissive spheres and are by far the
 * brightest thing drawn (edges sit at <= 0.36 opacity against the #0b0e14 background), so
 * scanning a screenshot for bright blobs finds them without needing the camera projection.
 *
 * Returns `[{x, y, lum}]` in CSS pixels relative to the clipped region's origin, one entry
 * per `cell`-sized bucket, sorted by descending peak luminance.
 */
export async function brightSpots(pngPath, {
  cell = 10, minLum = 60, minSat = 0.35, minDist = 22, limit = 160,
} = {}) {
  const { width, height } = await pngSize(pngPath);
  const { stdout } = await execFileAsync(
    'ffmpeg', ['-v', 'error', '-i', pngPath, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    { encoding: 'buffer', maxBuffer: 512 * 1024 * 1024 });
  const rgb = stdout;
  if (rgb.length < width * height * 3) {
    throw new Error(`decoded ${rgb.length} bytes, expected ${width * height * 3}`);
  }
  const cols = Math.ceil(width / cell), rows = Math.ceil(height / cell);
  const best = new Float32Array(cols * rows);
  const bestX = new Int32Array(cols * rows);
  const bestY = new Int32Array(cols * rows);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 3;
      const r = rgb[i], g = rgb[i + 1], b = rgb[i + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      // Rec. 601 luma: "brighter than the dark #0b0e14 background".
      const lum = 0.299 * r + 0.587 * g + 0.114 * b;
      if (lum < minLum) continue;
      // Saturation gate. Neuron bodies are emissive and strongly coloured (teal, gold,
      // pink, sky, purple); the in-scene legend and hint text are near-neutral greys
      // (#e7ecf5, #9aa6bd) and would otherwise dominate the brightness ranking with
      // hundreds of unclickable points.
      const sat = mx === 0 ? 0 : (mx - mn) / mx;
      if (sat < minSat) continue;
      const c = ((y / cell) | 0) * cols + ((x / cell) | 0);
      const score = lum * sat;
      if (score > best[c]) { best[c] = score; bestX[c] = x; bestY[c] = y; }
    }
  }
  const ranked = [];
  for (let c = 0; c < best.length; c++) {
    if (best[c] > 0) ranked.push({ x: bestX[c], y: bestY[c], score: best[c] });
  }
  ranked.sort((a, b) => b.score - a.score);
  // Spread the candidates: without this, one large bright blob supplies every point and the
  // caller wastes its whole click budget re-selecting the same neuron.
  const out = [];
  for (const p of ranked) {
    if (out.every((q) => Math.hypot(q.x - p.x, q.y - p.y) >= minDist)) out.push(p);
    if (out.length >= limit) break;
  }
  return out;
}

export async function pngSize(file) {
  const { stdout } = await execFileAsync('ffprobe', [
    '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height', '-of', 'csv=p=0', file,
  ]);
  const [w, h] = String(stdout).trim().split(',').map(Number);
  return { width: w, height: h };
}
