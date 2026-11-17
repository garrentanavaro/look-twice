const DEFAULTS = { enabled: true, backendUrl: "http://localhost:8000" };
const STATS_KEY = "lt_stats";
const STATS_CAP = 500;

async function settings() {
  return chrome.storage.sync.get(DEFAULTS);
}

async function recordStats(entry) {
  const { [STATS_KEY]: s = { rtt: [], capture: [], server: [], frames: 0, noFace: 0, errors: 0 } } =
    await chrome.storage.local.get(STATS_KEY);
  s.frames += 1;
  if (entry.error) s.errors += 1;
  if (entry.noFace) s.noFace += 1;
  for (const k of ["rtt", "capture", "server"]) {
    if (typeof entry[k] === "number") {
      s[k].push(Math.round(entry[k] * 10) / 10);
      if (s[k].length > STATS_CAP) s[k].shift();
    }
  }
  await chrome.storage.local.set({ [STATS_KEY]: s });
}

async function predict(msg) {
  const { backendUrl } = await settings();
  const t0 = performance.now();
  try {
    const blob = await (await fetch(msg.image)).blob();
    const r = await fetch(`${backendUrl.replace(/\/$/, "")}/predict`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg" },
      body: blob,
    });
    if (!r.ok) throw new Error(`backend HTTP ${r.status}`);
    const result = await r.json();
    const rtt = performance.now() - t0;
    recordStats({ rtt, capture: msg.captureMs, server: result.latency_ms?.total, noFace: !result.face_found });
    return { ok: true, result, rtt };
  } catch (e) {
    recordStats({ error: true, capture: msg.captureMs });
    return { ok: false, error: String(e.message || e) };
  }
}

async function health() {
  const { backendUrl } = await settings();
  try {
    const r = await fetch(`${backendUrl.replace(/\/$/, "")}/health`);
    return { ok: r.ok, ...(await r.json()) };
  } catch (e) {
    return { ok: false, error: String(e.message || e) };
  }
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  const handlers = {
    predict: () => predict(msg),
    health: () => health(),
    resetStats: () => chrome.storage.local.remove(STATS_KEY).then(() => ({ ok: true })),
  };
  const h = handlers[msg?.type];
  if (!h) return false;
  h().then(sendResponse);
  return true;
});

if (typeof module !== "undefined") module.exports = { predict, health, recordStats, DEFAULTS, STATS_KEY };
