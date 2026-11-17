const DEFAULTS = { enabled: true, backendUrl: "http://localhost:8000" };
const $ = (id) => document.getElementById(id);

const pct = (arr, q) => {
  if (!arr.length) return null;
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((q / 100) * s.length))];
};
const fmt = (v) => (v === null || v === undefined ? "–" : `${v.toFixed(0)} ms`);

async function showHealth() {
  const el = $("health");
  const h = await chrome.runtime.sendMessage({ type: "health" });
  el.className = `status ${h.ok ? "ok" : "err"}`;
  $("healthText").textContent = h.ok
    ? `Connected · ${h.model || "model"} · ${h.detector}`
    : `Backend not reachable (${h.error || "error"})`;
}

async function showStats() {
  const { lt_stats: s } = await chrome.storage.local.get("lt_stats");
  const st = s || { rtt: [], capture: [], server: [], frames: 0, noFace: 0, errors: 0 };
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);
  const rows = [
    ["Frames sent", String(st.frames)],
    ["No face", String(st.noFace)],
    ["Errors", String(st.errors)],
    ["Round trip mean / p95", `${fmt(mean(st.rtt))} / ${fmt(pct(st.rtt, 95))}`],
    ["Server mean / p95", `${fmt(mean(st.server))} / ${fmt(pct(st.server, 95))}`],
    ["Capture mean / p95", `${fmt(mean(st.capture))} / ${fmt(pct(st.capture, 95))}`],
  ];
  $("stats").replaceChildren(...rows.map(([k, v]) => {
    const tr = document.createElement("tr");
    tr.innerHTML = "<td></td><td></td>";
    tr.children[0].textContent = k;
    tr.children[1].textContent = v;
    return tr;
  }));
}

(async () => {
  const cfg = await chrome.storage.sync.get(DEFAULTS);
  $("enabled").checked = cfg.enabled;
  $("backendUrl").value = cfg.backendUrl;
  $("enabled").addEventListener("change", (e) => chrome.storage.sync.set({ enabled: e.target.checked }));
  $("backendUrl").addEventListener("change", async (e) => {
    await chrome.storage.sync.set({ backendUrl: e.target.value.trim() || DEFAULTS.backendUrl });
    showHealth();
  });
  $("reset").addEventListener("click", async () => {
    await chrome.runtime.sendMessage({ type: "resetStats" });
    showStats();
  });
  showHealth();
  showStats();
  setInterval(showStats, 1000);
})();
