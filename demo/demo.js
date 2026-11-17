const STORE = "looktwice_answers_v1";
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);

function loadAll() {
  try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch { return {}; }
}
function saveAll(all) {
  try { localStorage.setItem(STORE, JSON.stringify(all)); } catch {}
}
let answers = loadAll();
const sessionKey = () => `${$("pid").value.trim()}|${$("set").value}|${$("phase").value}`;

async function fetchClips(set) {
  const r = await fetch(`/api/clips?set=${encodeURIComponent(set)}`);
  return r.ok ? (await r.json()).clips : [];
}

function updateProgress(clips) {
  const prefix = sessionKey();
  const done = clips.filter((c) => {
    const a = answers[`${prefix}|${c.name}`];
    return a && a.answer && a.confidence;
  }).length;
  $("progress").textContent = `${done} of ${clips.length} answered`;
}

async function render() {
  const set = $("set").value;
  const clips = await fetchClips(set);
  const host = $("clips");
  host.replaceChildren();
  $("empty").hidden = clips.length > 0;
  $("empty").textContent = `No clips in demo/clips/${set}/ yet. Put neutral-named .mp4 files there and reload.`;

  clips.forEach((clip, i) => {
    const node = $("clip-tpl").content.firstElementChild.cloneNode(true);
    const title = node.querySelector(".clip-title");
    title.textContent = `Clip ${i + 1}`;
    const check = document.createElement("span");
    check.className = "check";
    title.appendChild(check);
    const video = node.querySelector("video");
    video.src = clip.url;
    video.setAttribute("aria-label", `Clip ${i + 1}`);

    const [qAnswer, qConf] = node.querySelectorAll("fieldset");
    qAnswer.querySelectorAll("input").forEach((el) => (el.name = `answer-${i}`));
    qConf.querySelectorAll("input").forEach((el) => (el.name = `conf-${i}`));

    const key = () => `${sessionKey()}|${clip.name}`;
    const restore = () => {
      const a = answers[key()] || {};
      qAnswer.querySelectorAll("input").forEach((el) => (el.checked = el.value === a.answer));
      qConf.querySelectorAll("input").forEach((el) => (el.checked = el.value === String(a.confidence)));
      const complete = a.answer && a.confidence;
      node.classList.toggle("done", Boolean(complete));
      check.textContent = complete ? "✓ answered" : "";
    };
    node.addEventListener("change", () => {
      if (!$("pid").value.trim()) {
        alert("Please fill in the Participant ID first.");
        restore();
        $("pid").focus();
        return;
      }
      const a = qAnswer.querySelector("input:checked")?.value || null;
      const c = qConf.querySelector("input:checked")?.value || null;
      answers[key()] = {
        participant: $("pid").value.trim(), set, phase: $("phase").value, clip: clip.name,
        position: i + 1, answer: a, confidence: c ? Number(c) : null, watched_s: Math.round(video.currentTime),
        saved_at: new Date().toISOString(),
      };
      saveAll(answers);
      restore();
      updateProgress(clips);
    });
    node.restore = restore;
    restore();
    host.appendChild(node);
  });
  updateProgress(clips);
  render.clips = clips;
}

function refreshAnswers() {
  document.querySelectorAll(".clip").forEach((n) => n.restore && n.restore());
  updateProgress(render.clips || []);
}

function downloadCsv() {
  const rows = Object.values(answers);
  if (!rows.length) return alert("No answers yet.");
  const cols = ["participant", "set", "phase", "clip", "position", "answer", "confidence", "watched_s", "saved_at"];
  const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const csv = [cols.join(","), ...rows.map((r) => cols.map((c) => esc(r[c])).join(","))].join("\n");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
  a.download = `looktwice_answers_${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function clearSession() {
  const prefix = sessionKey() + "|";
  if (!confirm(`Delete the answers of ${prefix.slice(0, -1).replaceAll("|", " / ")}?`)) return;
  for (const k of Object.keys(answers)) if (k.startsWith(prefix)) delete answers[k];
  saveAll(answers);
  refreshAnswers();
}

for (const [k, id] of [["pid", "pid"], ["set", "set"], ["phase", "phase"]]) {
  if (params.get(k)) $(id).value = params.get(k);
}
$("set").addEventListener("change", render);
$("phase").addEventListener("change", refreshAnswers);
$("pid").addEventListener("input", refreshAnswers);
$("download").addEventListener("click", downloadCsv);
$("clear").addEventListener("click", clearSession);
$("session").addEventListener("submit", (e) => e.preventDefault());
render();
