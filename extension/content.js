(() => {
  if (window.__lookTwiceLoaded) return;
  window.__lookTwiceLoaded = true;

  const CFG = {
    intervalMs: 500,
    window: 10,
    minSamples: 3,
    noFaceLimit: 4,
    maxSide: 640,
    jpegQuality: 0.8,
    minVideoSide: 160,
  };

  const LABELS = {
    analyzing: { icon: "…", text: "Analyzing" },
    authentic: { icon: "✓", text: "Likely authentic" },
    uncertain: { icon: "?", text: "Uncertain — look twice" },
    manipulated: { icon: "!", text: "Likely deepfake" },
    no_face: { icon: "–", text: "No face detected" },
    protected: { icon: "🔒", text: "Can't analyze this video" },
    offline: { icon: "×", text: "LookTwice backend offline" },
  };

  const states = new Map();
  let enabled = true;
  let timer = null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: false });

  function stateFor(video) {
    let s = states.get(video);
    if (!s) {
      const el = document.createElement("div");
      el.className = "looktwice-badge";
      el.setAttribute("role", "status");
      el.setAttribute("aria-live", "polite");
      el.innerHTML = '<span class="lt-icon"></span><span class="lt-text"></span><span class="lt-score"></span>';
      document.documentElement.appendChild(el);
      s = { el, probs: [], noFace: 0, busy: false, status: "analyzing", avg: null, thresholds: null };
      states.set(video, s);
    }
    return s;
  }

  function removeState(video) {
    const s = states.get(video);
    if (s) s.el.remove();
    states.delete(video);
  }

  function render(video, s) {
    const r = video.getBoundingClientRect();
    const notStarted = s.status === "analyzing" && !s.probs.length && (video.paused || video.ended);
    const visible = enabled && !notStarted && r.width >= CFG.minVideoSide && r.height >= CFG.minVideoSide / 2 &&
      r.bottom > 0 && r.right > 0 && r.top < innerHeight && r.left < innerWidth;
    s.el.hidden = !visible;
    if (!visible) return;
    s.el.style.top = `${Math.max(r.top, 0) + 8}px`;
    s.el.style.left = `${Math.max(r.left, 0) + 8}px`;
    const label = LABELS[s.status] || LABELS.analyzing;
    s.el.dataset.status = s.status;
    s.el.querySelector(".lt-icon").textContent = label.icon;
    s.el.querySelector(".lt-text").textContent = label.text;
    s.el.querySelector(".lt-score").textContent =
      s.avg !== null && ["authentic", "uncertain", "manipulated"].includes(s.status)
        ? `fake ${Math.round(s.avg * 100)}%` : "";
    s.el.title = s.thresholds
      ? `Average of last ${s.probs.length} frames. Uncertain band ${s.thresholds.lo.toFixed(2)}–${s.thresholds.hi.toFixed(2)}.`
      : "";
  }

  function decide(s) {
    if (s.noFace >= CFG.noFaceLimit) return "no_face";
    if (s.probs.length < CFG.minSamples || !s.thresholds) return "analyzing";
    const { lo, hi } = s.thresholds;
    if (s.avg >= hi) return "manipulated";
    if (s.avg < lo) return "authentic";
    return "uncertain";
  }

  function capture(video) {
    const scale = Math.min(1, CFG.maxSide / Math.max(video.videoWidth, video.videoHeight));
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", CFG.jpegQuality);
  }

  async function analyze(video, s) {
    if (s.busy || video.paused || video.ended || video.readyState < 2 || !video.videoWidth) return;
    if (document.visibilityState !== "visible") return;
    let image, captureMs;
    try {
      const t0 = performance.now();
      image = capture(video);
      captureMs = performance.now() - t0;
    } catch (e) {
      s.status = "protected";
      return;
    }
    s.busy = true;
    try {
      const resp = await chrome.runtime.sendMessage({ type: "predict", image, captureMs });
      if (!resp || !resp.ok) {
        s.status = "offline";
        return;
      }
      const res = resp.result;
      s.thresholds = res.thresholds;
      if (res.face_found) {
        s.noFace = 0;
        s.probs.push(res.prob);
        if (s.probs.length > CFG.window) s.probs.shift();
        s.avg = s.probs.reduce((a, b) => a + b, 0) / s.probs.length;
      } else {
        s.noFace += 1;
      }
      s.status = decide(s);
    } catch (e) {
      s.status = "offline";
    } finally {
      s.busy = false;
    }
  }

  function tick() {
    const videos = new Set(document.querySelectorAll("video"));
    for (const v of [...states.keys()]) if (!videos.has(v) || !v.isConnected) removeState(v);
    for (const v of videos) {
      const r = v.getBoundingClientRect();
      if (r.width < CFG.minVideoSide && !states.has(v)) continue;
      const s = stateFor(v);
      if (enabled) analyze(v, s);
      render(v, s);
    }
  }

  function reposition() {
    for (const [v, s] of states) render(v, s);
  }

  function start() {
    if (!timer) timer = setInterval(tick, CFG.intervalMs);
  }

  function setEnabled(on) {
    enabled = on;
    if (!on) for (const v of [...states.keys()]) removeState(v);
    on ? start() : (clearInterval(timer), (timer = null));
  }

  addEventListener("scroll", reposition, { passive: true, capture: true });
  addEventListener("resize", reposition, { passive: true });
  chrome.storage.sync.get({ enabled: true }).then(({ enabled: on }) => setEnabled(on));
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === "sync" && changes.enabled) setEnabled(changes.enabled.newValue);
  });

  window.__lookTwice = { states, CFG, decide, tick, setEnabled };
})();
