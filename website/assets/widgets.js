// Two small widgets, no dependencies. Images live in data/, see tools/build_assets.py.
(function () {
  const $ = (id) => document.getElementById(id);

  // ---------- forgetting slider ----------
  const fw = $("forget-widget");
  if (fw) {
    // seed per concept that the paper shows, then all twelve in order
    // seeds from tools/pick_seeds.py: CIDM visibly changes on them, ours is most stable
    const SEEDS = { 0: [11, 7, 6, 3], 1: [8, 3, 9, 2], 2: [10, 7, 3, 9] }, NAMES = ["dog", "duck toy", "cat"];
    const REF = ["dog", "duck_toy", "cat"];
    let c = 0, si = 0, seed = SEEDS[0][0], timer = null;
    const k = $("fw-k");
    const src = (m, kk) => `data/forget/${m}/c${c}/s${seed}/k${kk}.jpg`;
    function preload() {
      for (let kk = c; kk <= 9; kk++) for (const m of ["ours", "cidm", "ft"]) new Image().src = src(m, kk);
    }
    function draw() {
      const kk = Math.max(+k.value, c);
      k.value = kk;
      for (const m of ["ours", "cidm", "ft"]) $("fw-" + m).src = src(m, kk);
      $("fw-ref").querySelectorAll("img").forEach((im, i) => (im.src = `data/forget/ref/${REF[c]}_${i}.jpg`));
      $("fw-k-val").textContent = kk + 1;
      $("fw-note").textContent =
        `The ${NAMES[c]} is learned in task ${c + 1}. All three methods use the same prompt and the same initial noise.`;
    }
    function setConcept(cc) {
      c = cc; si = 0; seed = SEEDS[cc][0]; k.min = cc; k.value = 9; preload(); draw();
    }
    fw.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      fw.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
      setConcept(+b.dataset.c);
    }));
    $("fw-seed").addEventListener("click", () => { si = (si + 1) % SEEDS[c].length; seed = SEEDS[c][si]; preload(); draw(); });
    k.addEventListener("input", draw);
    $("fw-play").addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; $("fw-play").innerHTML = "&#9654;&#xFE0E;"; return; }
      $("fw-play").innerHTML = "&#10074;&#10074;&#xFE0E;";
      k.value = c; draw();
      timer = setInterval(() => {
        if (+k.value >= 9) { clearInterval(timer); timer = null; $("fw-play").innerHTML = "&#9654;&#xFE0E;"; return; }
        k.value = +k.value + 1; draw();
      }, 650);
    });
    setConcept(0);
  }

  // ---------- placement picker (the images of Figure 7) ----------
  const pw = $("place-widget");
  if (pw) {
    let boxes = null, concept = "teddybear", quad = "TL", front = $("pw-a"), back = $("pw-b"), tour = null;
    const NAME = { teddybear: "teddy bear", ducktoy: "duck toy" };
    const QNAME = { TL: "top left", TR: "top right", BL: "bottom left", BR: "bottom right" };
    const ORDER = ["TL", "TR", "BR", "BL"];
    const stage = $("pw-stage"), box = $("pw-box");
    const img = (c, q) => `data/place/${c}_${q}.jpg`;
    const place = (el, r) => {
      el.style.left = r[0] * 100 + "%"; el.style.top = r[1] * 100 + "%";
      el.style.width = (r[2] - r[0]) * 100 + "%"; el.style.height = (r[3] - r[1]) * 100 + "%";
    };
    function go(q, first) {
      if (!boxes) return;
      const src = img(concept, q);
      quad = q;
      if (first) front.src = src;
      else if (!front.src.endsWith(src)) {
        // crossfade: load into the hidden layer, then swap the layers
        back.onload = () => { back.classList.add("top"); front.classList.remove("top"); [front, back] = [back, front]; };
        back.src = src;
      }
      place(box, boxes[concept][q].req);
      pw.querySelectorAll(".pw-picker button").forEach((x) => x.classList.toggle("on", x.dataset.q === q));
      $("pw-note").innerHTML = `<span>prompt</span> a photo of ${NAME[concept]} on a beach <span>box</span> ${QNAME[q]} quadrant`;
    }
    function stopTour() { if (tour) { clearInterval(tour); tour = null; pw.classList.remove("touring"); } }
    function startTour() {
      if (tour || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
      let i = ORDER.indexOf(quad), n = 0;
      pw.classList.add("touring");
      tour = setInterval(() => { i = (i + 1) % 4; go(ORDER[i]); if (++n >= 8) stopTour(); }, 1600);
    }
    pw.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => {
      stopTour();
      pw.querySelectorAll(".seg button").forEach((x) => x.classList.toggle("on", x === b));
      concept = b.dataset.c; go(quad);
    }));
    pw.querySelectorAll(".pw-picker button").forEach((b) => b.addEventListener("click", () => { stopTour(); go(b.dataset.q); }));
    fetch("data/place/boxes.json").then((r) => r.json()).then((j) => {
      boxes = j;
      for (const c in j) for (const q in j[c]) new Image().src = img(c, q);
      front.classList.add("top"); go(quad, true);
      // a short tour the first time the widget comes into view
      const io = new IntersectionObserver((es) => {
        if (es[0].isIntersecting) { startTour(); io.disconnect(); }
      }, { threshold: 0.6 });
      io.observe(stage);
    });
  }
})();

// ---------- forgetting bar chart ----------
(function () {
  const svg = document.getElementById("fc-svg"), tip = document.getElementById("fc-tip");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a, parent = svg) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); parent.appendChild(e); return e; };
  const D = [
    { name: "Ours", v: 0.0060, sd: 0.0031, ta: 75.6, col: "#111111", note: "one fixed-size network" },
    { name: "CIDM", v: 0.0232, sd: 0.0014, ta: 75.9, col: "#56b4e9", note: "stores an adapter per concept" },
    { name: "Fine-tuning", v: 0.1386, sd: 0.0042, ta: 75.5, col: "#0072b2", note: "sequential, same budget as ours" },
  ];
  const L = 112, R = 540, top = 22, row = 46, bar = 18, max = 0.15;
  const x = (v) => L + (v / max) * (R - L);
  // recessive grid and axis
  for (const g of [0, 0.05, 0.1, 0.15]) {
    el("line", { x1: x(g), x2: x(g), y1: top - 6, y2: top + row * 3 - 10, class: g ? "fc-grid" : "fc-axis" });
    el("text", { x: x(g), y: top + row * 3 + 6, class: "fc-tick", "text-anchor": "middle" }).textContent = g ? g.toFixed(2) : "0";
  }
  el("text", { x: R, y: top + row * 3 + 24, class: "fc-axlabel", "text-anchor": "end" }).textContent = "forgetting (DINO) · lower is better";
  D.forEach((d, i) => {
    const y = top + i * row + (row - bar) / 2 - 8, w = Math.max(3, x(d.v) - L);
    const g = el("g", { class: "fc-row", tabindex: 0 });
    el("rect", { x: 0, y: y - 12, width: 640, height: row, class: "fc-hit" }, g);
    el("text", { x: L - 12, y: y + bar / 2 + 4, class: "fc-name", "text-anchor": "end" }, g).textContent = d.name;
    // bar anchored at the baseline, rounded only at the data end
    el("path", { d: `M${L},${y} h${w - 4} a4,4 0 0 1 4,4 v${bar - 8} a4,4 0 0 1 -4,4 h${-(w - 4)} z`, fill: d.col, class: "fc-bar" }, g);
    el("line", { x1: x(d.v - d.sd), x2: x(d.v + d.sd), y1: y + bar / 2, y2: y + bar / 2, class: "fc-err" }, g);
    const lab = el("text", { x: x(d.v + d.sd) + 8, y: y + bar / 2 + 4, class: "fc-val" }, g);
    lab.textContent = d.v.toFixed(4);
    if (i > 0) {
      const t = el("tspan", { class: "fc-ratio", dx: 8 }, lab);
      t.textContent = `${Math.round(d.v / D[0].v)}× ours`;
    }
    const show = (ev) => {
      tip.innerHTML = `<b>${d.name}</b><br>forgetting ${d.v.toFixed(4)} ± ${d.sd.toFixed(4)}<br>text alignment ${d.ta}<br><span>${d.note}</span>`;
      tip.style.opacity = 1;
      const box = svg.getBoundingClientRect(), s = box.width / 640;
      tip.style.left = Math.min(box.width - 200, (x(d.v) + 12) * s) + "px";
      tip.style.top = (i === D.length - 1 ? y - 6 : y + bar + 6) * s + "px";
      tip.style.transform = i === D.length - 1 ? "translateY(-100%)" : "none";
      g.classList.add("hover");
    };
    const hide = () => { tip.style.opacity = 0; g.classList.remove("hover"); };
    g.addEventListener("mouseenter", show); g.addEventListener("mouseleave", hide);
    g.addEventListener("focus", show); g.addEventListener("blur", hide);
  });
})();

// ---------- generation cost vs number of concepts ----------
(function () {
  const svg = document.getElementById("cc-svg"), tip = document.getElementById("cc-tip");
  if (!svg) return;
  const NS = "http://www.w3.org/2000/svg";
  const el = (n, a, parent = svg) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); parent.appendChild(e); return e; };
  const L = 66, R = 610, T = 30, B = 200, xmax = 100, ymax = 160;
  const x = (v) => L + (v / xmax) * (R - L), y = (v) => B - (v / ymax) * (B - T);
  for (const g of [0, 50, 100, 150]) {
    el("line", { x1: L, x2: R, y1: y(g), y2: y(g), class: g ? "fc-grid" : "fc-axis" });
    el("text", { x: L - 8, y: y(g) + 4, class: "fc-tick", "text-anchor": "end" }).textContent = "+" + g + "%";
  }
  for (const g of [0, 10, 37, 50, 100]) {
    el("text", { x: x(g), y: B + 18, class: "fc-tick", "text-anchor": "middle" }).textContent = g;
  }
  el("text", { x: R, y: B + 38, class: "fc-axlabel", "text-anchor": "end" }).textContent = "concepts learned";
  el("text", { x: 14, y: (T + B) / 2, class: "fc-axlabel", "text-anchor": "middle", transform: `rotate(-90 14 ${(T + B) / 2})` }).textContent = "added generation time";
  // CIDM beyond 37: out of range, shaded
  el("rect", { x: x(37), y: T, width: R - x(37), height: B - T, class: "cc-na" });
  el("text", { x: (x(37) + R) / 2, y: T + 16, class: "cc-na-lab", "text-anchor": "middle" }).textContent = "CIDM does not run past 37 concepts";
  const S = [
    { name: "CIDM", col: "#56b4e9", pts: [[10, 45], [37, 154]], dash: "" },
    { name: "Ours", col: "#111111", pts: [[10, 14], [37, 14], [100, 14]], dash: "" },
  ];
  for (const s of S) {
    el("polyline", { points: s.pts.map(([a, b]) => `${x(a)},${y(b)}`).join(" "), fill: "none", stroke: s.col, "stroke-width": 2.5, "stroke-linejoin": "round" });
    const last = s.pts[s.pts.length - 1];
    el("text", { x: x(last[0]) + (s.name === "Ours" ? -4 : 8), y: y(last[1]) + (s.name === "Ours" ? -10 : 5), class: "fc-name", "text-anchor": s.name === "Ours" ? "end" : "start" }).textContent = s.name;
    for (const [a, b] of s.pts) {
      const g = el("g", { tabindex: 0, class: "cc-pt" });
      el("circle", { cx: x(a), cy: y(b), r: 14, class: "fc-hit" }, g);
      el("circle", { cx: x(a), cy: y(b), r: 5, fill: s.col, stroke: "#fff", "stroke-width": 2 }, g);
      const show = () => {
        tip.innerHTML = `<b>${s.name}</b>, ${a} concepts<br>+${b}% generation time`;
        tip.style.opacity = 1;
        const box = svg.getBoundingClientRect(), k = box.width / 640;
        tip.style.left = Math.min(box.width - 190, (x(a) + 12) * k) + "px";
        tip.style.top = (y(b) + 10) * k + "px";
      };
      const hide = () => { tip.style.opacity = 0; };
      g.addEventListener("mouseenter", show); g.addEventListener("mouseleave", hide);
      g.addEventListener("focus", show); g.addEventListener("blur", hide);
    }
  }
})();

// ---------- copy BibTeX ----------
(function () {
  const b = document.getElementById("copy-bib");
  if (!b) return;
  b.addEventListener("click", () => {
    const code = document.querySelector("div.sourceCode pre code, pre.bibtex code, pre code");
    if (!code) return;
    navigator.clipboard.writeText(code.innerText).then(() => {
      b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy BibTeX"), 1500);
    });
  });
})();
