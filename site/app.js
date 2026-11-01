"use strict";

const state = { words: [], index: new Map(), dim: 0, q: null, scale: null, norm: null };

// ---------- 読み込み ----------
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (!res.body || !total) return new Uint8Array(await res.arrayBuffer());
  const reader = res.body.getReader();
  const buf = new Uint8Array(total);
  let off = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (off + value.length > buf.length) return null; // 圧縮転送などでサイズが合わない場合
    buf.set(value, off);
    off += value.length;
    onProgress(off / total);
  }
  return off === total ? buf : buf.slice(0, off);
}

async function load() {
  const setPct = (p) => {
    document.getElementById("loading-pct").textContent = Math.round(p * 100) + "%";
    document.getElementById("loading-bar").style.width = p * 100 + "%";
  };
  const [meta, words] = await Promise.all([
    fetch("data/meta.json").then((r) => r.json()),
    fetch("data/words.json").then((r) => r.json()),
  ]);
  let bin = await fetchWithProgress("data/vectors.bin", setPct);
  if (!bin) bin = new Uint8Array(await (await fetch("data/vectors.bin")).arrayBuffer());

  const dim = meta.dim, n = words.length, stride = 4 + dim;
  const dv = new DataView(bin.buffer, bin.byteOffset, bin.byteLength);
  const q = new Int8Array(n * dim);
  const scale = new Float32Array(n);
  const norm = new Float32Array(n);
  const raw = new Int8Array(bin.buffer, bin.byteOffset, bin.byteLength);
  for (let i = 0; i < n; i++) {
    const base = i * stride;
    scale[i] = dv.getFloat32(base, true);
    q.set(raw.subarray(base + 4, base + stride), i * dim);
    let s = 0;
    for (let k = 0; k < dim; k++) { const x = q[i * dim + k]; s += x * x; }
    norm[i] = Math.sqrt(s) || 1;
  }
  Object.assign(state, { words, dim, q, scale, norm });
  words.forEach((w, i) => state.index.set(w, i));
  document.getElementById("vocab-count").textContent = n.toLocaleString();
}

// ---------- ベクトル計算 ----------
function lookup(raw) {
  const w = raw.trim();
  if (!w) return -1;
  const cands = [w, w.normalize("NFKC"), w.toLowerCase(), w.normalize("NFKC").toLowerCase()];
  for (const c of cands) if (state.index.has(c)) return state.index.get(c);
  return -1;
}

// 単位ベクトル（Float32Array）を返す
function vec(i) {
  const { dim, q, norm } = state;
  const v = new Float32Array(dim);
  const inv = 1 / norm[i];
  for (let k = 0; k < dim; k++) v[k] = q[i * dim + k] * inv;
  return v;
}

function cosIdx(a, b) {
  const { dim, q, norm } = state;
  let s = 0;
  const oa = a * dim, ob = b * dim;
  for (let k = 0; k < dim; k++) s += q[oa + k] * q[ob + k];
  return s / (norm[a] * norm[b]);
}

// v（任意ベクトル）に近い単語を上位 k 個
function nearest(v, k, exclude = new Set()) {
  const { dim, q, norm, words } = state;
  let vn = 0;
  for (let j = 0; j < dim; j++) vn += v[j] * v[j];
  vn = Math.sqrt(vn) || 1;
  const top = []; // [score, idx] 昇順
  for (let i = 0; i < words.length; i++) {
    if (exclude.has(i)) continue;
    let s = 0;
    const o = i * dim;
    for (let j = 0; j < dim; j++) s += q[o + j] * v[j];
    s /= norm[i] * vn;
    if (top.length < k) {
      top.push([s, i]);
      top.sort((x, y) => x[0] - y[0]);
    } else if (s > top[0][0]) {
      top[0] = [s, i];
      top.sort((x, y) => x[0] - y[0]);
    }
  }
  return top.reverse().map(([s, i]) => ({ word: words[i], idx: i, score: s }));
}

function suggestions(raw, k = 8) {
  const w = raw.trim().normalize("NFKC");
  if (!w) return [];
  const out = [];
  for (const cand of state.words) {
    if (cand.includes(w) || (w.length >= 2 && w.includes(cand) && cand.length >= 2)) {
      out.push(cand);
      if (out.length >= k) break;
    }
  }
  return out;
}

// ---------- 表示ヘルパー ----------
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 3) => x.toFixed(d);

function verdict(c) {
  if (c >= 0.7) return "とても近い";
  if (c >= 0.5) return "近い";
  if (c >= 0.35) return "やや近い";
  if (c >= 0.2) return "遠い";
  return "とても遠い";
}

function notFound(word, onPick) {
  const sug = suggestions(word);
  const id = "s" + Math.random().toString(36).slice(2);
  setTimeout(() => {
    document.querySelectorAll(`#${id} button`).forEach((b) => b.addEventListener("click", () => onPick(b.textContent)));
  });
  return `<p class="error">「${esc(word)}」は辞書（頻出${state.words.length.toLocaleString()}語）にありません。</p>` +
    (sug.length ? `<p class="suggest" id="${id}">候補：${sug.map((s) => `<button type="button">${esc(s)}</button>`).join("")}</p>` : "");
}

function barList(items, { max = 1, clickable = true } = {}) {
  return `<ul class="bars">${items.map((it) => {
    if (it.missing) return `<li class="missing"><span class="w">${esc(it.word)}</span><span></span><span class="v">未収録</span></li>`;
    const pct = Math.max(0, Math.min(1, it.score / max)) * 100;
    const w = clickable ? `<button type="button" data-near="${esc(it.word)}">${esc(it.word)}</button>` : esc(it.word);
    return `<li><span class="w" title="${esc(it.word)}">${w}</span><span class="track"><span class="fill" style="width:${pct}%;display:block"></span></span><span class="v">${fmt(it.score)}</span></li>`;
  }).join("")}</ul>`;
}

// ---------- タブ ----------
function showTab(name) {
  document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== "tab-" + name));
}

function setHash(params) {
  history.replaceState(null, "", "#" + new URLSearchParams(params).toString());
}

// ---------- 2語の距離 ----------
function runPair() {
  const a = document.getElementById("pair-a").value.trim();
  const b = document.getElementById("pair-b").value.trim();
  const out = document.getElementById("pair-result");
  if (!a || !b) return;
  setHash({ tab: "pair", a, b });
  const ia = lookup(a), ib = lookup(b);
  const retry = (el) => (w) => { document.getElementById(el).value = w; runPair(); };
  if (ia < 0 || ib < 0) {
    out.innerHTML = (ia < 0 ? notFound(a, retry("pair-a")) : "") + (ib < 0 ? notFound(b, retry("pair-b")) : "");
    return;
  }
  const c = cosIdx(ia, ib);
  const dist = 1 - c;
  const angle = (Math.acos(Math.max(-1, Math.min(1, c))) * 180) / Math.PI;
  const euclid = Math.sqrt(Math.max(0, 2 - 2 * c));
  const pos = Math.max(0, Math.min(1, dist / 1.1)) * 100;
  const na = nearest(vec(ia), 8, new Set([ia]));
  const nb = nearest(vec(ib), 8, new Set([ib]));
  // 何位に入るか（a から見た b の順位）
  let rank = 1;
  for (let i = 0; i < state.words.length; i++) if (i !== ia && i !== ib && cosIdx(ia, i) > c) rank++;

  out.innerHTML = `
    <div class="score">
      <div>「${esc(state.words[ia])}」と「${esc(state.words[ib])}」の距離</div>
      <div class="big">${fmt(dist)}</div>
      <div class="label">${verdict(c)}</div>
    </div>
    <div class="meter"><div class="pin" style="left:${pos}%"></div></div>
    <div class="meter-scale"><span>近い（0）</span><span>遠い（1.1〜）</span></div>
    <div class="stats">
      <div><b>${fmt(c)}</b><span>コサイン類似度</span></div>
      <div><b>${angle.toFixed(1)}°</b><span>角度</span></div>
      <div><b>${rank.toLocaleString()}位</b><span>「${esc(state.words[ia])}」から見た近さ順位</span></div>
    </div>
    <div class="cols">
      <div><h3>「${esc(state.words[ia])}」に近い言葉</h3>${barList(na)}</div>
      <div><h3>「${esc(state.words[ib])}」に近い言葉</h3>${barList(nb)}</div>
    </div>
    <p class="hint">ユークリッド距離（単位ベクトル間）：${fmt(euclid)}</p>`;
}

// ---------- まとめて比較 ----------
function pca2(vectors) {
  const n = vectors.length, d = vectors[0].length;
  const mean = new Float32Array(d);
  for (const v of vectors) for (let k = 0; k < d; k++) mean[k] += v[k] / n;
  const X = vectors.map((v) => v.map((x, k) => x - mean[k]));
  const comps = [];
  for (let c = 0; c < 2; c++) {
    let p = new Float32Array(d).map(() => Math.random() - 0.5);
    for (let it = 0; it < 100; it++) {
      const next = new Float32Array(d);
      for (const x of X) {
        let s = 0;
        for (let k = 0; k < d; k++) s += x[k] * p[k];
        for (let k = 0; k < d; k++) next[k] += s * x[k];
      }
      for (const prev of comps) { // 直交化
        let s = 0;
        for (let k = 0; k < d; k++) s += next[k] * prev[k];
        for (let k = 0; k < d; k++) next[k] -= s * prev[k];
      }
      let nn = 0;
      for (let k = 0; k < d; k++) nn += next[k] * next[k];
      nn = Math.sqrt(nn) || 1;
      p = next.map((x) => x / nn);
    }
    comps.push(p);
  }
  return X.map((x) => comps.map((p) => x.reduce((s, xv, k) => s + xv * p[k], 0)));
}

function scatter(labels, pts, baseIdx) {
  const W = 640, H = 360, pad = 40, padR = 110;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const sx = (x) => pad + ((x - x0) / (x1 - x0 || 1)) * (W - pad - padR);
  const sy = (y) => H - pad - ((y - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
  const bx = sx(pts[baseIdx][0]), by = sy(pts[baseIdx][1]);
  return `<svg class="map" viewBox="0 0 ${W} ${H}" role="img" aria-label="単語の2次元マップ">
    ${pts.map((p, i) => i === baseIdx ? "" : `<line x1="${bx}" y1="${by}" x2="${sx(p[0])}" y2="${sy(p[1])}"/>`).join("")}
    ${pts.map((p, i) => `<circle cx="${sx(p[0])}" cy="${sy(p[1])}" r="${i === baseIdx ? 6 : 4}"/>
      <text x="${sx(p[0]) + 8}" y="${sy(p[1]) + 4}" class="${i === baseIdx ? "base" : ""}">${esc(labels[i])}</text>`).join("")}
  </svg>`;
}

function runMulti() {
  const base = document.getElementById("multi-base").value.trim();
  const list = [...new Set(document.getElementById("multi-list").value.split(/[\s,、，]+/).map((s) => s.trim()).filter(Boolean))];
  const out = document.getElementById("multi-result");
  if (!base || !list.length) return;
  setHash({ tab: "multi", base, list: list.join(" ") });
  const ib = lookup(base);
  if (ib < 0) {
    out.innerHTML = notFound(base, (w) => { document.getElementById("multi-base").value = w; runMulti(); });
    return;
  }
  const items = list.map((w) => {
    const i = lookup(w);
    return i < 0 ? { word: w, missing: true } : { word: state.words[i], idx: i, score: cosIdx(ib, i) };
  });
  const found = items.filter((x) => !x.missing).sort((a, b) => b.score - a.score);
  const missing = items.filter((x) => x.missing);
  let html = `<h3>「${esc(state.words[ib])}」に近い順（コサイン類似度）</h3>${barList([...found, ...missing])}`;
  if (found.length >= 2) {
    const idxs = [ib, ...found.map((f) => f.idx)];
    html += `<h3 style="margin-top:20px">意味の地図（主成分分析で2次元に投影）</h3>` +
      scatter(idxs.map((i) => state.words[i]), pca2(idxs.map(vec)), 0) +
      `<p class="hint">300次元を2次元に押し縮めた図なので、位置関係はおおまかな目安です。</p>`;
  }
  out.innerHTML = html;
}

// ---------- 近い言葉 ----------
function runNear() {
  const w = document.getElementById("near-word").value.trim();
  const out = document.getElementById("near-result");
  if (!w) return;
  setHash({ tab: "near", w });
  const i = lookup(w);
  if (i < 0) {
    out.innerHTML = notFound(w, (x) => { document.getElementById("near-word").value = x; runNear(); });
    return;
  }
  out.innerHTML = `<h3>「${esc(state.words[i])}」に近い言葉 TOP30</h3>` + barList(nearest(vec(i), 30, new Set([i])));
}

// ---------- 言葉の足し算 ----------
function runCalc() {
  const ids = ["calc-a", "calc-b", "calc-c"];
  const ws = ids.map((id) => document.getElementById(id).value.trim());
  const out = document.getElementById("calc-result");
  if (ws.some((w) => !w)) return;
  setHash({ tab: "calc", a: ws[0], b: ws[1], c: ws[2] });
  const is = ws.map(lookup);
  if (is.some((i) => i < 0)) {
    out.innerHTML = is.map((i, k) => i < 0 ? notFound(ws[k], (x) => { document.getElementById(ids[k]).value = x; runCalc(); }) : "").join("");
    return;
  }
  const [va, vb, vc] = is.map(vec);
  const v = va.map((x, k) => x - vb[k] + vc[k]);
  const res = nearest(v, 15, new Set(is));
  out.innerHTML = `<h3>${esc(ws[0])} − ${esc(ws[1])} ＋ ${esc(ws[2])} ≒</h3>` + barList(res);
}

// ---------- 初期化 ----------
function bind() {
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  const forms = { "pair-form": runPair, "multi-form": runMulti, "near-form": runNear, "calc-form": runCalc };
  for (const [id, fn] of Object.entries(forms)) {
    document.getElementById(id).addEventListener("submit", (e) => { e.preventDefault(); fn(); });
  }
  document.querySelectorAll('.examples[data-target="pair"] button').forEach((b) => b.addEventListener("click", () => {
    document.getElementById("pair-a").value = b.dataset.a;
    document.getElementById("pair-b").value = b.dataset.b;
    runPair();
  }));
  document.querySelectorAll('.examples[data-target="calc"] button').forEach((b) => b.addEventListener("click", () => {
    document.getElementById("calc-a").value = b.dataset.a;
    document.getElementById("calc-b").value = b.dataset.b;
    document.getElementById("calc-c").value = b.dataset.c;
    runCalc();
  }));
  // 結果内の単語クリック → 近い言葉タブへ
  document.getElementById("app").addEventListener("click", (e) => {
    const t = e.target.closest("[data-near]");
    if (!t) return;
    document.getElementById("near-word").value = t.dataset.near;
    showTab("near");
    runNear();
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

function restoreFromHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const tab = p.get("tab");
  const set = (id, v) => { if (v) document.getElementById(id).value = v; };
  if (tab === "multi") { set("multi-base", p.get("base")); set("multi-list", p.get("list")); showTab("multi"); runMulti(); }
  else if (tab === "near") { set("near-word", p.get("w")); showTab("near"); runNear(); }
  else if (tab === "calc") { set("calc-a", p.get("a")); set("calc-b", p.get("b")); set("calc-c", p.get("c")); showTab("calc"); runCalc(); }
  else if (tab === "pair") { set("pair-a", p.get("a")); set("pair-b", p.get("b")); runPair(); }
}

(async () => {
  try {
    await load();
    document.getElementById("loading").hidden = true;
    document.getElementById("app").hidden = false;
    bind();
    restoreFromHash();
  } catch (err) {
    document.getElementById("loading").innerHTML = `<p class="error">モデルの読み込みに失敗しました：${esc(err.message)}</p>`;
    console.error(err);
  }
})();
