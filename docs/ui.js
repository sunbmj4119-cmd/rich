/* 공용 도우미 — 데이터 읽기 · 숫자 형식 · 기능 켜기/끄기 · 차트(SVG).
   차트는 실제 화면 폭으로 다시 그린다(글자가 휴대폰에서 작아지지 않게). 색은 CSS 토큰만 쓴다. */
const U = (() => {
  const cache = {};
  const j = (url) => (cache[url] ||= fetch(url, { cache: "no-store" }).then((r) => {
    if (!r.ok) throw new Error(`${url} (${r.status})`);
    return r.json();
  }));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const ok = (v) => v !== null && v !== undefined && v !== "" && isFinite(v);
  const pct = (v, d = 1, sign = true) => ok(v) ? `${sign && v > 0 ? "+" : ""}${(+v).toFixed(d)}%` : "—";
  const num = (v, d = 0) => ok(v) ? (+v).toLocaleString("ko-KR", { maximumFractionDigits: d, minimumFractionDigits: d }) : "—";
  const won = (v) => ok(v) ? `${Math.round(v).toLocaleString("ko-KR")}원` : "—";
  const cls = (v) => ok(v) ? (v > 0 ? "pos" : v < 0 ? "neg" : "") : "";
  const $ = (s, r = document) => r.querySelector(s);

  /* ── 기능 켜기/끄기: 기본값은 config/features.yaml → docs/features.json, 기기별 변경은 localStorage ── */
  const KEY = "rich.features.v1";
  let defaults = {};
  const readLocal = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}"); } catch { return {}; } };
  const writeLocal = (o) => { try { localStorage.setItem(KEY, JSON.stringify(o)); } catch { /* 저장 불가 — 이번 방문에만 적용 */ } };
  let session = readLocal();
  const feat = {
    async load() {
      try { defaults = (await j("features.json")).display || {}; } catch { defaults = {}; }
      feat.apply();
      return defaults;
    },
    defaults: () => ({ ...defaults }),
    on: (k) => (k in session ? !!session[k] : k in defaults ? !!defaults[k] : true),
    set(k, v) { session = { ...session, [k]: !!v }; writeLocal(session); feat.apply(); },
    reset() { session = {}; writeLocal(session); feat.apply(); },
    all: () => { const o = { ...defaults }; for (const k in session) o[k] = session[k]; return o; },
    apply(root = document) {
      root.querySelectorAll("[data-feat]").forEach((el) => el.classList.toggle("feat-off", !feat.on(el.dataset.feat)));
      // 구역이 통째로 꺼진 탭은 버튼도 숨긴다
      document.querySelectorAll(".tabs button[data-t]").forEach((b) => {
        const pane = document.querySelector(`[data-tab="${b.dataset.t}"]`);
        b.hidden = !!pane && pane.classList.contains("feat-off");
      });
    },
  };

  /* ── 차트 공통 ─────────────────────────────────────────── */
  const mounts = new Map();
  const ro = typeof ResizeObserver !== "undefined" ? new ResizeObserver((ents) => {
    for (const e of ents) {
      const m = mounts.get(e.target);
      if (m && Math.abs(e.target.clientWidth - m.w) > 6) m.draw();
    }
  }) : null;
  function mount(el, render, after) {
    if (!el) return;
    const m = { w: 0, draw() { m.w = el.clientWidth || 340; el.innerHTML = render(m.w); after && after(el, m.w); } };
    mounts.set(el, m);
    m.draw();
    ro && ro.observe(el);
  }
  function extent(arrs) {
    let lo = Infinity, hi = -Infinity;
    for (const a of arrs) for (const v of a || []) if (ok(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (lo === Infinity) return [0, 1];
    if (lo === hi) { lo -= 1; hi += 1; }
    return [lo, hi];
  }
  function ticks(lo, hi, n = 4) {
    const step0 = (hi - lo) / n, mag = 10 ** Math.floor(Math.log10(step0)), e = step0 / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const T = (x, y, s, a = "start", extra = "") =>
    `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" text-anchor="${a}" style="fill:var(--chart-text);font-size:11px" ${extra}>${esc(s)}</text>`;

  /* 선 차트: labels[], series[{name,data,color,width,dash,area}], bands[{lo[],hi[],color,opacity}],
     vline{i,label}, yfmt, h, zero, tip */
  function line(el, o) {
    const H = o.h || 220, L = o.left ?? 48, R = 10, Tp = 10, B = 24;
    const n = o.labels.length;
    const render = (W) => {
      const all = o.series.map((s) => s.data).concat((o.bands || []).flatMap((b) => [b.lo, b.hi]));
      let [lo, hi] = extent(all);
      if (o.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
      const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
      const x = (i) => L + (W - L - R) * (n <= 1 ? 0 : i / (n - 1));
      const y = (v) => Tp + (H - Tp - B) * (1 - (v - lo) / (hi - lo));
      const yf = o.yfmt || ((v) => num(v, 1));
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "차트")}">`;
      for (const t of ticks(lo, hi, o.ticks || 4)) {
        s += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--chart-grid)" stroke-width="1"/>`;
        s += T(L - 6, y(t) + 4, yf(t), "end");
      }
      if (o.zero && lo < 0 && hi > 0) s += `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" style="stroke:var(--faint)" stroke-width="1"/>`;
      for (const b of o.bands || []) {
        const idx = b.lo.map((v, i) => (ok(v) && ok(b.hi[i]) ? i : -1)).filter((i) => i >= 0);
        if (!idx.length) continue;
        const pts = idx.map((i) => `${x(i)},${y(b.hi[i])}`).concat(idx.slice().reverse().map((i) => `${x(i)},${y(b.lo[i])}`));
        s += `<polygon points="${pts.join(" ")}" style="fill:${b.color};opacity:${b.opacity ?? 0.18}"/>`;
      }
      for (const se of o.series) {
        let d = "", pen = false, first = -1, last = -1;
        se.data.forEach((v, i) => {
          if (!ok(v)) { pen = false; return; }
          d += `${pen ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`; pen = true;
          if (first < 0) first = i; last = i;
        });
        if (se.area && first >= 0) {
          const base = y(Math.max(lo, Math.min(hi, se.areaBase ?? lo)));
          s += `<path d="${d}L${x(last)},${base}L${x(first)},${base}Z" style="fill:${se.color};opacity:.12"/>`;
        }
        s += `<path d="${d}" fill="none" style="stroke:${se.color}" stroke-width="${se.width || 2}" ${se.dash ? `stroke-dasharray="${se.dash}"` : ""} stroke-linejoin="round" stroke-linecap="round"/>`;
        if (se.end && last >= 0) s += `<circle cx="${x(last)}" cy="${y(se.data[last])}" r="3.5" style="fill:${se.color}"/>`;
      }
      if (o.vline && o.vline.i > 0 && o.vline.i < n) {
        const vx = x(o.vline.i);
        s += `<line x1="${vx}" x2="${vx}" y1="${Tp}" y2="${H - B}" style="stroke:var(--faint)" stroke-dasharray="3 3"/>`;
        s += T(vx + 4, Tp + 10, o.vline.label || "", "start");
      }
      const k = Math.max(1, Math.floor(n / Math.max(2, Math.floor((W - L) / 90))));
      for (let i = 0; i < n; i += k) if (i < n - k / 2) s += T(x(i), H - 6, o.xfmt ? o.xfmt(o.labels[i]) : o.labels[i], i === 0 ? "start" : "middle");
      s += T(x(n - 1), H - 6, o.xfmt ? o.xfmt(o.labels[n - 1]) : o.labels[n - 1], "end");
      s += `<line class="cur" x1="0" x2="0" y1="${Tp}" y2="${H - B}" style="stroke:var(--fg);opacity:0" stroke-width="1"/>`;
      s += `<rect class="hit" x="${L}" y="${Tp}" width="${W - L - R}" height="${H - Tp - B}" fill="transparent"/></svg>`;
      s += `<div class="tip" style="position:absolute;pointer-events:none;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 8px;font-size:12px;display:none;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,.12)"></div>`;
      el._geo = { L, R, W, n, x };
      return s;
    };
    const after = (box, W) => {
      box.style.position = "relative";
      const hit = box.querySelector(".hit"), cur = box.querySelector(".cur"), tip = box.querySelector(".tip");
      if (!hit) return;
      const show = (ev) => {
        const r = box.getBoundingClientRect();
        const px = ev.clientX - r.left;
        const i = Math.max(0, Math.min(n - 1, Math.round(((px - L) / (W - L - R)) * (n - 1))));
        const cx = L + (W - L - R) * (n <= 1 ? 0 : i / (n - 1));
        cur.setAttribute("x1", cx); cur.setAttribute("x2", cx); cur.style.opacity = 0.35;
        const yf = o.tipfmt || o.yfmt || ((v) => num(v, 1));
        tip.innerHTML = `<b>${esc(o.labels[i])}</b>` + o.series.filter((se) => ok(se.data[i]))
          .map((se) => `<div><span style="color:${se.color}">●</span> ${esc(se.name)} ${esc(yf(se.data[i]))}</div>`).join("")
          + (o.bands || []).filter((b) => b.name && ok(b.lo[i])).map((b) => `<div class="mut">${esc(b.name)} ${esc(yf(b.lo[i]))} ~ ${esc(yf(b.hi[i]))}</div>`).join("");
        tip.style.display = "block";
        const tw = tip.offsetWidth;
        tip.style.left = `${Math.min(Math.max(0, cx + 10), W - tw)}px`;
        tip.style.top = "4px";
      };
      hit.addEventListener("pointermove", show);
      hit.addEventListener("pointerdown", show);
      hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cur.style.opacity = 0; });
    };
    mount(el, render, after);
  }

  /* 세로 막대: labels[], values[], color(v,i), yfmt, h, note(i) */
  function bars(el, o) {
    const H = o.h || 200, L = 44, R = 8, Tp = 16, B = 30;
    mount(el, (W) => {
      let [lo, hi] = extent([o.values]);
      lo = Math.min(lo, 0); hi = Math.max(hi, 0);
      const pad = (hi - lo) * 0.12; hi += pad; if (lo < 0) lo -= pad;
      const n = o.values.length, bw = (W - L - R) / n;
      const y = (v) => Tp + (H - Tp - B) * (1 - (v - lo) / (hi - lo));
      const yf = o.yfmt || ((v) => num(v, 1));
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "막대 차트")}">`;
      for (const t of ticks(lo, hi, 4)) {
        s += `<line x1="${L}" x2="${W - R}" y1="${y(t)}" y2="${y(t)}" style="stroke:var(--chart-grid)"/>` + T(L - 6, y(t) + 4, yf(t), "end");
      }
      s += `<line x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}" style="stroke:var(--faint)"/>`;
      o.values.forEach((v, i) => {
        if (!ok(v)) return;
        const x0 = L + i * bw + bw * 0.16, w = bw * 0.68, y0 = y(Math.max(v, 0)), h = Math.abs(y(v) - y(0));
        const c = o.color ? o.color(v, i) : (v >= 0 ? "var(--green)" : "var(--red)");
        s += `<rect x="${x0}" y="${y0}" width="${w}" height="${Math.max(h, 1)}" rx="3" style="fill:${c}"><title>${esc(o.labels[i])} ${esc(yf(v))}</title></rect>`;
        if (bw > 30) s += T(x0 + w / 2, v >= 0 ? y0 - 4 : y0 + h + 12, yf(v), "middle");
        const lab = o.labels[i];
        const every = Math.ceil(46 / bw);                       // 라벨 하나에 46px 정도 필요
        if (i % every === 0) s += T(x0 + w / 2, H - 12, lab, "middle");
      });
      return s + "</svg>";
    });
  }

  /* 레이더: axes[{label, value 0~100}] */
  function radar(el, axes) {
    mount(el, (W) => {
      const n = axes.length, cx = W / 2;
      const r = Math.max(50, Math.min(126, W / 2 - 92));          // 좌우 라벨이 잘리지 않게
      const S = 2 * r + 64, cy = S / 2;
      const P = (i, v) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / n; return [cx + Math.cos(a) * r * v / 100, cy + Math.sin(a) * r * v / 100]; };
      let s = `<svg viewBox="0 0 ${W} ${S}" role="img" aria-label="팩터 레이더">`;
      for (const g of [25, 50, 75, 100]) s += `<polygon points="${axes.map((_, i) => P(i, g).join(",")).join(" ")}" fill="none" style="stroke:var(--chart-grid)"/>`;
      axes.forEach((a, i) => {
        const [x, y] = P(i, 100), [lx, ly] = P(i, 118);
        s += `<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" style="stroke:var(--chart-grid)"/>`;
        s += T(lx, ly + 4, a.label, Math.abs(lx - cx) < 8 ? "middle" : lx > cx ? "start" : "end");
      });
      s += `<polygon points="${axes.map((a, i) => P(i, 50).join(",")).join(" ")}" fill="none" style="stroke:var(--faint)" stroke-dasharray="3 3"/>`;
      s += `<polygon points="${axes.map((a, i) => P(i, ok(a.value) ? a.value : 50).join(",")).join(" ")}" style="fill:var(--blue);fill-opacity:.18;stroke:var(--blue)" stroke-width="2"/>`;
      axes.forEach((a, i) => { const [x, y] = P(i, ok(a.value) ? a.value : 50); s += `<circle cx="${x}" cy="${y}" r="3" style="fill:${a.nodata ? "var(--faint)" : "var(--blue)"}"/>`; });
      return s + "</svg>";
    });
  }

  /* 산점도: pts[[x,y]], xfmt, yfmt, xlab, ylab, color(p) */
  function scatter(el, o) {
    const H = o.h || 240, L = 46, R = 10, Tp = 10, B = 32;
    mount(el, (W) => {
      let [x0, x1] = extent([o.pts.map((p) => p[0])]), [y0, y1] = extent([o.pts.map((p) => p[1])]);
      const px = (x1 - x0) * 0.05, py = (y1 - y0) * 0.05; x0 -= px; x1 += px; y0 -= py; y1 += py;
      const X = (v) => L + (W - L - R) * (v - x0) / (x1 - x0), Y = (v) => Tp + (H - Tp - B) * (1 - (v - y0) / (y1 - y0));
      let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(o.aria || "산점도")}">`;
      for (const t of ticks(y0, y1, 4)) s += `<line x1="${L}" x2="${W - R}" y1="${Y(t)}" y2="${Y(t)}" style="stroke:var(--chart-grid)"/>` + T(L - 6, Y(t) + 4, o.yfmt(t), "end");
      for (const t of ticks(x0, x1, 4)) s += T(X(t), H - 16, o.xfmt(t), "middle");
      if (y0 < 0 && y1 > 0) s += `<line x1="${L}" x2="${W - R}" y1="${Y(0)}" y2="${Y(0)}" style="stroke:var(--faint)"/>`;
      for (const p of o.pts) s += `<circle cx="${X(p[0]).toFixed(1)}" cy="${Y(p[1]).toFixed(1)}" r="2.6" style="fill:${o.color ? o.color(p) : "var(--blue)"};fill-opacity:.55"/>`;
      s += T(W - R, H - 2, o.xlab || "", "end") + T(L, Tp - 1 + 10, o.ylab || "", "start");
      return s + "</svg>";
    });
  }

  /* 가로 기여도 막대 (HTML): items[{label, value, sub, faded}] — 0을 가운데 두고 좌우로 */
  function contrib(el, items, fmt = (v) => (v > 0 ? "+" : "") + v.toFixed(1)) {
    const m = Math.max(1, ...items.map((i) => Math.abs(i.value)));
    el.innerHTML = items.map((i) => {
      const w = (Math.abs(i.value) / m) * 50;
      const left = i.value >= 0 ? 50 : 50 - w;
      return `<div class="row" style="font-size:13px;opacity:${i.faded ? 0.5 : 1}">
        <div style="width:92px;flex-shrink:0">${esc(i.label)}${i.sub ? `<div class="small mut">${esc(i.sub)}</div>` : ""}</div>
        <div class="grow" style="position:relative;height:16px;background:var(--soft);border-radius:4px">
          <div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--faint)"></div>
          <div style="position:absolute;left:${left}%;width:${w}%;top:3px;bottom:3px;border-radius:3px;background:${i.value >= 0 ? "var(--green)" : "var(--red)"}"></div>
        </div>
        <div class="num" style="width:48px;text-align:right">${esc(fmt(i.value))}</div></div>`;
    }).join("");
  }

  /* 히트맵 셀 배경 — 값의 부호·크기로 초록/빨강 농도 */
  const heatBg = (v, cap = 10) => ok(v)
    ? `background:color-mix(in srgb, ${v >= 0 ? "var(--green)" : "var(--red)"} ${Math.min(70, (Math.abs(v) / cap) * 70).toFixed(0)}%, transparent)`
    : "";

  async function copy(text, btn) {
    try { await navigator.clipboard.writeText(text); if (btn) { const t = btn.textContent; btn.textContent = "복사됨"; setTimeout(() => (btn.textContent = t), 1400); } return true; }
    catch { return false; }
  }

  /* ── 페이지 이동: 휴대폰은 하단 탭바, 넓은 화면은 위쪽 칩 ── */
  const ICONS = {
    home: '<path d="M4 11l8-7 8 7v8a1 1 0 0 1-1 1h-4v-6h-6v6H5a1 1 0 0 1-1-1z"/>',
    bot: '<rect x="5" y="8" width="14" height="11" rx="3"/><path d="M12 4v4M9 13h.01M15 13h.01M9.5 16.5h5"/>',
    stock: '<path d="M4 19V5M4 19h16"/><path d="M7 15l4-4 3 3 5-6"/>',
    report: '<rect x="5" y="4" width="14" height="16" rx="2"/><path d="M9 9h6M9 13h6M9 17h3"/>',
    pick: '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
  };
  const PAGES = [["index.html", "대시보드", "home"], ["autotrade.html", "자동매매", "bot"], ["stock.html", "종목", "stock"],
                 ["report.html", "성과", "report"], ["features.html", "기능", "pick"]];
  function nav(active) {
    const top = document.getElementById("topnav");
    if (top) top.innerHTML = PAGES.map(([h, l]) => `<a href="${h}" class="${h === active ? "on" : ""}">${l}</a>`).join("");
    const b = document.createElement("nav");
    b.className = "bnav";
    b.setAttribute("aria-label", "페이지 이동");
    b.innerHTML = PAGES.map(([h, l, ic]) => `<a href="${h}" class="${h === active ? "on" : ""}" ${h === active ? 'aria-current="page"' : ""}>
      <svg viewBox="0 0 24 24" aria-hidden="true">${ICONS[ic]}</svg><span>${l}</span></a>`).join("");
    document.body.appendChild(b);
  }

  /* ── 페이지 안 탭: [data-tab=id] 구역 중 하나만 보인다. 마지막 탭을 기억하고 #id 로 바로 열 수 있다 ── */
  function tabs(bar, key, onShow) {
    const btns = [...bar.querySelectorAll("button[data-t]")];
    const panes = [...document.querySelectorAll("[data-tab]")];
    const ids = btns.map((b) => b.dataset.t);
    bar.setAttribute("role", "tablist");
    btns.forEach((b) => { b.setAttribute("role", "tab"); b.type = "button"; });
    const anchor = document.createElement("div");
    bar.closest(".stick")?.before(anchor);
    function show(t, user) {
      btns.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.t === t)));
      panes.forEach((p) => (p.hidden = p.dataset.tab !== t));
      try { localStorage.setItem("rich.tab." + key, t); } catch { /* 기억 못 해도 동작은 같다 */ }
      if (user) {
        const y = anchor.getBoundingClientRect().top + window.scrollY;
        if (window.scrollY > y) window.scrollTo(0, y);      // 탭을 바꾸면 새 내용의 처음부터
        bar.querySelector(`[data-t="${t}"]`)?.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
      onShow && onShow(t);
    }
    bar.addEventListener("click", (e) => { const b = e.target.closest("button[data-t]"); if (b) show(b.dataset.t, true); });
    bar.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
      const i = ids.indexOf(btns.find((b) => b.getAttribute("aria-selected") === "true")?.dataset.t);
      const n = ids[(i + (e.key === "ArrowRight" ? 1 : ids.length - 1)) % ids.length];
      show(n, true); bar.querySelector(`[data-t="${n}"]`).focus();
    });
    let saved = null;
    try { saved = localStorage.getItem("rich.tab." + key); } catch { /* 없음 */ }
    const h = (location.hash || "").replace(/^#(t-)?/, "");
    show(ids.includes(h) ? h : ids.includes(saved) ? saved : ids[0], false);
    return { show };
  }

  /* ── 바텀시트: 목록을 늘리지 않고 상세를 띄운다 ── */
  function sheet(title, html, onOpen) {
    const last = document.activeElement;
    const bg = document.createElement("div");
    bg.className = "sheet-bg";
    bg.innerHTML = `<div class="sheet" role="dialog" aria-modal="true" aria-label="${esc(title)}"><div class="grab" aria-hidden="true"></div>
      <header><h2>${esc(title)}</h2><button class="x" type="button" aria-label="닫기">✕</button></header><div class="sbody" style="display:flex;flex-direction:column;gap:10px">${html}</div></div>`;
    document.body.appendChild(bg);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const sh = bg.querySelector(".sheet");
    const close = () => { bg.remove(); document.body.style.overflow = prev; document.removeEventListener("keydown", key); last && last.focus && last.focus(); };
    const key = (e) => { if (e.key === "Escape") close(); };
    document.addEventListener("keydown", key);
    bg.addEventListener("click", (e) => { if (e.target === bg) close(); });
    bg.querySelector(".x").addEventListener("click", close);
    // 손잡이·머리를 아래로 끌면 닫힌다
    let y0 = null;
    const head = [bg.querySelector(".grab"), bg.querySelector("header")];
    head.forEach((h) => h.addEventListener("touchstart", (e) => { y0 = e.touches[0].clientY; }, { passive: true }));
    sh.addEventListener("touchmove", (e) => { if (y0 === null) return; const d = e.touches[0].clientY - y0; if (d > 0) sh.style.transform = `translateY(${d}px)`; }, { passive: true });
    sh.addEventListener("touchend", (e) => {
      if (y0 === null) return;
      const d = e.changedTouches[0].clientY - y0; y0 = null;
      if (d > 90) close(); else sh.style.transform = "";
    });
    bg.querySelector(".x").focus();
    onOpen && onOpen(bg.querySelector(".sbody"));
    feat.apply(bg);
    return close;
  }

  /* ── 앞의 n개만 보이고 '더 보기'로 늘린다 ── */
  function more(box, items, render, n = 5, step = 10) {
    let shown = n;
    const draw = () => {
      box.innerHTML = items.slice(0, shown).map(render).join("") +
        (items.length > shown ? `<button class="more" type="button">${Math.min(step, items.length - shown)}개 더 보기 · 남은 ${items.length - shown}개</button>` : "");
      const m = box.querySelector(".more");
      if (m) m.addEventListener("click", () => { shown += step; draw(); });
    };
    draw();
  }

  return { j, esc, ok, pct, num, won, cls, $, feat, line, bars, radar, scatter, contrib, heatBg, copy, mount, nav, tabs, sheet, more };
})();
