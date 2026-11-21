"""Generate the profile README artwork (assets/*.svg) and README.md.

    python tools/fetch_metrics.py   # refresh tools/metrics.json from GitHub (optional)
    python tools/build.py

The header rasterises "YOGESHWAR" onto a character grid: solid blocks inside the
letters, shading toward the edges, and binary digits that break off the edges,
drift away and stream back in. Below it, captions decode out of binary and
dissolve again in a loop. Everything is plain SVG + CSS so it animates inside
GitHub's <img>, and all motion stops under prefers-reduced-motion.
"""

import base64
import datetime as dt
import io
import json
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
METRICS = ROOT / "tools" / "metrics.json"
FONT_PATH = "C:/Windows/Fonts/seguibl.ttf"  # Segoe UI Black, only used to rasterise the name

MONO = "'JetBrains Mono', 'Cascadia Code', SFMono-Regular, Consolas, Menlo, 'Liberation Mono', monospace"

PAPER = "#0a0f1c"
LINE = "#26324a"
INK = "#e8eef8"
INK2 = "#a9b8d0"
MUTED = "#7f8ca6"
WHITE = "#ffffff"
BLUE = "#3d9bff"
SKY = "#7cc8ff"
RED = "#ff3b5c"
GREEN = "#10b981"

# accent -> (main, light, soft, dark): light tints titles and tag text, dark fills a lit tag
ACCENTS = {
    "blue": (BLUE, "#9fd2ff", SKY, "#1f5fd6"),
    "red": (RED, "#ff9aac", "#ff6b85", "#be123c"),
    "green": (GREEN, "#86efac", "#34d399", "#047857"),
}

CAPTIONS = [
    "Systems Architecture and Engineering",
    "Foundational AI Primitives",
    "Edge Computing Frameworks",
    "Decentralized Learning",
    "Neuromorphic Research",
    "Author of CortexNET, HaDA Systems and Many more",
    "AI Security and Anomaly Detection",
    "Information Retrieval at the Edge",
]

SUMMARY = ("A 3rd Year Cyber Security Undergraduate, working at the intersection of AI, Security, "
           "Systems, Information Retrieval and Distributed Computing to build efficient and powerful "
           "Solutions that make real world impacts.")

REDUCED_MOTION = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"

def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def rrect(x: float, y: float, w: float, h: float, r: float) -> str:
    return (f"M{x + r},{y} H{x + w - r} A{r},{r} 0 0 1 {x + w},{y + r} V{y + h - r} "
            f"A{r},{r} 0 0 1 {x + w - r},{y + h} H{x + r} A{r},{r} 0 0 1 {x},{y + h - r} "
            f"V{y + r} A{r},{r} 0 0 1 {x + r},{y} Z")


def wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        lines.append(line)
    return lines


def bit_field(rng: random.Random, x0: float, x1: float, y0: float, y1: float, step: float, alpha: float = 0.7) -> str:
    """A block of binary that is dense on the left and breaks apart to the right."""
    out = []
    y = y0
    while y <= y1:
        x = x0 + rng.uniform(0, step)
        while x < x1:
            t = (x - x0) / (x1 - x0)
            if rng.random() < 0.92 - t * 0.85:
                lift = t * t * rng.uniform(0, 18)
                out.append(f'<text x="{x:.0f}" y="{y - lift:.0f}" fill-opacity="{max(0.08, alpha - t * alpha * 0.85):.2f}">{rng.choice("01")}</text>')
            x += step * (1 + t * 1.6)
        y += step * 1.7
    return "".join(out)


def card_defs(W: float, H: float, m: float, r: float, shadow: str) -> str:
    return f"""<clipPath id="c"><rect x="{m}" y="{m}" width="{W - 2 * m}" height="{H - 2 * m}" rx="{r}"/></clipPath>"""


# ---------------------------------------------------------------------------
# Header: character-art name dissolving into binary, plus rotating captions
# ---------------------------------------------------------------------------

def name_mask(text: str, cols: int, cell_w: float, cell_h: float, stretch: float) -> np.ndarray:
    """Coverage (0..1) of `text` sampled on a cols x rows character grid."""
    font = ImageFont.truetype(FONT_PATH, 400)
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 40, b - t + 40), 0)
    ImageDraw.Draw(img).text((20 - l, 20 - t), text, font=font, fill=255)
    w, h = img.size
    rows = round(cols * (h / w) * stretch * (cell_w / cell_h))
    small = img.resize((cols, rows), Image.BOX)
    return np.asarray(small, dtype=np.float32) / 255.0


def depth_map(filled: np.ndarray) -> np.ndarray:
    """Cell distance to the nearest empty cell (4-neighbour BFS); 0 for empty cells."""
    rows, cols = filled.shape
    depth = np.where(filled, 10_000, 0)
    frontier = [(y, x) for y in range(rows) for x in range(cols) if not filled[y, x]]
    while frontier:
        nxt = []
        for y, x in frontier:
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols and depth[ny, nx] > depth[y, x] + 1:
                    depth[ny, nx] = depth[y, x] + 1
                    nxt.append((ny, nx))
        frontier = nxt
    return depth


def outward(filled: np.ndarray, y: int, x: int) -> tuple[float, float]:
    """Unit vector pointing from a filled cell toward its empty neighbours."""
    rows, cols = filled.shape
    vx = vy = 0.0
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny, nx = y + dy, x + dx
            empty = not (0 <= ny < rows and 0 <= nx < cols) or not filled[ny, nx]
            if empty and (dy or dx):
                vx += dx
                vy += dy
    n = math.hypot(vx, vy) or 1.0
    return vx / n, vy / n


def pct(seconds: float, total: float) -> str:
    return f"{100 * seconds / total:.3f}%"


def caption_layer(rng: random.Random, cx: float, base_y: float) -> tuple[str, str]:
    """Captions that decode out of flickering binary, hold, then blow away as digits."""
    slot, reveal, hide, gone = 5.5, 0.55, 4.1, 5.2   # seconds within each caption's slot
    total = slot * len(CAPTIONS)
    cw, size = 13.2, 22

    css = f"""
  .cap {{ font-size: {size}px; font-weight: 700; }}
  .ci, .ch, .co {{ animation-duration: {total}s; animation-iteration-count: infinite; animation-fill-mode: backwards; animation-timing-function: linear; }}
  .ci {{ animation-name: cin; opacity: 0; }}
  .ch {{ animation-name: cshow; opacity: 0; }}
  .ch.k0 {{ opacity: 1; }}
  .co {{ animation-name: cout; opacity: 0; transform-box: fill-box; transform-origin: center; animation-timing-function: cubic-bezier(.2,.6,.3,1); }}
  @keyframes cin {{ 0% {{ opacity: 0; }} 0.05% {{ opacity: 1; }} {pct(reveal, total)} {{ opacity: 1; }} {pct(reveal + 0.02, total)}, 100% {{ opacity: 0; }} }}
  @keyframes cshow {{ 0%, {pct(reveal, total)} {{ opacity: 0; }} {pct(reveal + 0.02, total)}, {pct(hide, total)} {{ opacity: 1; }} {pct(hide + 0.06, total)}, 100% {{ opacity: 0; }} }}
  @keyframes cout {{
    0%, {pct(hide, total)} {{ opacity: 0; transform: translate(0, 0) scale(1); }}
    {pct(hide + 0.06, total)} {{ opacity: 1; transform: translate(0, 0) scale(1); }}
    {pct(gone, total)} {{ opacity: 0; transform: translate(var(--dx), var(--dy)) scale(.4); }}
    100% {{ opacity: 0; transform: translate(var(--dx), var(--dy)) scale(.4); }}
  }}"""

    out = [f'<g class="cap" aria-hidden="true">']
    for k, caption in enumerate(CAPTIONS):
        x0 = cx - len(caption) * cw / 2
        for i, ch in enumerate(caption):
            if ch == " ":
                continue
            x = round(x0 + i * cw + cw / 2, 1)
            phase = round(k * slot + i * 0.035 + rng.uniform(0, 0.2), 3)
            flip = round(rng.uniform(0.14, 0.3), 2)
            out.append(
                f'<text class="ci" x="{x}" y="{base_y}" fill="{BLUE}" text-anchor="middle" '
                f'style="animation-delay:{phase}s">{rng.choice("01")}</text>'
            )
            k0 = " k0" if k == 0 else ""
            out.append(f'<text class="ch{k0}" x="{x}" y="{base_y}" text-anchor="middle" fill="url(#capInk)" style="animation-delay:{phase}s">{esc(ch)}</text>')
            dx, dy = rng.uniform(18, 70), -rng.uniform(8, 38)
            fill = RED if rng.random() < 0.22 else SKY
            out.append(
                f'<text class="co" x="{x}" y="{base_y}" text-anchor="middle" fill="{fill}" '
                f'style="--dx:{dx:.0f}px;--dy:{dy:.0f}px;animation-delay:{phase}s">{rng.choice("01")}</text>'
            )
    out.append("</g>")
    return css, "\n".join(out)


def header_svg() -> str:
    rng = random.Random(7)
    W, H = 1200, 420
    cols, cell_w, cell_h = 168, 6.0, 9.0
    cov = name_mask("YOGESHWAR", cols, cell_w, cell_h, stretch=1.3)
    pad = 4
    cov = np.pad(cov, pad)
    rows, cols = cov.shape
    filled = cov >= 0.5
    depth = depth_map(filled)

    x0 = (W - cols * cell_w) / 2
    y0 = 34.0

    def cx(x: float) -> float:
        return round(x0 + x * cell_w, 1)

    def cy(y: float) -> float:
        return round(y0 + (y + 1) * cell_h, 1)

    body: dict[int, list[tuple[int, str]]] = {}   # row -> [(col, glyph)] for the solid letters
    boil: list[tuple[int, int]] = []             # edge cells that flip between 0 and 1
    dust: list[tuple[float, float, str, float]] = []
    leaving: list[str] = []                      # particles blowing off the edges
    arriving: list[str] = []                     # particles streaming back in

    wind = (0.9, -0.45)  # dust blows up and to the right
    for y in range(rows):
        for x in range(cols):
            d = int(depth[y, x])
            if d == 0:
                if cov[y, x] > 0.12 and rng.random() < 0.5:
                    dust.append((cx(x), cy(y), rng.choice("01░"), round(0.3 + cov[y, x] * 0.5, 2)))
                continue

            ox, oy = outward(filled, y, x)
            downwind = max(0.0, ox * wind[0] + oy * wind[1])  # edges facing the wind erode more
            if d >= 3:
                glyph = "█"
            elif d == 2:
                glyph = "▓" if rng.random() > 0.12 else "▒"
            else:
                r = rng.random()
                if r < 0.18 + 0.32 * downwind:
                    glyph = ""  # already blown away
                elif r < 0.62:
                    glyph = "01"
                else:
                    glyph = rng.choice("▒░")

            if glyph == "01":
                boil.append((x, y))
            elif glyph:
                body.setdefault(y, []).append((x, glyph))

            if d != 1:
                continue
            for _ in range(1 + int(downwind * 2)):
                if rng.random() < 0.45:
                    dist = rng.uniform(1.0, 2.5 + 5.0 * downwind)
                    ddx = ox * dist + wind[0] * dist * 0.8
                    ddy = oy * dist * 0.6 + wind[1] * dist * 0.5
                    alpha = max(0.1, 0.75 - dist * 0.1)
                    dust.append((cx(x + ddx), cy(y + ddy), rng.choice("01"), round(alpha, 2)))

            vx, vy = ox * 0.7 + wind[0] + rng.uniform(-0.35, 0.35), oy * 0.7 + wind[1] + rng.uniform(-0.3, 0.2)
            n = math.hypot(vx, vy) or 1.0
            vx, vy = vx / n, vy / n
            px, py = -vy, vx  # perpendicular, for a curved path
            if rng.random() < 0.16 + 0.34 * downwind:
                dist = rng.uniform(45, 160)
                swirl = rng.uniform(-1, 1) * dist * 0.28
                mx, my = vx * dist * 0.45 + px * swirl, vy * dist * 0.45 + py * swirl
                ex, ey = vx * dist + px * swirl * 0.4, vy * dist + py * swirl * 0.4 - dist * 0.1
                hot = " hot" if rng.random() < 0.22 else ""
                leaving.append(
                    f'<text class="p{hot}" x="{cx(x)}" y="{cy(y)}" font-size="{rng.uniform(7.5, 11.5):.1f}" '
                    f'style="--mx:{mx:.0f}px;--my:{my:.0f}px;--dx:{ex:.0f}px;--dy:{ey:.0f}px;'
                    f'animation-duration:{rng.uniform(4.5, 9.5):.2f}s;animation-delay:-{rng.uniform(0, 9.5):.2f}s">{rng.choice("01")}</text>'
                )
            elif rng.random() < 0.07:
                dist = rng.uniform(60, 140)
                swirl = rng.uniform(-1, 1) * dist * 0.3
                sx, sy = vx * dist + px * swirl, vy * dist + py * swirl
                mx, my = sx * 0.4 - px * swirl * 0.3, sy * 0.4 - py * swirl * 0.3
                arriving.append(
                    f'<text class="q" x="{cx(x)}" y="{cy(y)}" font-size="{rng.uniform(7.5, 10.5):.1f}" '
                    f'style="--mx:{mx:.0f}px;--my:{my:.0f}px;--dx:{sx:.0f}px;--dy:{sy:.0f}px;'
                    f'animation-duration:{rng.uniform(5, 9):.2f}s;animation-delay:-{rng.uniform(0, 9):.2f}s">{rng.choice("01")}</text>'
                )

    if len(leaving) > 80:
        step = len(leaving) / 80
        leaving = [leaving[int(i * step)] for i in range(80)]
    if len(arriving) > 16:
        step = len(arriving) / 16
        arriving = [arriving[int(i * step)] for i in range(16)]

    name_top, name_bottom = y0, y0 + rows * cell_h
    name_left, name_right = x0, x0 + cols * cell_w
    cap_y = round(name_bottom + 64)
    cap_css, cap_svg = caption_layer(random.Random(23), W / 2, cap_y)

    def rows_to_text(attrs: str, dx: float = 0.0) -> list[str]:
        lines = []
        for y, items in sorted(body.items()):
            xs = " ".join(f"{cx(x) + dx:.1f}" for x, _ in items)
            chars = "".join(g for _, g in items)
            lines.append(f'<text class="g" x="{xs}" y="{cy(y)}" {attrs}>{esc(chars)}</text>')
        return lines

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">',
           '<title id="t">YOGESHWAR</title>',
           '<desc id="d">The name YOGESHWAR drawn in character art, its edges disintegrating into drifting binary digits. '
           f'Below it, captions decode from binary in turn: {esc(", ".join(CAPTIONS))}.</desc>']
    out.append(f"""<defs>
  <linearGradient id="ink" gradientUnits="userSpaceOnUse" x1="{name_left:.0f}" y1="{name_top:.0f}" x2="{name_left + 260:.0f}" y2="{name_bottom + 160:.0f}">
    <stop offset="0" stop-color="{WHITE}"/>
    <stop offset="0.45" stop-color="#e8f4ff"/>
    <stop offset="0.75" stop-color="{SKY}"/>
    <stop offset="1" stop-color="{BLUE}"/>
  </linearGradient>
  <linearGradient id="capInk" gradientUnits="userSpaceOnUse" x1="300" y1="0" x2="900" y2="0">
    <stop offset="0" stop-color="{WHITE}"/><stop offset="0.6" stop-color="#9fd2ff"/><stop offset="1" stop-color="#ff6b85"/>
  </linearGradient>
  <linearGradient id="ghost" gradientUnits="userSpaceOnUse" x1="{name_left:.0f}" y1="0" x2="{name_right:.0f}" y2="0">
    <stop offset="0" stop-color="{RED}" stop-opacity="0.75"/>
    <stop offset="0.35" stop-color="{RED}" stop-opacity="0.12"/>
    <stop offset="0.7" stop-color="{RED}" stop-opacity="0"/>
    <stop offset="1" stop-color="{RED}" stop-opacity="0.5"/>
  </linearGradient>
  <linearGradient id="sheenGrad" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="#ffffff" stop-opacity="0"/>
    <stop offset="0.5" stop-color="#ffffff" stop-opacity="0.3"/>
    <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
  </linearGradient>
  <linearGradient id="frameGrad" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{SKY}"/><stop offset="0.5" stop-color="{LINE}"/><stop offset="1" stop-color="#fda4af"/>
  </linearGradient>
  <radialGradient id="glowBlue" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{BLUE}" stop-opacity="0.4"/>
    <stop offset="0.6" stop-color="#1f5fd6" stop-opacity="0.12"/>
    <stop offset="1" stop-color="#1f5fd6" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowRed" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{RED}" stop-opacity="0.3"/>
    <stop offset="1" stop-color="{RED}" stop-opacity="0"/>
  </radialGradient>
  <pattern id="dots" width="22" height="22" patternUnits="userSpaceOnUse">
    <circle cx="1" cy="1" r="1" fill="{BLUE}" fill-opacity="0.08"/>
  </pattern>
  <filter id="bloom" x="-10%" y="-30%" width="120%" height="160%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="5" result="b"/>
    <feFlood flood-color="{BLUE}" flood-opacity="0.85"/>
    <feComposite in2="b" operator="in" result="g"/>
    <feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <mask id="nameMask" maskUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}">
    <rect width="{W}" height="{H}" fill="#000"/>
    {"".join(rows_to_text('fill="#fff"'))}
  </mask>
  <clipPath id="frame"><rect width="{W}" height="{H}" rx="16"/></clipPath>
</defs>
<style>
  text {{ font-family: {MONO}; }}
  .g {{ font-size: 10.5px; }}
  .fa, .fb {{ animation-iteration-count: infinite; animation-timing-function: steps(1, end); }}
  .fa {{ animation-name: flipA; }}
  .fb {{ animation-name: flipB; opacity: 0; }}
  .gone {{ animation: gone ease-in-out infinite; }}
  .p, .q {{ opacity: 0; transform-box: fill-box; transform-origin: center; animation-iteration-count: infinite; animation-timing-function: cubic-bezier(.2,.6,.35,1); }}
  .p {{ fill: {BLUE}; animation-name: drift; }}
  .p.hot {{ animation-name: driftHot; }}
  .q {{ fill: {SKY}; animation-name: reform; }}
  .sheen {{ animation: sweep 9s ease-in-out infinite; }}
  @keyframes flipA {{ 0% {{ opacity: 1; }} 50% {{ opacity: 0; }} }}
  @keyframes flipB {{ 0% {{ opacity: 0; }} 50% {{ opacity: 1; }} }}
  @keyframes gone {{ 0%, 55% {{ opacity: 1; }} 65%, 88% {{ opacity: 0; }} 100% {{ opacity: 1; }} }}
  @keyframes drift {{
    0% {{ transform: translate(0, 0) scale(1); opacity: 0; }}
    8% {{ opacity: .95; }}
    45% {{ transform: translate(var(--mx), var(--my)) scale(.85); }}
    100% {{ transform: translate(var(--dx), var(--dy)) scale(.35); opacity: 0; }}
  }}
  @keyframes driftHot {{
    0% {{ transform: translate(0, 0) scale(1); opacity: 0; fill: {BLUE}; }}
    8% {{ opacity: .95; }}
    45% {{ transform: translate(var(--mx), var(--my)) scale(.85); fill: {RED}; }}
    100% {{ transform: translate(var(--dx), var(--dy)) scale(.35); opacity: 0; fill: {RED}; }}
  }}
  @keyframes reform {{
    0% {{ transform: translate(var(--dx), var(--dy)) scale(.4); opacity: 0; }}
    25% {{ opacity: .8; }}
    55% {{ transform: translate(var(--mx), var(--my)) scale(.8); }}
    92% {{ transform: translate(0, 0) scale(1); opacity: .9; }}
    100% {{ transform: translate(0, 0) scale(1); opacity: 0; }}
  }}
  @keyframes sweep {{ 0% {{ transform: translateX(0); }} 70%, 100% {{ transform: translateX({name_right - name_left + 320:.0f}px); }} }}
  {cap_css}
  {REDUCED_MOTION}
</style>""")

    out.append('<g clip-path="url(#frame)">')
    out.append(f'<ellipse cx="{W / 2}" cy="{(name_top + name_bottom) / 2:.0f}" rx="600" ry="190" fill="url(#glowBlue)"/>')
    out.append(f'<ellipse cx="{name_right - 30:.0f}" cy="{name_top + 10:.0f}" rx="260" ry="130" fill="url(#glowRed)"/>')
    out.append(f'<ellipse cx="{name_left + 10:.0f}" cy="{name_bottom + 10:.0f}" rx="200" ry="100" fill="url(#glowRed)" opacity="0.7"/>')

    out.append('<g aria-hidden="true">')
    out += rows_to_text('fill="url(#ghost)"', dx=-3.0)
    out.append("</g>")
    out.append('<g filter="url(#bloom)">')
    out += rows_to_text('fill="url(#ink)"')
    out.append("</g>")
    out.append(f'<g mask="url(#nameMask)"><rect class="sheen" x="{name_left - 300:.0f}" y="{name_top:.0f}" width="160" height="{name_bottom - name_top:.0f}" fill="url(#sheenGrad)"/></g>')

    out.append('<g class="g" fill="url(#ink)">')
    for x, y in boil:
        if rng.random() > 0.30:
            out.append(f'<text x="{cx(x)}" y="{cy(y)}">{rng.choice("01")}</text>')
            continue
        flip = rng.uniform(2.4, 5.0)
        delay = -rng.uniform(0, 5.0)
        pair = (f'<text class="fa" x="{cx(x)}" y="{cy(y)}" style="animation-duration:{flip:.2f}s;animation-delay:{delay:.2f}s">0</text>'
                f'<text class="fb" x="{cx(x)}" y="{cy(y)}" style="animation-duration:{flip:.2f}s;animation-delay:{delay:.2f}s">1</text>')
        if rng.random() < 0.4:
            pair = f'<g class="gone" style="animation-duration:{rng.uniform(5, 11):.2f}s;animation-delay:-{rng.uniform(0, 11):.2f}s">{pair}</g>'
        out.append(pair)
    out.append("</g>")

    out.append('<g font-size="9.5">')
    for x, y, g, a in dust:
        fill = BLUE if rng.random() < 0.8 else RED
        out.append(f'<text x="{x}" y="{y}" fill="{fill}" fill-opacity="{a}">{esc(g)}</text>')
    out.append("</g>")
    out.append("<g>" + "".join(arriving) + "</g>")
    out.append("<g>" + "".join(leaving) + "</g>")

    out.append(cap_svg)
    out.append("</g>")
    out.append("</svg>")

    print(f"header: grid {cols}x{rows}, {sum(len(v) for v in body.values())} glyphs, {len(boil)} boiling, "
          f"{len(dust)} dust, {len(leaving)} leaving, {len(arriving)} arriving")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Summary and section titles
# ---------------------------------------------------------------------------

def summary_svg() -> str:
    W, m = 1200, 14
    lines = wrap(SUMMARY, 84)
    H = m * 2 + 52 + len(lines) * 34 + 18
    rng = random.Random(5)
    text = "".join(f'<text x="{m + 62}" y="{m + 58 + k * 34}">{esc(line)}</text>' for k, line in enumerate(lines))
    frame = rrect(m + 0.75, m + 0.75, W - 2 * m - 1.5, H - 2 * m - 1.5, 15.5)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(SUMMARY)}">
<defs>
  {card_defs(W, H, m, 16, BLUE)}
</defs>
<style>
  {REDUCED_MOTION}
</style>
<rect x="{m}" y="{m}" width="{W - 2 * m}" height="{H - 2 * m}" rx="16" fill="{PAPER}"/>
<path d="{frame}" fill="none" stroke="{LINE}" stroke-width="1.5"/>
<rect x="{m + 34}" y="{m + 34}" width="4" height="{len(lines) * 34 - 4}" rx="2" fill="{BLUE}"/>
<g font-family="{MONO}" font-size="19" fill="{INK}">{text}</g>
<g font-family="{MONO}" font-size="12" fill="{BLUE}">{bit_field(rng, W - 260, W - 40, m + 30, m + 70, 10, 0.5)}</g>
</svg>
"""


def section_svg(title: str) -> str:
    rng = random.Random(11)
    W, H = 1200, 84
    line_x = 32 + len(title) * 16.5 + 28
    bits = []
    x = 820.0
    while x < 1180:
        fade = 1 - (x - 820) / 360
        if rng.random() < 0.35 + 0.6 * fade:
            bits.append(f'<text x="{x:.0f}" y="{rng.uniform(40, 50):.0f}" fill-opacity="{max(0.08, fade * 0.7):.2f}">{rng.choice("01")}</text>')
        x += 9 + 14 * (1 - fade)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">
<text x="32" y="53" font-family="{MONO}" font-size="27" font-weight="700" fill="{INK}">{esc(title)}</text>
<rect x="{line_x:.0f}" y="44" width="{800 - line_x:.0f}" height="1.5" fill="{LINE}"/>
<g font-family="{MONO}" font-size="12" fill="{BLUE}">{"".join(bits)}</g>
</svg>
"""


# ---------------------------------------------------------------------------
# Project cards
# ---------------------------------------------------------------------------

# wide=True cards span the full README width and bookend the 2-column grid
PROJECTS = [
    {
        "slug": "cortexnet", "repo": "cortexnet", "title": "CortexNET", "accent": "green", "wide": True, "flagship": True,
        "desc": "Outline-first context for coding agents. A learned coprocessor ranks "
                "which symbol bodies the next step needs, so a small local model can "
                "work through long coding sessions on a 6 GB laptop GPU.",
        "tags": ["Python", "Agents", "Retrieval"],
    },
    {
        "slug": "samos", "repo": "SAMOS", "title": "SAMOS", "accent": "blue",
        "desc": "Turns a phone, SSD or USB device into explicit secondary memory: "
                "hot data stays in laptop RAM, cold data lives on the device at "
                "known coordinates and returns on demand, about 5 ms over USB.",
        "tags": ["Python", "Systems", "Memory"],
    },
    {
        "slug": "vault", "repo": "iqoo-hackathon", "title": "Vault", "accent": "red",
        "desc": "Turns an Android phone into a private AI coprocessor. Documents stay "
                "encrypted on-device; the laptop only receives a signed answer "
                "capsule and checks it before trusting it.",
        "tags": ["Dart", "On-device LLM", "Crypto"],
    },
    {
        "slug": "sse", "repo": "SSE-Semantic-Split-Entropy-Gating", "title": "SSE Gating", "accent": "blue",
        "desc": "CPU-only anomaly gates that learn from their own mistakes: statistical "
                "and semantic scoring plus a calibration loop. About 380 lines of "
                "numpy, ~1 ms per request on one core.",
        "tags": ["Python", "Security", "numpy"],
    },
    {
        "slug": "saber", "repo": "Amazon-ML-2026-SABER", "title": "SABER", "accent": "red",
        "desc": "Business entity resolution for the Amazon ML Challenge 2026: hybrid "
                "candidate blocking and an F0.5-tuned match selector.",
        "tags": ["Python", "Entity resolution", "ML"],
    },
    {
        "slug": "anw", "repo": "Autonomous-Neighbourhood-Watch-ANW-", "title": "Autonomous Neighbourhood Watch",
        "accent": "red", "wide": True,
        "desc": "Decentralized autonomous agents that detect abnormal behaviour from "
                "local observations and peer-to-peer messages alone, with no central "
                "controller.",
        "tags": ["Go", "Multi-agent", "P2P"],
    },
]


def card_svg(i: int, p: dict) -> str:
    rng = random.Random(100 + i)
    wide = p.get("wide", False)
    m = 14
    W, H = (1200, 258) if wide else (600, 278)
    main, light, soft, dark = ACCENTS[p["accent"]]
    other = RED if p["accent"] != "red" else BLUE
    lines = wrap(p["desc"], 66 if wide else 60)
    assert len(lines) <= 4, (p["slug"], lines)
    L = m + 34  # content left edge

    desc = "".join(f'<text x="{L}" y="{m + 100 + k * 22}">{esc(line)}</text>' for k, line in enumerate(lines))

    tags, tx, tag_y = [], float(L), H - m - 50
    for tag in p["tags"]:
        w = 20 + len(tag) * 7.6
        tags.append(
            f'<rect x="{tx:.1f}" y="{tag_y}" width="{w:.1f}" height="26" rx="13" fill="{main}" fill-opacity=".12" stroke="{main}" stroke-opacity=".45"/>'
            f'<text x="{tx + w / 2:.1f}" y="{tag_y + 17}" text-anchor="middle" fill="{light}">{esc(tag)}</text>'
        )
        tx += w + 8

    if wide:
        field = bit_field(rng, 700, W - 40, m + 40, H - m - 40, 11, 0.6)
    else:
        field = bit_field(rng, 390, W - m - 20, m + 26, m + 62, 9, 0.6)

    flag = ""
    if p.get("flagship"):
        fx = L + len(p["title"]) * 18.2 + 20
        flag = (f'<rect x="{fx:.0f}" y="{m + 37}" width="88" height="24" rx="12" fill="{dark}"/>'
                f'<text x="{fx + 44:.0f}" y="{m + 53}" text-anchor="middle" font-family="{MONO}" font-size="12" font-weight="700" fill="#fff">Flagship</text>')

    frame = rrect(m + 0.75, m + 0.75, W - 2 * m - 1.5, H - 2 * m - 1.5, 15.5)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(p['title'])}: {esc(p['desc'])}">
<defs>
  {card_defs(W, H, m, 16, main)}
</defs>
<style>
  {REDUCED_MOTION}
</style>
<rect x="{m}" y="{m}" width="{W - 2 * m}" height="{H - 2 * m}" rx="16" fill="{PAPER}"/>
<g clip-path="url(#c)">
  <rect x="{m}" y="{m}" width="{(W - 2 * m) * 0.6:.0f}" height="3" fill="{main}"/>
</g>
<path d="{frame}" fill="none" stroke="{LINE}" stroke-width="1.5"/>
<g font-family="{MONO}" font-size="12" fill="{main}">{field}</g>
<text x="{L}" y="{m + 56}" font-family="{MONO}" font-size="{30 if wide else 26}" font-weight="700" fill="{INK}">{esc(p['title'])}</text>
{flag}
<g font-family="{MONO}" font-size="13.5" fill="{INK2}">{desc}</g>
<g font-family="{MONO}" font-size="12" font-weight="600">{"".join(tags)}</g>
</svg>
"""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def load_metrics() -> dict:
    data = json.loads(METRICS.read_text(encoding="utf-8"))
    weeks = data["weeks"]
    first = next(k for k in range(len(weeks) - 26, len(weeks)) if weeks[k]["count"] > 0)
    data["recent"] = weeks[max(0, first - 1):]
    total = sum(data["languages"].values())
    top = list(data["languages"].items())[:6]
    rest = total - sum(v for _, v in top)
    data["lang_share"] = [(k, v / total) for k, v in top] + ([("Other", rest / total)] if rest else [])
    data["lang_repos"] = data["public_repos"] - len(data["excluded"])
    return data


def fmt_date(iso: str) -> str:
    d = dt.date.fromisoformat(iso)
    return f"{d.day} {d.strftime('%b %Y')}"


def metrics_svg(data: dict) -> str:
    W, H, m = 1200, 580, 14
    L, R = m + 40, W - m - 40
    out = []

    tiles = [
        (data["contributions"], "contributions, last 12 months"),
        (data["commits"], "commits"),
        (data["pull_requests"], "pull requests"),
        (data["public_repos"], "public repositories"),
    ]
    tw = (R - L) / len(tiles)
    for k, (value, label) in enumerate(tiles):
        x = L + k * tw
        if k:
            out.append(f'<rect x="{x - 20:.0f}" y="{m + 44}" width="1" height="72" fill="{LINE}"/>')
        out.append(f'<text x="{x:.0f}" y="{m + 92}" font-size="44" font-weight="700" fill="{INK}">{value}</text>')
        out.append(f'<text x="{x:.0f}" y="{m + 118}" font-size="13" fill="{INK2}">{esc(label)}</text>')
    out.append(f'<rect x="{L}" y="{m + 150}" width="{R - L}" height="1" fill="{LINE}"/>')

    # weekly contributions: one series, so no legend; the title names it and only the peak is labelled
    weeks = data["recent"]
    counts = [w["count"] for w in weeks]
    top = 5 * math.ceil((max(counts) + 1) / 5)
    px0, px1, py0, py1 = L + 34, 700, m + 240, m + 470
    out.append(f'<text x="{L}" y="{m + 190}" font-size="15" font-weight="700" fill="{INK}">Weekly contributions</text>')
    out.append(f'<text x="{L}" y="{m + 210}" font-size="12.5" fill="{MUTED}">{fmt_date(weeks[0]["start"])} to {fmt_date(data["as_of"])}</text>')
    for tick in range(0, top + 1, 5):
        y = py1 - (py1 - py0) * tick / top
        out.append(f'<rect x="{px0}" y="{y:.1f}" width="{px1 - px0}" height="1" fill="{LINE if tick else "#3a4660"}"/>')
        out.append(f'<text x="{px0 - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="11.5" fill="{MUTED}">{tick}</text>')
    band = (px1 - px0) / len(weeks)
    bw = band - 4
    peak = max(range(len(counts)), key=lambda k: (counts[k], k))
    prev_month = None
    for k, (w, c) in enumerate(zip(weeks, counts)):
        x = px0 + k * band + 2
        if c:
            h = (py1 - py0) * c / top
            r = min(4, h)
            d = f"M{x:.1f},{py1} V{py1 - h + r:.1f} Q{x:.1f},{py1 - h:.1f} {x + r:.1f},{py1 - h:.1f} H{x + bw - r:.1f} Q{x + bw:.1f},{py1 - h:.1f} {x + bw:.1f},{py1 - h + r:.1f} V{py1} Z"
            out.append(f'<path d="{d}" fill="{BLUE}"/>')
            if k == peak:
                out.append(f'<text x="{x + bw / 2:.1f}" y="{py1 - h - 8:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{INK}">{c}</text>')
        month = dt.date.fromisoformat(w["start"]).strftime("%b")
        if month != prev_month:
            out.append(f'<text x="{x:.1f}" y="{py1 + 22}" font-size="11.5" fill="{MUTED}">{month}</text>')
            prev_month = month

    # languages: one measure across categories, so a single hue; identity comes from the row labels
    lx, lbar0, lbar1 = 770, 870, R - 52
    out.append(f'<text x="{lx}" y="{m + 190}" font-size="15" font-weight="700" fill="{INK}">Languages by code size</text>')
    out.append(f'<text x="{lx}" y="{m + 210}" font-size="12.5" fill="{MUTED}">share of bytes across {data["lang_repos"]} public repos</text>')
    biggest = data["lang_share"][0][1]
    for k, (name, share) in enumerate(data["lang_share"]):
        y = m + 252 + k * 33
        w = max(6, (lbar1 - lbar0) * share / biggest)
        fill = "#4b5873" if name == "Other" else BLUE
        out.append(f'<text x="{lx}" y="{y + 10}" font-size="13" fill="{INK}">{esc(name)}</text>')
        out.append(f'<rect x="{lbar0}" y="{y}" width="{w:.1f}" height="12" rx="4" fill="{fill}"/>')
        out.append(f'<text x="{lbar0 + w + 8:.1f}" y="{y + 10}" font-size="12.5" fill="{INK2}">{share * 100:.1f}%</text>')

    excluded = ", ".join(data["excluded"])
    note = (f"Public GitHub data as of {fmt_date(data['as_of'])}. "
            f"{excluded} is left out of the language split because it is mostly third-party code.")
    out.append(f'<text x="{L}" y="{H - m - 24}" font-size="11.5" fill="{MUTED}">{esc(note)}</text>')

    frame = rrect(m + 0.75, m + 0.75, W - 2 * m - 1.5, H - 2 * m - 1.5, 15.5)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Metrics: {data['contributions']} contributions in the last 12 months, {data['commits']} commits, {data['pull_requests']} pull requests, {data['public_repos']} public repositories, a weekly contribution chart and a language breakdown.">
<defs>
  {card_defs(W, H, m, 16, BLUE)}
</defs>
<style>
  text {{ font-family: {MONO}; }}
  {REDUCED_MOTION}
</style>
<rect x="{m}" y="{m}" width="{W - 2 * m}" height="{H - 2 * m}" rx="16" fill="{PAPER}"/>
<path d="{frame}" fill="none" stroke="{LINE}" stroke-width="1.5"/>
{chr(10).join(out)}
</svg>
"""


# ---------------------------------------------------------------------------
# Page: every piece composed onto one sheet of frosted glass
# ---------------------------------------------------------------------------
# GitHub can't style a README's background, and separate images always leave
# seams of the page colour between them, so the whole README is one image.

def nest(svg: str, ns: str, x: float, y: float, width: float) -> tuple[str, float]:
    """Inline a standalone SVG at (x, y) scaled to `width`; ids get an `ns` prefix so pieces can't collide."""
    head, _, rest = svg.partition(">")
    vw, vh = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', head).groups())
    body = rest.rsplit("</svg>", 1)[0]
    body = re.sub(r"<title[^>]*>.*?</title>|<desc[^>]*>.*?</desc>", "", body, flags=re.S)
    body = re.sub(r'\bid="([\w-]+)"', lambda mt: f'id="{ns}-{mt.group(1)}"', body)
    body = re.sub(r"url\(#([\w-]+)\)", lambda mt: f"url(#{ns}-{mt.group(1)})", body)
    height = width * vh / vw
    return f'<svg x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" viewBox="0 0 {vw:g} {vh:g}">{body}</svg>', height


def frost_tile(size: int = 128) -> str:
    """A small RGBA noise tile (base64 PNG) for the frosted grain of the glass."""
    rng = np.random.default_rng(9)
    lum = rng.random((size, size))
    rgb = np.where(lum[..., None] > 0.5, 255, 0).astype(np.uint8).repeat(3, axis=2)
    alpha = (np.abs(lum - 0.5) * 2) ** 3 * 46
    img = Image.fromarray(np.dstack([rgb, alpha.astype(np.uint8)]), "RGBA")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def profile_svg(data: dict) -> str:
    W, pad = 1200, 22
    inner = W - 2 * pad
    parts: list[str] = []
    y = 6.0

    def put(svg: str, ns: str, x: float, width: float) -> float:
        el, h = nest(svg, ns, x, y, width)
        parts.append(el)
        return h

    y += put(header_svg(), "hd", 0, W) - 40
    y += put(summary_svg(), "sm", pad, inner) + 4
    y += put(section_svg("Top projects"), "s1", pad, inner)
    pair: list[tuple[int, dict]] = []
    for i, p in enumerate(PROJECTS):
        if p.get("wide"):
            y += put(card_svg(i, p), f"c{i}", pad, inner)
            continue
        pair.append((i, p))
        if len(pair) == 2:
            heights = [put(card_svg(j, q), f"c{j}", pad + k * inner / 2, inner / 2) for k, (j, q) in enumerate(pair)]
            y += max(heights)
            pair = []
    y += 4
    y += put(section_svg("Metrics"), "s2", pad, inner)
    y += put(metrics_svg(data), "mt", pad, inner) + 4
    y += put(section_svg("All repositories ↗"), "s3", pad, inner)
    H = round(y + 16)

    frame = rrect(0.75, 0.75, W - 1.5, H - 1.5, 13.5)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="pt">
<title id="pt">Yogeshwar S: profile</title>
<defs>
  <pattern id="frost" width="128" height="128" patternUnits="userSpaceOnUse">
    <image width="128" height="128" href="data:image/png;base64,{frost_tile()}"/>
  </pattern>
  <clipPath id="page"><rect width="{W}" height="{H}" rx="14"/></clipPath>
</defs>
<style>
  {REDUCED_MOTION}
</style>
<g clip-path="url(#page)">
  <rect width="{W}" height="{H}" fill="#0a0f16"/>
  <rect width="{W}" height="{H}" fill="url(#frost)"/>
  {chr(10).join(parts)}
</g>
<path d="{frame}" fill="none" stroke="{LINE}" stroke-width="1.5"/>
</svg>
"""


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def readme(data: dict) -> str:
    projects = " ".join(f"{p['title']}{' (flagship)' if p.get('flagship') else ''}: {p['desc']}" for p in PROJECTS)
    alt = (f"Yogeshwar S. {', '.join(CAPTIONS)}. {SUMMARY} Top projects. {projects} "
           f"Metrics: {data['contributions']} contributions in the last 12 months, {data['commits']} commits, "
           f"{data['pull_requests']} pull requests, {data['public_repos']} public repositories.")
    return (f'<a href="https://github.com/yogeshwars-cys?tab=repositories">'
            f'<img src="assets/profile.svg" width="100%" alt="{esc(alt)}"></a>\n')


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for old in ASSETS.glob("*.svg"):
        old.unlink()
    data = load_metrics()
    (ASSETS / "profile.svg").write_text(profile_svg(data), encoding="utf-8")
    (ROOT / "README.md").write_text(readme(data), encoding="utf-8")


if __name__ == "__main__":
    main()
