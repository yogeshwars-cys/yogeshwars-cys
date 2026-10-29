"""Generate the profile README artwork (assets/*.svg).

    python tools/build.py

The header rasterises "YOGESHWAR" onto a character grid: solid blocks inside the
letters, shading toward the edges, and binary digits that break off the edges and
drift away. Everything is plain SVG + CSS so it animates inside GitHub's <img>.
"""

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
FONT_PATH = "C:/Windows/Fonts/seguibl.ttf"  # Segoe UI Black, only used to rasterise the name

SANS = "'Segoe UI', Inter, -apple-system, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Code', SFMono-Regular, Consolas, Menlo, 'Liberation Mono', monospace"

INK = "#03060f"
BLUE = "#3d9bff"
BLUE_HOT = "#7cc8ff"
RED = "#ff3b5c"
MUTED = "#8ea3c4"

REDUCED_MOTION = "@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# Header: character-art name dissolving into binary
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


def header_svg() -> str:
    rng = random.Random(7)
    W, H = 1200, 400
    cols, cell_w, cell_h = 168, 6.0, 9.0
    cov = name_mask("YOGESHWAR", cols, cell_w, cell_h, stretch=1.3)
    pad = 4
    cov = np.pad(cov, pad)
    rows, cols = cov.shape
    filled = cov >= 0.5
    depth = depth_map(filled)

    x0 = (W - cols * cell_w) / 2
    y0 = 52.0

    def cx(x: float) -> float:
        return round(x0 + x * cell_w, 1)

    def cy(y: float) -> float:
        return round(y0 + (y + 1) * cell_h, 1)

    body: dict[int, list[tuple[int, str]]] = {}   # row -> [(col, glyph)] for the solid letters
    flicker: list[tuple[int, int, str]] = []     # edge digits that blink
    dust: list[tuple[float, float, str, float]] = []  # static detached digits around the edges
    particles: list[tuple[float, float, str, float, float, float, float, bool]] = []

    wind = (0.9, -0.45)  # dust blows up and to the right
    for y in range(rows):
        for x in range(cols):
            d = int(depth[y, x])
            if d == 0:
                # anti-aliased fringe just outside the letters
                if cov[y, x] > 0.12 and rng.random() < 0.55:
                    dust.append((cx(x), cy(y), rng.choice("01░"), round(0.35 + cov[y, x] * 0.6, 2)))
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
                    glyph = ""  # this cell has already blown away
                elif r < 0.62:
                    glyph = rng.choice("01")
                else:
                    glyph = rng.choice("▒░")

            if glyph in ("0", "1") and rng.random() < 0.35:
                flicker.append((x, y, glyph))
            elif glyph:
                body.setdefault(y, []).append((x, glyph))

            if d == 1:
                # loose digits hanging off the edge, a few cells out
                for _ in range(1 + int(downwind * 2)):
                    if rng.random() < 0.45:
                        dist = rng.uniform(1.0, 2.5 + 5.0 * downwind)
                        dx = ox * dist + wind[0] * dist * 0.8
                        dy = oy * dist * 0.6 + wind[1] * dist * 0.5
                        alpha = max(0.12, 0.9 - dist * 0.11)
                        dust.append((cx(x + dx), cy(y + dy), rng.choice("01"), round(alpha, 2)))
                # animated particles drifting away
                if rng.random() < 0.10 + 0.30 * downwind:
                    dist = rng.uniform(40, 130)
                    jitter = rng.uniform(-0.35, 0.35)
                    vx = ox * 0.6 + wind[0] + jitter
                    vy = oy * 0.6 + wind[1] + rng.uniform(-0.3, 0.2)
                    n = math.hypot(vx, vy) or 1.0
                    particles.append((
                        cx(x), cy(y), rng.choice("01"),
                        round(vx / n * dist, 1), round(vy / n * dist, 1),
                        round(rng.uniform(3.5, 7.5), 2), round(-rng.uniform(0, 7.5), 2),
                        rng.random() < 0.18,  # a share of the dust burns red
                    ))

    name_top = y0
    name_bottom = y0 + rows * cell_h
    name_left, name_right = x0, x0 + cols * cell_w

    out = []
    out.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="t d">')
    out.append('<title id="t">YOGESHWAR</title>')
    out.append('<desc id="d">The name YOGESHWAR drawn in character art, its edges disintegrating into drifting binary digits.</desc>')
    out.append(f"""<defs>
  <linearGradient id="ink" gradientUnits="userSpaceOnUse" x1="{name_left:.0f}" y1="{name_top:.0f}" x2="{name_left + 260:.0f}" y2="{name_bottom + 160:.0f}">
    <stop offset="0" stop-color="#ffffff"/>
    <stop offset="0.45" stop-color="#e8f4ff"/>
    <stop offset="0.75" stop-color="{BLUE_HOT}"/>
    <stop offset="1" stop-color="{BLUE}"/>
  </linearGradient>
  <linearGradient id="ghost" gradientUnits="userSpaceOnUse" x1="{name_left:.0f}" y1="0" x2="{name_right:.0f}" y2="0">
    <stop offset="0" stop-color="{RED}" stop-opacity="0.9"/>
    <stop offset="0.35" stop-color="{RED}" stop-opacity="0.15"/>
    <stop offset="0.7" stop-color="{RED}" stop-opacity="0"/>
    <stop offset="1" stop-color="{RED}" stop-opacity="0.55"/>
  </linearGradient>
  <linearGradient id="rule" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{BLUE}" stop-opacity="0"/>
    <stop offset="0.3" stop-color="{BLUE}"/>
    <stop offset="0.55" stop-color="#ffffff"/>
    <stop offset="0.8" stop-color="{RED}"/>
    <stop offset="1" stop-color="{RED}" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="glowBlue" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{BLUE}" stop-opacity="0.42"/>
    <stop offset="0.6" stop-color="#1f5fd6" stop-opacity="0.12"/>
    <stop offset="1" stop-color="#1f5fd6" stop-opacity="0"/>
  </radialGradient>
  <radialGradient id="glowRed" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{RED}" stop-opacity="0.32"/>
    <stop offset="1" stop-color="{RED}" stop-opacity="0"/>
  </radialGradient>
  <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse">
    <circle cx="1" cy="1" r="1" fill="#8fb4ff" fill-opacity="0.09"/>
  </pattern>
  <filter id="bloom" x="-10%" y="-30%" width="120%" height="160%">
    <feGaussianBlur in="SourceAlpha" stdDeviation="5" result="b"/>
    <feFlood flood-color="{BLUE}" flood-opacity="0.85"/>
    <feComposite in2="b" operator="in" result="g"/>
    <feMerge><feMergeNode in="g"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
  <clipPath id="frame"><rect width="{W}" height="{H}" rx="16"/></clipPath>
</defs>
<style>
  text {{ font-family: {MONO}; }}
  .g {{ font-size: 10.5px; }}
  .f {{ animation: flick 2.6s steps(1, end) infinite; }}
  .p {{ font-size: 9px; opacity: 0; animation: drift linear infinite; }}
  .caret {{ animation: blink 1.1s steps(1, end) infinite; }}
  @keyframes flick {{ 0%, 100% {{ opacity: 1; }} 40% {{ opacity: 0.25; }} 55% {{ opacity: 0.9; }} 80% {{ opacity: 0.4; }} }}
  @keyframes drift {{
    0% {{ transform: translate(0, 0); opacity: 0; }}
    12% {{ opacity: 0.95; }}
    100% {{ transform: translate(var(--dx), var(--dy)); opacity: 0; }}
  }}
  @keyframes blink {{ 0%, 49% {{ opacity: 1; }} 50%, 100% {{ opacity: 0; }} }}
  {REDUCED_MOTION}
</style>""")

    out.append('<g clip-path="url(#frame)">')
    out.append(f'<rect width="{W}" height="{H}" fill="{INK}"/>')
    out.append(f'<rect width="{W}" height="{H}" fill="url(#dots)"/>')
    out.append(f'<ellipse cx="{W / 2}" cy="{(name_top + name_bottom) / 2:.0f}" rx="560" ry="170" fill="url(#glowBlue)"/>')
    out.append(f'<ellipse cx="{name_right - 40:.0f}" cy="{name_top + 20:.0f}" rx="230" ry="120" fill="url(#glowRed)"/>')
    out.append(f'<ellipse cx="{name_left + 20:.0f}" cy="{name_bottom:.0f}" rx="160" ry="80" fill="url(#glowRed)" opacity="0.6"/>')

    def rows_to_text(cells: dict[int, list[tuple[int, str]]], attrs: str, dx: float = 0.0) -> None:
        for y, items in sorted(cells.items()):
            xs = " ".join(f"{cx(x) + dx:.1f}" for x, _ in items)
            chars = "".join(g for _, g in items)
            out.append(f'<text class="g" x="{xs}" y="{cy(y)}" {attrs}>{esc(chars)}</text>')

    # red chromatic ghost, offset to the left
    out.append('<g aria-hidden="true">')
    rows_to_text(body, 'fill="url(#ghost)"', dx=-3.0)
    out.append("</g>")
    # the solid letters, with bloom
    out.append('<g filter="url(#bloom)">')
    rows_to_text(body, 'fill="url(#ink)"')
    out.append("</g>")

    out.append('<g fill="url(#ink)">')
    for i, (x, y, g) in enumerate(flicker):
        delay = -round(rng.uniform(0, 2.6), 2)
        out.append(f'<text class="g f" x="{cx(x)}" y="{cy(y)}" style="animation-delay:{delay}s">{g}</text>')
    out.append("</g>")

    out.append('<g font-size="9.5">')
    for x, y, g, a in dust:
        fill = BLUE_HOT if rng.random() < 0.8 else RED
        out.append(f'<text x="{x}" y="{y}" fill="{fill}" fill-opacity="{a}">{esc(g)}</text>')
    out.append("</g>")

    out.append("<g>")
    for x, y, g, dx, dy, dur, delay, red in particles:
        fill = RED if red else BLUE_HOT
        out.append(
            f'<text class="p" x="{x}" y="{y}" fill="{fill}" '
            f'style="--dx:{dx}px;--dy:{dy}px;animation-duration:{dur}s;animation-delay:{delay}s">{g}</text>'
        )
    out.append("</g>")

    # rule + tagline
    rule_y = name_bottom + 28
    out.append(f'<rect x="{W / 2 - 360}" y="{rule_y:.0f}" width="720" height="1.5" fill="url(#rule)"/>')
    tag_y = rule_y + 40
    out.append(
        f'<text x="{W / 2}" y="{tag_y:.0f}" text-anchor="middle" font-size="15" fill="#c9d8ef" letter-spacing="0.5">'
        f'I build AI, security and distributed systems that run close to the data'
        f'<tspan class="caret" fill="{BLUE_HOT}"> ▍</tspan></text>'
    )
    out.append("</g>")
    out.append(f'<rect x="0.75" y="0.75" width="{W - 1.5}" height="{H - 1.5}" rx="15.5" fill="none" stroke="url(#rule)" stroke-opacity="0.45" stroke-width="1.5"/>')
    out.append("</svg>")

    print(f"header: grid {cols}x{rows}, {sum(len(v) for v in body.values())} glyphs, "
          f"{len(flicker)} flicker, {len(dust)} dust, {len(particles)} particles")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Section title
# ---------------------------------------------------------------------------

def section_svg(title: str) -> str:
    rng = random.Random(11)
    W, H = 1200, 84
    line_x = 32 + len(title) * 16.5 + 28
    bits = []
    x = 820.0
    while x < 1180:
        fade = 1 - (x - 820) / 360
        if rng.random() < 0.35 + 0.6 * fade:
            bits.append(f'<text x="{x:.0f}" y="{rng.uniform(40, 50):.0f}" fill-opacity="{max(0.08, fade * 0.8):.2f}">{rng.choice("01")}</text>')
        x += 9 + 14 * (1 - fade)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(title)}">
<defs>
  <linearGradient id="tt" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="{BLUE_HOT}"/>
  </linearGradient>
  <linearGradient id="ln" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{BLUE}"/><stop offset="0.6" stop-color="{RED}"/><stop offset="1" stop-color="{RED}" stop-opacity="0"/>
  </linearGradient>
</defs>
<rect width="{W}" height="{H}" rx="16" fill="{INK}"/>
<text x="32" y="53" font-family="{MONO}" font-size="27" font-weight="700" fill="url(#tt)">{esc(title)}</text>
<rect x="{line_x:.0f}" y="44" width="{800 - line_x:.0f}" height="1.5" fill="url(#ln)"/>
<g font-family="{MONO}" font-size="12" fill="{BLUE_HOT}">{"".join(bits)}</g>
</svg>
"""


# ---------------------------------------------------------------------------
# Project cards
# ---------------------------------------------------------------------------

# wide=True cards span the full README width and bookend the 2-column grid
PROJECTS = [
    {
        "slug": "cortexnet", "repo": "cortexnet", "title": "CortexNET", "accent": "blue", "wide": True,
        "desc": "Outline-first context for coding agents. A learned coprocessor ranks "
                "which symbol bodies the next step needs, so a small local model can "
                "work through long coding sessions on a 6 GB laptop GPU.",
        "tags": ["Python", "Agents", "Retrieval"],
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
        "slug": "pramana", "repo": "Pramana-AI-Risk-Manager", "title": "Pramana", "accent": "blue",
        "desc": "Cross-merchant fraud detection where merchants share conclusions and "
                "hashed sketches, never data, and only first-hand evidence can "
                "freeze a merchant.",
        "tags": ["Python", "Fraud", "Privacy"],
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


def bit_field(rng: random.Random, x0: float, x1: float, y0: float, y1: float, step: float) -> str:
    """A block of binary that is dense on the left and breaks apart to the right."""
    out = []
    y = y0
    while y <= y1:
        x = x0 + rng.uniform(0, step)
        while x < x1:
            t = (x - x0) / (x1 - x0)
            if rng.random() < 0.92 - t * 0.85:
                lift = t * t * rng.uniform(0, 18)
                out.append(f'<text x="{x:.0f}" y="{y - lift:.0f}" fill-opacity="{max(0.08, 0.7 - t * 0.6):.2f}">{rng.choice("01")}</text>')
            x += step * (1 + t * 1.6)
        y += step * 1.7
    return "".join(out)


def card_svg(i: int, p: dict) -> str:
    rng = random.Random(100 + i)
    wide = p.get("wide", False)
    W, H = (1200, 230) if wide else (600, 250)
    hot = RED if p["accent"] == "red" else BLUE
    cool = BLUE if p["accent"] == "red" else RED
    lines = wrap(p["desc"], 66 if wide else 62)
    assert len(lines) <= 4, (p["slug"], lines)

    title_y = 70
    desc_y = 108
    desc = "".join(f'<text x="34" y="{desc_y + k * 22}">{esc(line)}</text>' for k, line in enumerate(lines))

    tags, tx, tag_y = [], 34.0, H - 50
    for k, tag in enumerate(p["tags"]):
        w = 20 + len(tag) * 7.6
        first = k == 0
        tags.append(
            f'<rect x="{tx:.1f}" y="{tag_y}" width="{w:.1f}" height="26" rx="13" '
            f'fill="{hot if first else "#0c1428"}" fill-opacity="{0.16 if first else 1}" '
            f'stroke="{hot if first else "#2a3a5c"}" stroke-opacity="0.85"/>'
            f'<text x="{tx + w / 2:.1f}" y="{tag_y + 17}" text-anchor="middle">{esc(tag)}</text>'
        )
        tx += w + 8

    if wide:
        field = bit_field(rng, 700, 1170, 50, 190, 11)
        glow = f'<ellipse cx="900" cy="{H / 2}" rx="380" ry="150" fill="url(#gl)"/>'
    else:
        field = bit_field(rng, 400, 580, 34, 70, 9)
        glow = f'<ellipse cx="{80 if p["accent"] == "blue" else 520}" cy="10" rx="260" ry="130" fill="url(#gl)"/>'

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(p['title'])}: {esc(p['desc'])}">
<defs>
  <linearGradient id="bd" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{hot}" stop-opacity="0.9"/>
    <stop offset="0.5" stop-color="#ffffff" stop-opacity="0.16"/>
    <stop offset="1" stop-color="{cool}" stop-opacity="0.55"/>
  </linearGradient>
  <linearGradient id="tt" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="{BLUE_HOT}"/>
  </linearGradient>
  <linearGradient id="bar" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{hot}"/><stop offset="1" stop-color="{hot}" stop-opacity="0"/>
  </linearGradient>
  <radialGradient id="gl" cx="0.5" cy="0.5" r="0.5">
    <stop offset="0" stop-color="{hot}" stop-opacity="0.26"/><stop offset="1" stop-color="{hot}" stop-opacity="0"/>
  </radialGradient>
  <clipPath id="c"><rect width="{W}" height="{H}" rx="16"/></clipPath>
</defs>
<g clip-path="url(#c)">
  <rect width="{W}" height="{H}" fill="{INK}"/>
  {glow}
  <rect x="0" y="0" width="{W * 0.6:.0f}" height="3" fill="url(#bar)"/>
</g>
<rect x="0.75" y="0.75" width="{W - 1.5}" height="{H - 1.5}" rx="15.5" fill="none" stroke="url(#bd)" stroke-width="1.5"/>
<g font-family="{MONO}" font-size="12" fill="{BLUE_HOT}">{field}</g>
<text x="34" y="{title_y}" font-family="{MONO}" font-size="{30 if wide else 26}" font-weight="700" fill="url(#tt)">{esc(p['title'])}</text>
<g font-family="{MONO}" font-size="13.5" fill="#aebfd9">{desc}</g>
<g font-family="{MONO}" font-size="12" fill="#d6e2f5">{"".join(tags)}</g>
</svg>
"""


def readme() -> str:
    def link(p: dict, width: str) -> str:
        return (f'<a href="https://github.com/yogeshwars-cys/{p["repo"]}">'
                f'<img src="assets/card-{p["slug"]}.svg" width="{width}" alt="{esc(p["title"])}: {esc(p["desc"])}"></a>')

    blocks, pending = [], []
    for p in PROJECTS:
        if p.get("wide"):
            if pending:
                blocks.append(pending)
                pending = []
            blocks.append([p])
        else:
            pending.append(p)
            if len(pending) == 2:
                blocks.append(pending)
                pending = []
    if pending:
        blocks.append(pending)

    grid = []
    for block in blocks:
        width = "100%" if len(block) == 1 else "49%"
        cells = "\n".join(f"  {link(p, width)}" for p in block)
        grid.append(f'<p align="center">\n{cells}\n</p>')

    return f"""<p align="center">
  <img src="assets/header.svg" width="100%" alt="YOGESHWAR, written in character art that dissolves into binary at the edges">
</p>

<p align="center">
  <img src="assets/section-projects.svg" width="100%" alt="Top projects">
</p>

{chr(10).join(grid)}

<p align="center">
  <a href="https://github.com/yogeshwars-cys?tab=repositories"><img src="assets/section-more.svg" width="100%" alt="All repositories"></a>
</p>
"""


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    for old in ASSETS.glob("card-*.svg"):
        old.unlink()
    (ASSETS / "header.svg").write_text(header_svg(), encoding="utf-8")
    (ASSETS / "section-projects.svg").write_text(section_svg("Top projects"), encoding="utf-8")
    (ASSETS / "section-more.svg").write_text(section_svg("All repositories ↗"), encoding="utf-8")
    for i, p in enumerate(PROJECTS):
        (ASSETS / f"card-{p['slug']}.svg").write_text(card_svg(i, p), encoding="utf-8")
    (ROOT / "README.md").write_text(readme(), encoding="utf-8")


if __name__ == "__main__":
    main()
