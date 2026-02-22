"""
3D Animated Visualization of Segmentation Heads
================================================
Cycles through all 7 heads from the offroad_training_pipeline:

  1. linear_head       – single 1×1 conv baseline
  2. mlp_head          – two-layer MLP via 1×1 convs
  3. convnext_head     – ConvNeXt stem + DW-PW block
  4. convnext_deep_head– ConvNeXt stem + 2 residual blocks
  5. multiscale_head   – parallel 1/3/5/7 branches + fuse
  6. hybrid_head       – ASPP + SE + spatial-attn + residual decoder
  7. segformer_head    – transformer encoder + MLP decoder

Features:
  • Proper fan-out / fan-in connectors for parallel branches
  • Sequential left→right particle flow with transition graph
  • Particle arcs for visual depth
  • Progress bar & active-step label
  • Subtle floor grid for 3-D grounding

Dependencies:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Palette ────────────────────────────────────────────────────────────────────
BG           = "#07080f"
BACKBONE_COL = "#3a86ff"

HEAD_COLORS = {
    "linear_head":        "#ff006e",
    "mlp_head":           "#fb5607",
    "convnext_head":      "#ffbe0b",
    "convnext_deep_head": "#8338ec",
    "multiscale_head":    "#00f5d4",
    "hybrid_head":        "#f72585",
    "segformer_head":     "#4cc9f0",
}

# ── Architecture definitions ───────────────────────────────────────────────────
# Each layer: (label, relative_width, color)

ARCHITECTURES = {

    "linear_head": {
        "title": "Linear Head  ·  1×1 Conv baseline",
        "desc":  "Simplest possible head. One 1×1 conv directly maps\n"
                 "each patch token to class logits. No spatial context.",
        "layers": [
            ("DINOv2\nTokens\n(B,N,C)", 1.2, BACKBONE_COL),
            ("Reshape\n→ spatial",       0.6, "#aaaaaa"),
            ("Conv2d\n1×1",              1.2, "#ff006e"),
            ("Logits",                   0.5, "#ffffff"),
        ],
    },

    "mlp_head": {
        "title": "MLP Head  ·  Two hidden layers",
        "desc":  "Per-token MLP: two 1×1 conv layers with GELU.\n"
                 "More capacity than linear, still purely local.",
        "layers": [
            ("DINOv2\nTokens",  1.2, BACKBONE_COL),
            ("Reshape",         0.6, "#aaaaaa"),
            ("Conv 1×1\n+ GELU", 1.4, "#fb5607"),
            ("Conv 1×1\n+ GELU", 1.4, "#fb5607"),
            ("Conv 1×1\nLogits", 0.5, "#ffffff"),
        ],
    },

    "convnext_head": {
        "title": "ConvNeXt Head  ·  DW-PW block",
        "desc":  "Stem (7×7 conv) + one DW-PW block.\n"
                 "Captures local spatial structure around each token.",
        "layers": [
            ("DINOv2\nTokens",         1.2, BACKBONE_COL),
            ("Reshape",                0.6, "#aaaaaa"),
            ("Stem\n7×7 Conv\n+ GELU", 1.5, "#ffbe0b"),
            ("DWConv\n7×7",            1.3, "#ffbe0b"),
            ("PWConv\n1×1",            1.3, "#e6a500"),
            ("Classifier\n1×1",        0.5, "#ffffff"),
        ],
    },

    "convnext_deep_head": {
        "title": "ConvNeXt Deep Head  ·  2 Residual Blocks",
        "desc":  "Stem + TWO ConvNeXt residual blocks (BN added).\n"
                 "More depth & skip connections for richer features.",
        "layers": [
            ("DINOv2\nTokens",          1.2, BACKBONE_COL),
            ("Reshape",                 0.6, "#aaaaaa"),
            ("Stem\n7×7+BN",            1.5, "#8338ec"),
            ("Resid.\nBlock 1\nDW+PW",  1.6, "#8338ec"),
            ("Resid.\nBlock 2\nDW+PW",  1.6, "#6a1fd0"),
            ("Classifier\n1×1",         0.5, "#ffffff"),
        ],
    },

    "multiscale_head": {
        "title": "MultiScale Head  ·  Parallel Branches",
        "desc":  "4 parallel branches (kernels 1,3,5,7) capture\n"
                 "different receptive fields, then concatenated & fused.",
        "layers": [
            ("DINOv2\nTokens",          1.2, BACKBONE_COL),
            ("Reshape",                 0.6, "#aaaaaa"),
            ("Branch\n1×1",             1.0, "#00f5d4"),
            ("Branch\n3×3",             1.0, "#00d4b8"),
            ("Branch\n5×5",             1.0, "#00b39c"),
            ("Branch\n7×7",             1.0, "#009280"),
            ("Concat\n+Fuse\n3×3→1×1",  1.8, "#00f5d4"),
            ("Logits",                  0.5, "#ffffff"),
        ],
        "parallel": (2, 6),   # block indices 2-5 are parallel branches
    },

    "hybrid_head": {
        "title": "Hybrid Head  ·  ASPP + Attention + Residual",
        "desc":  "Most complex head: dilated ASPP context, SE channel\n"
                 "attention, spatial attention gate, 2 residual blocks.",
        "layers": [
            ("DINOv2\nTokens",             1.2, BACKBONE_COL),
            ("Reshape",                    0.6, "#aaaaaa"),
            ("Proj\n1×1+BN",               1.3, "#f72585"),
            ("ASPP\nd=1,3,6,12\n+GlobalPool", 2.0, "#c61a6d"),
            ("SE\nChannel\nAttention",     1.4, "#f72585"),
            ("Spatial\nAttention\n7×7",    1.4, "#ff6eb4"),
            ("Resid.\nBlock×2",            1.5, "#c61a6d"),
            ("Classifier\n3×3→1×1",        1.2, "#ffffff"),
        ],
    },

    "segformer_head": {
        "title": "SegFormer Head  ·  Transformer Encoder",
        "desc":  "Projects tokens → Transformer blocks (efficient\n"
                 "self-attn + Mix-FFN) → MLP decoder. Global context.",
        "layers": [
            ("DINOv2\nTokens",          1.2, BACKBONE_COL),
            ("Linear\nProject\n+LN",    1.3, "#4cc9f0"),
            ("Xfmr Blk\nsr=2\n(×2)",   1.8, "#4cc9f0"),
            ("Xfmr Blk\nsr=1\n(×2)",   1.8, "#23a8d8"),
            ("LayerNorm",               0.8, "#4cc9f0"),
            ("Reshape\n→ spatial",      0.6, "#aaaaaa"),
            ("MLP\nDecoder\n3×3→1×1",   1.5, "#0096c7"),
            ("Logits",                  0.5, "#ffffff"),
        ],
    },
}

MODEL_ORDER  = list(ARCHITECTURES.keys())
PHASE_FRAMES = 160          # frames per model (slightly more for smoother flow)


# ── 3-D box helper ─────────────────────────────────────────────────────────────

def cuboid(ax, x, y, z, dx, dy, dz, color, alpha=0.35):
    """Draw a filled 3-D cuboid and return the Poly3DCollection."""
    verts = [
        [[x,    y,    z   ], [x+dx, y,    z   ], [x+dx, y+dy, z   ], [x,    y+dy, z   ]],
        [[x,    y,    z+dz], [x+dx, y,    z+dz], [x+dx, y+dy, z+dz], [x,    y+dy, z+dz]],
        [[x,    y,    z   ], [x+dx, y,    z   ], [x+dx, y,    z+dz], [x,    y,    z+dz]],
        [[x,    y+dy, z   ], [x+dx, y+dy, z   ], [x+dx, y+dy, z+dz], [x,    y+dy, z+dz]],
        [[x,    y,    z   ], [x,    y+dy, z   ], [x,    y+dy, z+dz], [x,    y,    z+dz]],
        [[x+dx, y,    z   ], [x+dx, y+dy, z   ], [x+dx, y+dy, z+dz], [x+dx, y,    z+dz]],
    ]
    poly = Poly3DCollection(verts, alpha=alpha, linewidth=0.5)
    poly.set_facecolor(color)
    poly.set_edgecolor(color)
    ax.add_collection3d(poly)
    return poly


# ── Particle ───────────────────────────────────────────────────────────────────

class Particle:
    """A single animated dot flowing between two blocks."""
    __slots__ = ("sx", "sy", "sz", "ex", "ey", "ez",
                 "t", "speed", "color", "size", "arc")

    def __init__(self, sx, sy, sz, ex, ey, ez, color):
        self.sx, self.sy, self.sz = sx, sy, sz
        self.ex, self.ey, self.ez = ex, ey, ez
        self.t     = 0.0
        self.speed = np.random.uniform(0.04, 0.09)
        self.color = color
        self.size  = float(np.random.uniform(6, 18))
        self.arc   = float(np.random.uniform(0.06, 0.22))   # vertical arc height

    @property
    def done(self):
        return self.t >= 1.0

    def step(self):
        self.t = min(self.t + self.speed, 1.0)

    @property
    def pos(self):
        s = self.t ** 2 * (3 - 2 * self.t)        # smoothstep
        return (
            self.sx + (self.ex - self.sx) * s,
            self.sy + (self.ey - self.sy) * s,
            self.sz + (self.ez - self.sz) * s + np.sin(s * np.pi) * self.arc,
        )


# ── Layout builder ─────────────────────────────────────────────────────────────

def layout_layers(arch_key):
    """Return (blocks, parallel_range).

    blocks : list of (x, y, z, dx, dy, dz, color, label)
    parallel_range : None  or  (start_idx, end_idx_exclusive) in *blocks* list
    """
    arch   = ARCHITECTURES[arch_key]
    layers = arch["layers"]
    par_spec = arch.get("parallel", None)       # (layer_start, layer_end)

    DEPTH  = 1.8
    HEIGHT = 2.2
    GAP    = 0.45

    blocks = []
    x = 0.0

    if par_spec is not None:
        p_start, p_end = par_spec
        pre  = layers[:p_start]
        par  = layers[p_start:p_end]
        post = layers[p_end:]

        # ── pre-parallel ──
        for label, w, col in pre:
            blocks.append((x, -DEPTH/2, -HEIGHT/2, w, DEPTH, HEIGHT, col, label))
            x += w + GAP

        # ── parallel stack (offset in z) ──
        n = len(par)
        z_offsets = np.linspace(-HEIGHT * (n - 1) / 2 * 0.6,
                                 HEIGHT * (n - 1) / 2 * 0.6, n)
        par_w = max(p[1] for p in par)
        for i, (label, w, col) in enumerate(par):
            zo = z_offsets[i]
            h2 = HEIGHT * 0.55
            blocks.append((x, -DEPTH/2, zo - h2/2, w, DEPTH, h2, col, label))
        x += par_w + GAP

        # ── post-parallel ──
        for label, w, col in post:
            blocks.append((x, -DEPTH/2, -HEIGHT/2, w, DEPTH, HEIGHT, col, label))
            x += w + GAP
    else:
        for label, w, col in layers:
            blocks.append((x, -DEPTH/2, -HEIGHT/2, w, DEPTH, HEIGHT, col, label))
            x += w + GAP

    # centre around x = 0
    total = x - GAP
    shift = total / 2
    centred = [(b[0] - shift, *b[1:]) for b in blocks]
    return centred, par_spec          # par_spec indices match block indices


# ── Transition graph ───────────────────────────────────────────────────────────

def get_transitions(n_blocks, parallel_range):
    """Build an *ordered* list of transitions.

    Each transition is a list of ``(src_block_idx, dst_block_idx)`` pairs.
    Sequential gaps produce single-pair transitions.
    Parallel fan-out / fan-in produce multi-pair transitions.
    """
    if parallel_range is None or n_blocks < 2:
        return [[(i, i + 1)] for i in range(n_blocks - 1)]

    ps, pe = parallel_range
    transitions = []

    # pre-parallel sequential
    for i in range(ps - 1):
        transitions.append([(i, i + 1)])

    # fan-out: last pre-block → every parallel block
    if ps > 0:
        transitions.append([(ps - 1, j) for j in range(ps, pe)])

    # fan-in: every parallel block → first post-block
    if pe < n_blocks:
        transitions.append([(j, pe) for j in range(ps, pe)])

    # post-parallel sequential
    for i in range(pe, n_blocks - 1):
        transitions.append([(i, i + 1)])

    return transitions


# ── Main animation ─────────────────────────────────────────────────────────────

def build():
    fig = plt.figure(figsize=(19, 9), facecolor=BG)
    ax  = fig.add_subplot(111, projection="3d", facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")
    ax.view_init(elev=22, azim=-50)

    # ── text overlays ──
    title_t = fig.text(0.5, 0.95, "", ha="center", color="white",
                       fontsize=17, fontweight="bold", fontfamily="monospace")
    desc_t  = fig.text(0.5, 0.88, "", ha="center", color="#aaaacc",
                       fontsize=9.5, fontfamily="monospace")
    fig.text(0.5, 0.02,
             "DINOv2 (frozen) + Segmentation Head (trained)"
             "  ·  off-road terrain perception",
             ha="center", color="#555577", fontsize=9, fontfamily="monospace")
    step_t  = fig.text(0.5, 0.83, "", ha="center", color="#666688",
                       fontsize=8, fontfamily="monospace")

    # ── progress dots (one per model) ──
    dot_artists = []
    for i, key in enumerate(MODEL_ORDER):
        col = HEAD_COLORS[key]
        dot = fig.text(0.5 + (i - len(MODEL_ORDER) / 2 + 0.5) * 0.065, 0.06,
                       "●", ha="center", color=col, fontsize=10,
                       fontfamily="monospace", alpha=0.25)
        dot_artists.append(dot)

    # ── progress bar (thin axes at bottom) ──
    prog_ax = fig.add_axes([0.2, 0.045, 0.6, 0.005])
    prog_ax.set_xlim(0, 1); prog_ax.set_ylim(0, 1)
    prog_ax.axis("off")
    prog_ax.set_facecolor("#111120")
    prog_bar = prog_ax.barh(0.5, 0, height=1.0, color=BACKBONE_COL,
                            align="center")

    # ── particle scatter (seed with invisible dummy) ──
    scat = ax.scatter([0], [0], [0], s=np.array([1.]),
                      c=[BG], alpha=0.0, depthshade=False)

    # ── mutable state ──
    state = {
        "frame":       0,
        "model_idx":   0,
        "blocks":      [],
        "particles":   [],
        "transitions": [],
    }

    drawn_artists = []        # everything we need to remove on model switch

    # ─────────────────────────────────────────── scene helpers ──
    def clear_scene():
        for a in drawn_artists:
            try:
                a.remove()
            except Exception:
                pass
        drawn_artists.clear()
        ax.set_xlim(-8, 8)
        ax.set_ylim(-3, 3)
        ax.set_zlim(-3, 3)

    def _connector(b1, b2):
        """Draw a thin line from right-centre of *b1* to left-centre of *b2*."""
        x0 = b1[0] + b1[3]                    # right edge
        x1 = b2[0]                             # left edge
        y0, y1 = b1[1] + b1[4] / 2, b2[1] + b2[4] / 2
        z0, z1 = b1[2] + b1[5] / 2, b2[2] + b2[5] / 2
        line, = ax.plot([x0, x1], [y0, y1], [z0, z1],
                        color="#444466", linewidth=0.8, alpha=0.50)
        drawn_artists.append(line)

    def _floor_grid():
        """Subtle wireframe on the floor plane for 3-D grounding."""
        for xi in np.linspace(-7, 7, 8):
            ln, = ax.plot([xi, xi], [-3, 3], [-3, -3],
                          color="#111122", lw=0.3, alpha=0.20)
            drawn_artists.append(ln)
        for yi in np.linspace(-3, 3, 4):
            ln, = ax.plot([-7, 7], [yi, yi], [-3, -3],
                          color="#111122", lw=0.3, alpha=0.20)
            drawn_artists.append(ln)

    # ─────────────────────────────────────────── draw model ──
    def draw_model(key):
        clear_scene()
        blocks, par_range = layout_layers(key)
        state["blocks"]      = blocks
        state["transitions"] = get_transitions(len(blocks), par_range)

        arch = ARCHITECTURES[key]
        title_t.set_text(arch["title"])
        desc_t.set_text(arch["desc"])

        for i, d in enumerate(dot_artists):
            d.set_alpha(1.0 if i == state["model_idx"] else 0.22)

        # draw blocks + labels
        for bx, by, bz, dx, dy, dz, col, label in blocks:
            poly = cuboid(ax, bx, by, bz, dx, dy, dz, col, alpha=0.32)
            drawn_artists.append(poly)
            drawn_artists.append(
                ax.text(bx + dx / 2, by + dy + 0.05, bz + dz / 2,
                        label, color=col, fontsize=6.5,
                        ha="center", va="bottom",
                        fontfamily="monospace", fontweight="bold")
            )

        # draw connectors (proper fan-out / fan-in for parallel blocks)
        if par_range is None:
            for i in range(1, len(blocks)):
                _connector(blocks[i - 1], blocks[i])
        else:
            ps, pe = par_range
            # pre-parallel sequential
            for i in range(1, ps):
                _connector(blocks[i - 1], blocks[i])
            # fan-out: last pre → each parallel
            if ps > 0:
                for j in range(ps, pe):
                    _connector(blocks[ps - 1], blocks[j])
            # fan-in: each parallel → first post
            if pe < len(blocks):
                for j in range(ps, pe):
                    _connector(blocks[j], blocks[pe])
            # post-parallel sequential
            for i in range(pe + 1, len(blocks)):
                _connector(blocks[i - 1], blocks[i])

        _floor_grid()

    # ─────────────────────────────────────────── initial draw ──
    draw_model(MODEL_ORDER[0])

    # ─────────────────────────────────────────── animate ──
    def animate(_):
        fc = state["frame"]
        state["frame"] += 1

        mi          = state["model_idx"]
        key         = MODEL_ORDER[mi]
        blocks      = state["blocks"]
        particles   = state["particles"]
        transitions = state["transitions"]
        color       = HEAD_COLORS[key]

        local = fc % PHASE_FRAMES

        # ── switch model on phase boundary ──
        if local == 0 and fc > 0:
            state["model_idx"] = (mi + 1) % len(MODEL_ORDER)
            state["particles"].clear()
            draw_model(MODEL_ORDER[state["model_idx"]])
            mi          = state["model_idx"]
            key         = MODEL_ORDER[mi]
            blocks      = state["blocks"]
            transitions = state["transitions"]
            color       = HEAD_COLORS[key]

        # ── progress bar ──
        progress = local / PHASE_FRAMES
        prog_bar[0].set_width(progress)
        prog_bar[0].set_color(color)

        # ── sequential particle flow ──
        if transitions:
            n_trans     = len(transitions)
            segment_len = max(PHASE_FRAMES // n_trans, 1)
            active_idx  = min(local // segment_len, n_trans - 1)
            trans       = transitions[active_idx]

            # step label
            if len(trans) == 1:
                si, di = trans[0]
                src_lbl = blocks[si][7].split("\n")[0]
                dst_lbl = blocks[di][7].split("\n")[0]
                step_t.set_text(f"▸ {src_lbl}  →  {dst_lbl}")
            elif len(trans) > 1:
                if all(t[0] == trans[0][0] for t in trans):
                    src_lbl = blocks[trans[0][0]][7].split("\n")[0]
                    step_t.set_text(
                        f"▸ {src_lbl}  →  {len(trans)} branches  (fan-out)")
                else:
                    dst_lbl = blocks[trans[0][1]][7].split("\n")[0]
                    step_t.set_text(
                        f"▸ {len(trans)} branches  →  {dst_lbl}  (fan-in)")

            # spawn particles every 2 frames
            if local % 2 == 0:
                for src_idx, dst_idx in trans:
                    src = blocks[src_idx]
                    dst = blocks[dst_idx]
                    sx  = src[0] + src[3]                  # right edge
                    ex  = dst[0]                           # left edge
                    for _ in range(2):
                        sy = np.random.uniform(src[1], src[1] + src[4])
                        sz = np.random.uniform(src[2], src[2] + src[5])
                        ey = np.random.uniform(dst[1], dst[1] + dst[4])
                        ez = np.random.uniform(dst[2], dst[2] + dst[5])
                        particles.append(
                            Particle(sx, sy, sz, ex, ey, ez, src[6]))
        else:
            step_t.set_text("")

        # ── step & cull particles ──
        for p in particles:
            p.step()
        alive = [p for p in particles if not p.done]
        state["particles"] = alive

        # ── update scatter ──
        if alive:
            xs = np.array([p.pos[0] for p in alive], dtype=np.float64)
            ys = np.array([p.pos[1] for p in alive], dtype=np.float64)
            zs = np.array([p.pos[2] for p in alive], dtype=np.float64)
            cs = [p.color for p in alive]
            ss = np.array([p.size  for p in alive], dtype=np.float64)
            al = 0.85
        else:
            xs = np.array([0.]); ys = np.array([0.]); zs = np.array([0.])
            cs = [BG]; ss = np.array([1.]); al = 0.0

        scat._offsets3d = (xs, ys, zs)
        scat.set_sizes(ss.ravel())
        scat.set_facecolor(cs)
        scat.set_alpha(al)

        # ── slow camera orbit ──
        ax.view_init(elev=20 + 5 * np.sin(fc * 0.012),
                     azim=-50 + fc * 0.10)

        return scat,

    anim = FuncAnimation(fig, animate,
                         frames=PHASE_FRAMES * len(MODEL_ORDER) + 10,
                         interval=40, blit=False)
    return fig, anim


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    fig, anim = build()

    anim.save("segheads.mp4", writer="ffmpeg", fps=25, dpi=150)

    plt.tight_layout()
    plt.show()
