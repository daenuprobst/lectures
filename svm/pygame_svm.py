import sys, math
import pygame
import numpy as np
from sklearn.svm import SVC, LinearSVC

W, H, PANEL_W, MESH = 1000, 700, 260, 6
CANVAS_W = W - PANEL_W

BG = (245, 243, 255)
GRID_C = (230, 228, 245)
BOUNDARY = (40, 40, 40)
M_POS = (200, 80, 80)
M_NEG = (80, 120, 220)
CAT_REGION = (255, 228, 205)
DOG_REGION = (208, 222, 255)
SV_RING = (255, 215, 0)
PANEL_BG = (30, 28, 50, 220)
FG = (240, 235, 255)
ACCENT = (180, 150, 255)
CAT_C = (255, 175, 95)
DOG_C = (120, 165, 245)


# Some very, very basic pixel art ...
SCALE = 2

CAT_SPRITE = [
    ".EE...EE.",
    ".EI...IE.",
    ".BBBBBBB.",
    "BBBBBBBBB",
    "BBKBBBKBB",
    "BBBBBBBBB",
    "BBBBnBBBB",
    "BwBBMBBwB",
    "BBBBBBBBB",
    ".BBBBBBB.",
    "..BBBBB..",
]

CAT_PAL = {
    "B": (255, 175, 95),
    "E": (210, 115, 50),
    "I": (255, 150, 150),
    "K": (50, 28, 8),
    "n": (195, 70, 60),
    "w": (180, 120, 75),
    "M": (160, 78, 48),
}

DOG_SPRITE = [
    "ee.....ee",
    "eebbbbbee",
    "eebbbbbee",
    "bbbbbbbbb",
    "bbkbbbkbb",
    "bbbbbbbbb",
    "bbbsssbbb",
    "bbssnssbb",
    "bbbsssbbb",
    ".bbbbbbb.",
    "..bbbbb..",
]
DOG_PAL = {
    "b": (120, 165, 245),
    "e": (75, 115, 205),
    "k": (35, 22, 45),
    "s": (225, 200, 172),
    "n": (45, 30, 52),
}


def bake(sprite, palette):
    cols = max(len(r) for r in sprite)
    surf = pygame.Surface((cols * SCALE, len(sprite) * SCALE), pygame.SRCALPHA)
    for ri, row in enumerate(sprite):
        for ci, ch in enumerate(row):
            if ch in palette:
                pygame.draw.rect(
                    surf, palette[ch], (ci * SCALE, ri * SCALE, SCALE, SCALE)
                )
    return surf


def draw_sprite(surf, spr, cx, cy, is_sv):
    w, h = spr.get_size()
    x, y = cx - w // 2, cy - h // 2
    if is_sv:
        p = 4
        pygame.draw.rect(
            surf, SV_RING, (x - p, y - p, w + p * 2, h + p * 2), 2, border_radius=3
        )
    surf.blit(spr, (x, y))


# The SVM part
def fit_svm(pts, lbls, C, kernel):
    if len(set(lbls)) < 2:
        return None, False
    try:
        if kernel == "linear":
            clf = LinearSVC(C=C, max_iter=5000, dual=True)
        else:
            clf = SVC(kernel=kernel, C=C, degree=3, gamma="scale")
        clf.fit(pts, lbls)
        
        return clf, kernel == "linear"
    except Exception:
        return None, False


def support_vectors(clf, is_linear, pts, lbls):
    if clf is None:
        return set()
    if not is_linear:
        return set(clf.support_)
    ydf = np.array(lbls, float) * clf.decision_function(pts)
    return set(np.where(ydf <= 1 + 1e-4)[0].tolist())


def compute_df(clf, show_regions):
    xs = np.arange(0, CANVAS_W, MESH)
    ys = np.arange(0, H, MESH)
    xx, yy = np.meshgrid(xs, ys)
    grid = np.c_[xx.ravel(), yy.ravel()]
    DF = clf.decision_function(grid).reshape(xx.shape)

    region_surf = None
    if show_regions:
        Z = clf.predict(grid).reshape(xx.shape)
        region_surf = pygame.Surface((CANVAS_W, H), pygame.SRCALPHA)
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                col = CAT_REGION if Z[j, i] == 1 else DOG_REGION
                pygame.draw.rect(region_surf, col, (int(x), int(y), MESH, MESH))

    return region_surf, xs, ys, DF


def contour(surf, xs, ys, DF, level, color, width=2):
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            vals = [DF[j, i], DF[j, i + 1], DF[j + 1, i + 1], DF[j + 1, i]]
            signs = [v >= level for v in vals]
            if all(signs) or not any(signs):
                continue
            corners = [
                (xs[i], ys[j]),
                (xs[i + 1], ys[j]),
                (xs[i + 1], ys[j + 1]),
                (xs[i], ys[j + 1]),
            ]
            pts = []
            for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
                if signs[a] != signs[b]:
                    t = (level - vals[a]) / (vals[b] - vals[a] + 1e-12)
                    pts.append(
                        (
                            int(corners[a][0] + t * (corners[b][0] - corners[a][0])),
                            int(corners[a][1] + t * (corners[b][1] - corners[a][1])),
                        )
                    )
            if len(pts) >= 2:
                pygame.draw.line(surf, color, pts[0], pts[1], width)


# Drawing the gui...
def draw_panel(surf, fb, fs, kernel, C, n_sv, regions, cat_spr, dog_spr):
    x0 = CANVAS_W
    panel = pygame.Surface((PANEL_W, H), pygame.SRCALPHA)
    panel.fill(PANEL_BG)
    surf.blit(panel, (x0, 0))

    def t(text, y, f=None, c=FG):
        surf.blit((f or fs).render(text, True, c), (x0 + 10, y))

    def div(y):
        pygame.draw.line(surf, ACCENT, (x0 + 8, y), (x0 + PANEL_W - 8, y), 1)

    t("SVM Visualizer", 14, fb, ACCENT)
    div(46)
    t("Kernel:", 56)
    for i, (k, key) in enumerate(
        [("linear", "L"), ("rbf", "R"), ("poly", "P")]
    ):
        t(
            f"[{key}]  {k}",
            74 + i * 26,
            c=(255, 220, 80) if k == kernel else (160, 155, 185),
        )

    div(182)
    t("Regularisation C:", 190)
    t(f"  {C:.3f}   [+] / [-]", 208)
    bx, by, bw, bh = x0 + 10, 232, PANEL_W - 20, 8
    pygame.draw.rect(surf, (80, 75, 100), (bx, by, bw, bh), border_radius=4)
    fill = int(bw * max(0, min(1, (math.log10(max(C, 0.01)) + 2) / 4)))
    if fill:
        pygame.draw.rect(surf, ACCENT, (bx, by, fill, bh), border_radius=4)

    div(252)
    t(f"Support vectors: {n_sv}", 262, c=SV_RING if n_sv else (100, 95, 120))
    t(
        f"[D]  Regions: {'ON' if regions else 'OFF'}",
        288,
        c=(100, 230, 120) if regions else (160, 155, 185),
    )

    div(312)
    t("Controls:", 320, c=ACCENT)
    for i, line in enumerate(
        [
            "Left-click \u2192 Cat",
            "Right-click \u2192 Dog",
            "[C]  Clear all",
            "[H]  Help",
            "[Q]  Quit",
        ]
    ):
        t(line, 338 + i * 22, c=(180, 175, 200))


def draw_legend(surf, fs):
    x = 10
    for col, label in [
        (BOUNDARY, "Decision boundary"),
        (M_POS, "Margin (+1)"),
        (M_NEG, "Margin (\u22121)"),
        (SV_RING, "Support vector"),
    ]:
        pygame.draw.rect(surf, col, (x, H - 28, 12, 12), border_radius=2)
        s = fs.render(label, True, (60, 55, 80))
        surf.blit(s, (x + 16, H - 29))
        x += s.get_width() + 34



def main():
    pygame.init()
    surf = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SVM Visualizer \u2014 Cats vs Dogs")
    fb, fs = pygame.font.SysFont("segoeui", 20, bold=True), pygame.font.SysFont(
        "segoeui", 15
    )
    cat_spr, dog_spr = bake(CAT_SPRITE, CAT_PAL), bake(DOG_SPRITE, DOG_PAL)
    clock = pygame.time.Clock()

    points, labels = [], []
    kernel, C_val = "linear", 1.0
    show_regions = True
    clf, is_linear, region_surf, xs, ys, DF = None, False, None, None, None, None
    dirty = False

    KEYS = {
        pygame.K_l: "linear",
        pygame.K_r: "rbf",
        pygame.K_p: "poly",
    }
    
    # The game loop
    while True:
        clock.tick(60)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                elif k == pygame.K_c:
                    points.clear()
                    labels.clear()
                    dirty = True
                elif k == pygame.K_d:
                    show_regions = not show_regions
                    dirty = True
                elif k in KEYS:
                    kernel = KEYS[k]
                    dirty = True
                elif k in (pygame.K_EQUALS, pygame.K_PLUS):
                    C_val = min(C_val * 1.5, 100.0)
                    dirty = True
                elif k == pygame.K_MINUS:
                    C_val = max(C_val / 1.5, 0.01)
                    dirty = True

            if (
                ev.type == pygame.MOUSEBUTTONDOWN
                and ev.pos[0] < CANVAS_W
            ):
                points.append(list(ev.pos))
                labels.append(1 if ev.button == 1 else -1)
                dirty = True

        if dirty:
            dirty = False
            clf, is_linear = (
                fit_svm(np.array(points), np.array(labels), C_val, kernel)
                if len(set(labels)) >= 2
                else (None, False)
            )
            region_surf, xs, ys, DF = (
                compute_df(clf, show_regions) if clf else (None, None, None, None)
            )

        sv_set = support_vectors(clf, is_linear, points, labels) if clf else set()

        # Draw
        surf.fill(BG)
        for gx in range(0, CANVAS_W, 20):
            pygame.draw.line(surf, GRID_C, (gx, 0), (gx, H))
        for gy in range(0, H, 20):
            pygame.draw.line(surf, GRID_C, (0, gy), (CANVAS_W, gy))

        if region_surf:
            surf.blit(region_surf, (0, 0))
        if DF is not None:
            contour(surf, xs, ys, DF, 0.0, BOUNDARY, 2)
            contour(surf, xs, ys, DF, 1.0, M_POS, 1)
            contour(surf, xs, ys, DF, -1.0, M_NEG, 1)

        for idx, (p, lbl) in enumerate(zip(points, labels)):
            draw_sprite(
                surf, cat_spr if lbl == 1 else dog_spr, p[0], p[1], idx in sv_set
            )

        draw_legend(surf, fs)
        draw_panel(
            surf,
            fb,
            fs,
            kernel,
            C_val,
            len(sv_set),
            show_regions,
            cat_spr,
            dog_spr,
        )
        pygame.display.flip()


if __name__ == "__main__":
    main()
