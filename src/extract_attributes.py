"""
Image-derived structured dermoscopic attributes.

Attribute provenance: the structured attribute vector is computed algorithmically from
each image using classical dermoscopy-inspired image analysis (lesion segmentation
followed by shape, colour, and texture descriptors). These are reproducible image-derived
proxies, not expert annotations. They serve three roles in the architecture:
  (1) the "reference" attribute vector supplied to the fusion module (oracle condition),
  (2) regression targets for the attribute-reconstruction head,
  (3) the quantity the predicted-attribute head learns to estimate from the image.

Primary attribute set (paper's Step 3 primary list), all normalized to [0,1]:
  asymmetry, border_irregularity, color_variegation, num_colors, pigment_network,
  dots_globules, streaks, blue_white_veil, regression_structures, vascular_patterns,
  ulceration_crusting.
"""
import os, csv, json
import numpy as np
import cv2
from sklearn.cluster import KMeans

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMG_DIR = os.path.join(ROOT, "images")
OUT = os.path.join(HERE, "attributes")
os.makedirs(OUT, exist_ok=True)

ATTRS = ["asymmetry", "border_irregularity", "color_variegation", "num_colors",
         "pigment_network", "dots_globules", "streaks", "blue_white_veil",
         "regression_structures", "vascular_patterns", "ulceration_crusting"]

def segment_lesion(bgr):
    """Otsu on inverted grayscale (lesion usually darker); largest component; fallback ellipse."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (7, 7), 0)
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, 8)
    H, W = g.shape
    mask = np.zeros((H, W), np.uint8)
    if n > 1:
        idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[idx, cv2.CC_STAT_AREA]
        if area > 0.02 * H * W:
            mask = (lab == idx).astype(np.uint8) * 255
    if mask.sum() == 0:  # fallback: central ellipse
        cv2.ellipse(mask, (W // 2, H // 2), (W // 3, H // 3), 0, 0, 360, 255, -1)
    return mask

def _sig(x, k=1.0):
    return float(1.0 / (1.0 + np.exp(-k * x)))

def asymmetry(mask):
    m = (mask > 0).astype(np.uint8)
    M = cv2.moments(m, binaryImage=True)
    if M["m00"] == 0:
        return 0.0
    # principal-axis alignment via moments
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    mu20, mu02, mu11 = M["mu20"] / M["m00"], M["mu02"] / M["m00"], M["mu11"] / M["m00"]
    theta = 0.5 * np.arctan2(2 * mu11, (mu20 - mu02))
    H, W = m.shape
    Rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(theta), 1.0)
    r = cv2.warpAffine(m, Rot, (W, H))
    area = r.sum() + 1e-6
    # vertical flip diff & horizontal flip diff over principal axes
    dv = np.logical_xor(r, np.flipud(r)).sum() / (2 * area)
    dh = np.logical_xor(r, np.fliplr(r)).sum() / (2 * area)
    return float(np.clip(0.5 * (dv + dh), 0, 1))

def border_irregularity(mask):
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    per = cv2.arcLength(c, True)
    if area <= 0:
        return 0.0
    compactness = (per * per) / (4 * np.pi * area)  # 1 for a circle, >1 irregular
    return float(np.clip((compactness - 1.0) / 3.0, 0, 1))

def _lab_stats(bgr, mask):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[mask > 0]
    return lab, px

def color_variegation(bgr, mask):
    _, px = _lab_stats(bgr, mask)
    if len(px) < 10:
        return 0.0
    # variance of chroma channels a,b normalized
    v = np.sqrt(px[:, 1].var() + px[:, 2].var())
    return float(np.clip(v / 40.0, 0, 1))

def num_colors(bgr, mask):
    """Count distinct clinical dominant colors present (>3% of lesion) among a 6-color palette."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    px = rgb[mask > 0]
    if len(px) < 20:
        return 0.0
    # clinical dermoscopy colors (RGB): white, red, light-brown, dark-brown, blue-gray, black
    palette = np.array([[255, 255, 255], [200, 60, 60], [190, 140, 90],
                        [110, 70, 40], [90, 110, 140], [30, 30, 30]], np.float32)
    d = np.linalg.norm(px[:, None, :] - palette[None, :, :], axis=2)
    assign = d.argmin(1)
    frac = np.bincount(assign, minlength=6) / len(px)
    k = int((frac > 0.03).sum())
    return float(np.clip(k / 6.0, 0, 1))

def pigment_network(bgr, mask):
    """Regular reticular texture proxy: mid-frequency edge energy inside lesion."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    e = np.abs(lap)[mask > 0]
    if len(e) < 10:
        return 0.0
    return float(np.clip(e.mean() / 25.0, 0, 1))

def dots_globules(bgr, mask):
    """Blob density via LoG-style detector inside lesion."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bitwise_and(g, g, mask=mask)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True; params.minArea = 8; params.maxArea = 400
    params.filterByCircularity = True; params.minCircularity = 0.6
    params.filterByConvexity = False; params.filterByInertia = False
    params.filterByColor = False
    det = cv2.SimpleBlobDetector_create(params)
    kp = det.detect(g)
    area = (mask > 0).sum() + 1e-6
    dens = len(kp) / (area / 1000.0)
    return float(np.clip(dens / 5.0, 0, 1))

def streaks(bgr, mask):
    """Radial/elongated structures at periphery: edge density in a peripheral ring."""
    m = (mask > 0).astype(np.uint8)
    er = cv2.erode(m, np.ones((15, 15), np.uint8))
    ring = ((m - er) > 0)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150) > 0
    if ring.sum() < 10:
        return 0.0
    return float(np.clip((edges & ring).sum() / ring.sum() * 3.0, 0, 1))

def blue_white_veil(bgr, mask):
    """Fraction of bluish-white pixels: bluish hue + moderate/high lightness, low-mid saturation."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    bw = (h > 100) & (h < 140) & (s > 20) & (s < 160) & (v > 90)
    px = (mask > 0)
    if px.sum() < 10:
        return 0.0
    return float(np.clip((bw & px).sum() / px.sum() * 4.0, 0, 1))

def regression_structures(bgr, mask):
    """Scar-like white/depigmented areas: high L, low chroma in Lab."""
    lab, _ = _lab_stats(bgr, mask)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.sqrt((a - 128) ** 2 + (b - 128) ** 2)
    reg = (L > 180) & (chroma < 12)
    px = (mask > 0)
    if px.sum() < 10:
        return 0.0
    return float(np.clip((reg & px).sum() / px.sum() * 4.0, 0, 1))

def vascular_patterns(bgr, mask):
    """Reddish vascular proxy: high a* (red) in Lab."""
    lab, _ = _lab_stats(bgr, mask)
    a = lab[..., 1]
    red = a > 150
    px = (mask > 0)
    if px.sum() < 10:
        return 0.0
    return float(np.clip((red & px).sum() / px.sum() * 3.0, 0, 1))

def ulceration_crusting(bgr, mask):
    """Crust/ulcer proxy: yellow-red high-intensity keratin/blood regions."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    crust = ((h < 25) | (h > 165)) & (s > 60) & (v > 120)
    px = (mask > 0)
    if px.sum() < 10:
        return 0.0
    return float(np.clip((crust & px).sum() / px.sum() * 3.0, 0, 1))

def compute_all(bgr):
    mask = segment_lesion(bgr)
    return {
        "asymmetry": asymmetry(mask),
        "border_irregularity": border_irregularity(mask),
        "color_variegation": color_variegation(bgr, mask),
        "num_colors": num_colors(bgr, mask),
        "pigment_network": pigment_network(bgr, mask),
        "dots_globules": dots_globules(bgr, mask),
        "streaks": streaks(bgr, mask),
        "blue_white_veil": blue_white_veil(bgr, mask),
        "regression_structures": regression_structures(bgr, mask),
        "vascular_patterns": vascular_patterns(bgr, mask),
        "ulceration_crusting": ulceration_crusting(bgr, mask),
    }

def main():
    imgs = sorted(f for f in os.listdir(IMG_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg")))
    rows = []
    for i, im in enumerate(imgs):
        bgr = cv2.imread(os.path.join(IMG_DIR, im))
        if bgr is None:
            continue
        bgr = cv2.resize(bgr, (512, 512))
        a = compute_all(bgr)
        a["image"] = im
        rows.append(a)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(imgs)} images processed")
    out = os.path.join(OUT, "attributes.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image"] + ATTRS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in ["image"] + ATTRS})
    # normalization / distribution report
    arr = np.array([[r[a] for a in ATTRS] for r in rows], np.float32)
    report = {a: {"mean": float(arr[:, j].mean()), "std": float(arr[:, j].std()),
                  "min": float(arr[:, j].min()), "max": float(arr[:, j].max())}
              for j, a in enumerate(ATTRS)}
    with open(os.path.join(OUT, "attribute_report.json"), "w") as f:
        json.dump({"n": len(rows), "attrs": ATTRS, "stats": report,
                   "provenance": "algorithmic image-derived proxies (NOT expert annotations)"},
                  f, indent=2)
    print(f"Wrote {out} for {len(rows)} images")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
