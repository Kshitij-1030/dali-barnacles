"""Helper functions for the barnacle-counting notebook (barnacles.ipynb).

Ground truth from the masks, the scorer and chance baseline, and the detectors' building blocks
(patch classifier, probability map, peak finding, frame detection).
"""
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def getBarnacleContours(maskPath, cutoff=30):
    mask = cv2.imread(maskPath)
    if mask is None:
        raise FileNotFoundError(maskPath)
    
    isLine = mask[:, :, 2] < 200
    lineImg = isLine.astype(np.uint8) * 255

    kernel = np.ones((3, 3), np.uint8)
    closedImg = cv2.morphologyEx(lineImg, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleanContours = [c for c in contours if cv2.contourArea(c) > cutoff]
    return cleanContours


def getCenters(contours):
    centers = []
    for c in contours:
        m = cv2.moments(c)
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        centers.append((cx, cy))
    return np.array(centers)


def evaluate(pred, true, maxDist=10):
    pred = np.asarray(pred, dtype=float).reshape(-1, 2)
    true = np.asarray(true, dtype=float).reshape(-1, 2)
    if len(pred) == 0:
        return {"predicted": 0, "true": len(true), "count_error": -len(true),
                "precision": 0.0, "recall": 0.0}

    # distance between every predicted point and every true point
    d = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=2)

    # candidate pairs within maxDist, closest first
    pairs = np.argwhere(d <= maxDist)
    pairs = pairs[np.argsort(d[pairs[:, 0], pairs[:, 1]])]

    usedPred, usedTrue, matches = set(), set(), 0
    for p, t in pairs:
        if p not in usedPred and t not in usedTrue:   # each point can match only once
            usedPred.add(p)
            usedTrue.add(t)
            matches += 1

    return {"predicted": len(pred), "true": len(true),
            "count_error": len(pred) - len(true),
            "precision": matches / len(pred),
            "recall": matches / len(true)}


def f1(s):
    return 2 * s["precision"] * s["recall"] / (s["precision"] + s["recall"] + 1e-9)


def randomBaseline(trueCenters, box, nPoints, maxDist, repeats=30):
    """Score nPoints uniformly random points placed inside box = (xmin, xmax, ymin, ymax)."""
    xmin, xmax, ymin, ymax = box
    scores = []
    for seed in range(repeats):
        r = np.random.default_rng(1000 + seed)
        pts = np.column_stack([r.uniform(xmin, xmax, nPoints), r.uniform(ymin, ymax, nPoints)])
        s = evaluate(pts, trueCenters, maxDist)
        scores.append((s["precision"], s["recall"], f1(s)))
    p, rec, f = np.mean(scores, axis=0)
    return {"precision": p, "recall": rec, "F1": f}


def describe(s, trueCenters, box, maxDist):
    """one-line summary: count error, precision, recall, F1, and the chance F1 for the same number of detections"""
    chance = randomBaseline(trueCenters, box, max(s["predicted"], 1), maxDist)["F1"]
    return (f"predicted={s['predicted']:4d} (count {100 * s['count_error'] / s['true']:+5.1f}%)  "
            f"P={s['precision']:.2f} R={s['recall']:.2f} F1={f1(s):.2f}  chance F1={chance:.2f}")


def nearestOffsets(pred, true, maxSearch=25):
    d = np.linalg.norm(pred[:, None, :] - true[None, :, :], axis=2)
    nearest = d.argmin(axis=1)                  # closest true barnacle for each detection
    offsets = true[nearest] - pred              # (dx, dy) needed to reach it
    close = d.min(axis=1) < maxSearch           # ignore detections with nothing nearby
    return offsets[close]


def findPeaks(response, thresh, minDist=10, offset=0):
    kernel = np.ones((2 * minDist + 1, 2 * minDist + 1), np.uint8)
    localMax = (response == cv2.dilate(response, kernel)) & (response > thresh)
    ys, xs = np.nonzero(localMax)
    return np.column_stack([xs + offset, ys + offset])


def getPatch(img, x, y, half=20):
    x, y = int(round(x)), int(round(y))
    p = img[y - half : y + half + 1, x - half : x + half + 1]
    if p.shape != (2 * half + 1, 2 * half + 1):
        return None
    p = cv2.resize(p, (21, 21), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return (p - p.mean()) / (p.std() + 1e-6)      # per-patch normalization: ignore overall brightness


def makeDataset(gray, trueCenters, allCenters, box, half, nNeg, rng):
    """positives: patches centered on labeled barnacles; negatives: random spots in box away from every center"""
    xmin, xmax, ymin, ymax = box
    X, labels = [], []
    for cx, cy in trueCenters:
        p = getPatch(gray, cx, cy, half)
        if p is not None:
            X.append(p); labels.append(1)
    nPos = len(labels)
    minAway = 12 * half / 20                      # "not near a center" distance scales with barnacle size
    while len(labels) < nPos + nNeg:
        px, py = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
        if np.linalg.norm(allCenters - [px, py], axis=1).min() < minAway:
            continue
        p = getPatch(gray, px, py, half)
        if p is not None:
            X.append(p); labels.append(0)
    return np.array(X), np.array(labels)


def trainForest(parts, seed=0):
    """parts: list of (gray, trueCenters, allCenters, box, half, nNeg), one per training photo"""
    rng = np.random.default_rng(seed)
    Xs, ys = [], []
    for gray, trueC, allC, box, half, nNeg in parts:
        X, y = makeDataset(gray, trueC, allC, box, half, nNeg, rng)
        Xs.append(X); ys.append(y)
    return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed).fit(
        np.vstack(Xs), np.concatenate(ys))


def probabilityMap(clf, gray, xmin, xmax, ymin, ymax, half, stride):
    xsGrid = np.arange(int(xmin), int(xmax), stride)
    ysGrid = np.arange(int(ymin), int(ymax), stride)
    coords, feats = [], []
    for iy, gy in enumerate(ysGrid):
        for ix, gx in enumerate(xsGrid):
            p = getPatch(gray, gx, gy, half)
            if p is not None:
                coords.append((iy, ix))
                feats.append(p)
    probs = clf.predict_proba(np.array(feats))[:, 1]       # chance each spot is a barnacle center
    probMap = np.zeros((len(ysGrid), len(xsGrid)), np.float32)
    for (iy, ix), pr in zip(coords, probs):
        probMap[iy, ix] = pr
    return cv2.GaussianBlur(probMap, (0, 0), 1.5), xsGrid, ysGrid    # smooth out jitter


def detect(probMap, xsGrid, ysGrid, thresh, minDist=4):
    peaks = findPeaks(probMap, thresh, minDist=minDist)    # coordinates here are grid cells, not pixels
    return np.column_stack([xsGrid[peaks[:, 0]], ysGrid[peaks[:, 1]]])


def runs(mask1d, minLen=1):
    """start/end index of every run of True in a 1-D boolean array"""
    out, start = [], None
    for i, v in enumerate(mask1d):
        if v and start is None:
            start = i
        if not v and start is not None:
            out.append((start, i - 1)); start = None
    if start is not None:
        out.append((start, len(mask1d) - 1))
    return [r for r in out if r[1] - r[0] + 1 >= minLen]


def findInnerSquare(img, shrink=0.25, margin=50, frac=0.5):
    """Box (xmin, xmax, ymin, ymax) in full-size pixels of the empty square inside the wire frame, inset by margin."""
    small = cv2.resize(img, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
    b, g, r = [c.astype(int) for c in cv2.split(small)]
    teal = ((b - r) > 40) & ((g - r) > 25)                      # wire = teal
    H, W = teal.shape
    # vertical bars: columns that are mostly teal across the middle band of the photo
    vbars = runs(teal[int(0.2 * H):int(0.65 * H), :].mean(axis=0) > frac, minLen=3)
    left = max([v for v in vbars if v[1] < W / 2], key=lambda v: v[1])     # bar just left of centre
    right = min([v for v in vbars if v[0] > W / 2], key=lambda v: v[0])    # bar just right of centre
    # horizontal bars: rows that are mostly teal between those two bars
    strip = teal[:, left[1] + 3 : right[0] - 2]
    hbars = [h for h in runs(strip.mean(axis=1) > frac, minLen=2) if h[0] > 0.03 * H]
    top, bottom = hbars[0], hbars[1]                                       # first two bars below the top edge
    return ((left[1] + 1) / shrink + margin, (right[0] - 1) / shrink - margin,
            (top[1] + 1) / shrink + margin, (bottom[0] - 1) / shrink - margin)


def detectOnPhoto(clf, gray, half, stride, thresh, box=None):
    """Run the detector over the whole photo (or over box). The photo is reflected at its borders,
    so barnacles within `half` px of an edge are not skipped. Peaks closer than about 0.6 * half px
    count as one barnacle, so the suppression distance grows with barnacle size."""
    h, w = gray.shape
    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REFLECT_101)
    xmin, xmax, ymin, ymax = box if box is not None else (0, w, 0, h)
    pm, xg, yg = probabilityMap(clf, padded, xmin + half, xmax + half, ymin + half, ymax + half, half, stride)
    return detect(pm, xg, yg, thresh, minDist=max(2, round(0.6 * half / stride))) - half
