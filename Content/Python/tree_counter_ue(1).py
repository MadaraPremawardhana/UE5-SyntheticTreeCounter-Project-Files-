"""
tree_counter_ue.py

Drop this file into  <YourProject>/Content/Python/tree_counter_ue.py
Unreal's Python Editor Script Plugin auto-adds that folder to sys.path, so
it's importable from an "Execute Python Command (Advanced)" Blueprint node
with no extra setup.

This is the same heuristic tree-counting algorithm as tree_counter.py
(saturation threshold -> local brightness peaks -> watershed split), just
with the CLI/argparse bits swapped for a single entry point, run_from_path(),
meant to be called from a one-line Blueprint-built Python statement:

    import importlib, tree_counter_ue
    importlib.reload(tree_counter_ue)
    tree_counter_ue.run_from_path(r"<CapturedPhotoPath>")

`importlib.reload` matters while you're iterating: Unreal's embedded Python
keeps modules cached for the whole editor session, so without it, edits to
this file won't take effect until you restart the editor or explicitly
reload like this.

run_from_path() prints EXACTLY ONE line on stdout: the tree count as a
bare integer (e.g. "1377"), nothing else -- no prefix, no label. On
failure it still prints exactly one line, "-1", so the calling Blueprint
always gets a single parseable integer out of the node's Log Output
array with a plain String-to-Int conversion -- no Contains/Split String
needed. Error details (message + traceback) go to Unreal's Output Log via
unreal.log_error() instead of stdout, so they're visible for debugging
without showing up in that Log Output array.
"""

import json
import os
import traceback
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

try:
    import unreal

    def _log_error(msg):
        unreal.log_error(msg)
except ImportError:  # running outside Unreal (e.g. local testing) -- fall back to stderr
    import sys

    def _log_error(msg):
        print(msg, file=sys.stderr)


# ---- tunables, same as tree_counter.py -------------------------------------
SAT_THRESH = 60
MIN_TREE_PX = 14
MIN_BLOB_AREA = 40
MAX_BLOB_AREA_FRAC = 0.02
MORPH_OPEN_PX = 3
# -----------------------------------------------------------------------------


@dataclass
class TreeBox:
    id: int
    x: int
    y: int
    w: int
    h: int
    cx: int
    cy: int


def build_canopy_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, mask = cv2.threshold(sat, SAT_THRESH, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN_PX, MORPH_OPEN_PX))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    return mask


def count_trees(bgr, mask=None):
    if mask is None:
        mask = build_canopy_mask(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.0)

    coords = peak_local_max(
        smooth,
        min_distance=MIN_TREE_PX,
        labels=(mask > 0).astype(np.uint8),
        exclude_border=False,
    )

    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i

    labels = watershed(-smooth, markers=markers, mask=(mask > 0))

    canopy_area = float((mask > 0).sum())
    boxes = []
    for region_id in range(1, labels.max() + 1):
        ys, xs = np.where(labels == region_id)
        if ys.size == 0:
            continue
        area = ys.size
        if area < MIN_BLOB_AREA:
            continue
        if canopy_area > 0 and area / canopy_area > MAX_BLOB_AREA_FRAC:
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        boxes.append(
            TreeBox(
                id=len(boxes) + 1,
                x=x0, y=y0,
                w=x1 - x0 + 1, h=y1 - y0 + 1,
                cx=int(xs.mean()), cy=int(ys.mean()),
            )
        )
    return boxes


def draw_labels(bgr, boxes):
    out = bgr.copy()
    for b in boxes:
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), (0, 255, 0), 1)
        cv2.circle(out, (b.cx, b.cy), 2, (0, 0, 255), -1)
    label = f"Tree count: {len(boxes)}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(out, (10, 10), (20 + tw, 30 + th), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 20 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return out


_last_seen_size = {}  # path -> size, persisted across calls for check_screenshot_ready()


def check_screenshot_ready(path):
    """Instant, NON-blocking readiness check -- does not sleep or loop, so
    it's safe to call from Execute Python Command without freezing the
    engine. Meant to be called repeatedly from a Blueprint-side poll loop
    that waits via a Delay node between calls (a Delay yields back to the
    engine each iteration, unlike time.sleep() inside Python, which is why
    the polling has to happen on the Blueprint side, not in here).

    Returns True once the file exists and its size matches what it was on
    the *previous* call to this function for the same path (i.e. the write
    has stopped growing since you last checked)."""
    path = os.path.normpath(path)
    if not os.path.exists(path):
        _last_seen_size.pop(path, None)
        return False
    size = os.path.getsize(path)
    prev = _last_seen_size.get(path)
    _last_seen_size[path] = size
    return size > 0 and size == prev


def run_from_path(input_path, output_path=None, json_path=None):
    """Entry point for the Blueprint one-liner. Prints exactly one line on
    stdout -- the bare tree count integer on success, or "-1" on failure
    -- so the calling Blueprint can read it straight out of Log Output
    with a plain String-to-Int conversion. Failure details go to
    unreal.log_error() (Output Log), not stdout.

    Call this only after your Blueprint poll loop has confirmed
    check_screenshot_ready() returned True -- this function itself does
    a single instant existence check and fails fast (no waiting/looping)
    so it can never stall the engine."""
    try:
        input_path = os.path.normpath(input_path)
        if output_path is None:
            root, ext = os.path.splitext(input_path)
            output_path = f"{root}_labeled.png"

        if not os.path.exists(input_path) or os.path.getsize(input_path) == 0:
            raise FileNotFoundError(
                f"image not ready yet (call check_screenshot_ready() from a "
                f"Blueprint Delay-loop before calling run_from_path): {input_path}"
            )

        bgr = cv2.imread(input_path)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {input_path}")

        boxes = count_trees(bgr)
        labeled = draw_labels(bgr, boxes)
        cv2.imwrite(output_path, labeled)

        if json_path:
            with open(json_path, "w") as f:
                json.dump({"count": len(boxes), "boxes": [asdict(b) for b in boxes]}, f, indent=2)

        print(len(boxes))
        return len(boxes)

    except Exception as e:  # noqa: BLE001 -- want any failure surfaced, not swallowed
        _log_error(f"tree_counter_ue.run_from_path failed: {e}\n{traceback.format_exc()}")
        print(-1)
        return -1
