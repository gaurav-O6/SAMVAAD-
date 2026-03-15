"""
================================================================================
  Project SAMVAAD – Braille Recognition
  samvaad_braille.py  (ONE FILE – engine + runner combined)

  HOW TO RUN  (from D:\\SAMVAAD in VS Code terminal):
  ──────────────────────────────────────────────────
  Interactive (type paths one by one):
      python samvaad_braille.py

  Single image:
      python samvaad_braille.py D:\\SAMVAAD\\dataset\\G.png

  Test entire dataset folder (A-Z, 0-9, test images):
      python samvaad_braille.py --test D:\\SAMVAAD\\dataset

  With debug overlays saved next to each image:
      python samvaad_braille.py --test D:\\SAMVAAD\\dataset --debug

  DEPENDENCIES:
      pip install opencv-python numpy
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  WORD SPLITTER (for Braille images without spaces between words)
# ══════════════════════════════════════════════════════════════════════════════

_COMMON_WORDS = {
    'a', 'about', 'all', 'also', 'and', 'as', 'at', 'be', 'because', 'but', 'by',
    'can', 'come', 'could', 'day', 'do', 'even', 'find', 'first', 'for', 'from',
    'get', 'give', 'go', 'have', 'he', 'her', 'here', 'him', 'his', 'how', 'i',
    'if', 'in', 'into', 'it', 'its', 'just', 'know', 'like', 'look', 'make',
    'man', 'many', 'me', 'more', 'my', 'new', 'no', 'not', 'now', 'of', 'on',
    'one', 'only', 'or', 'other', 'our', 'out', 'people', 'say', 'see', 'she',
    'so', 'some', 'take', 'tell', 'than', 'that', 'the', 'their', 'them', 'then',
    'there', 'these', 'they', 'thing', 'think', 'this', 'time', 'to', 'two', 'up',
    'use', 'very', 'want', 'way', 'we', 'well', 'what', 'when', 'which', 'who',
    'will', 'with', 'would', 'year', 'you', 'your',
    'hello', 'world', 'test', 'braille', 'detection', 'system', 'image', 'text',
    'is', 'are', 'was', 'were', 'been', 'being', 'am', 'has', 'had', 'does', 'did',
    'message', 'welcome', 'thanks', 'please', 'help', 'yes', 'no',
}

def _split_concatenated_words(text: str) -> str:
    """
    Split text without spaces into words using dynamic programming.
    Example: 'hellothisis' -> 'hello this is'
    """
    if not text or len(text) < 3:
        return text
    
    text_lower = text.lower()
    n = len(text_lower)
    
    # dp[i] = (cost, prev_position)
    dp = [(float('inf'), -1)] * (n + 1)
    dp[0] = (0, -1)
    
    for i in range(1, n + 1):
        for j in range(max(0, i - 15), i):  # Max word length 15
            word = text_lower[j:i]
            if word in _COMMON_WORDS:
                cost = dp[j][0]
            else:
                cost = dp[j][0] + len(word)
            
            if cost < dp[i][0]:
                dp[i] = (cost, j)
    
    # Backtrack
    words = []
    pos = n
    while pos > 0:
        prev = dp[pos][1]
        if prev == -1:
            break
        words.append(text[prev:pos])  # Preserve original case
        pos = prev
    
    words.reverse()
    return ' '.join(words) if words else text


def auto_space_text(raw_text: str, remove_newlines: bool = True) -> str:
    """
    Post-process Braille decoder output to add spaces and clean formatting.
    
    Parameters
    ----------
    raw_text : str
        Raw output from BrailleRecognizer.recognize()
    remove_newlines : bool
        If True, converts multi-line Braille into single line (default: True)
        
    Returns
    -------
    str
        Clean text with proper word spacing
        
    Examples
    --------
    >>> auto_space_text('hellothisis\ntestforbraille')
    'hello this is test for braille'
    
    >>> auto_space_text('hello world')  # already has spaces
    'hello world'
    """
    if not raw_text:
        return ""
    
    # Process each line separately FIRST (before joining)
    lines = raw_text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Split words if line has no spaces
        if ' ' not in line and len(line) > 2:
            line = _split_concatenated_words(line)
        
        processed_lines.append(line)
    
    # Then join lines based on remove_newlines flag
    if remove_newlines:
        return ' '.join(processed_lines)
    else:
        return '\n'.join(processed_lines)


# ══════════════════════════════════════════════════════════════════════════════
#  BRAILLE LOOKUP TABLE  (Grade-1 English)
#
#  Dot layout inside one Braille cell:
#    Left col   Right col
#      1           4
#      2           5
#      3           6
#
#  Key  = tuple (dot1, dot2, dot3, dot4, dot5, dot6)   1=raised, 0=absent
# ══════════════════════════════════════════════════════════════════════════════

BRAILLE_MAP: Dict[Tuple[int, ...], str] = {
    (1,0,0,0,0,0): 'a',  (1,1,0,0,0,0): 'b',  (1,0,0,1,0,0): 'c',
    (1,0,0,1,1,0): 'd',  (1,0,0,0,1,0): 'e',  (1,1,0,1,0,0): 'f',
    (1,1,0,1,1,0): 'g',  (1,1,0,0,1,0): 'h',  (0,1,0,1,0,0): 'i',
    (0,1,0,1,1,0): 'j',  (1,0,1,0,0,0): 'k',  (1,1,1,0,0,0): 'l',
    (1,0,1,1,0,0): 'm',  (1,0,1,1,1,0): 'n',  (1,0,1,0,1,0): 'o',
    (1,1,1,1,0,0): 'p',  (1,1,1,1,1,0): 'q',  (1,1,1,0,1,0): 'r',
    (0,1,1,1,0,0): 's',  (0,1,1,1,1,0): 't',  (1,0,1,0,0,1): 'u',
    (1,1,1,0,0,1): 'v',  (0,1,0,1,1,1): 'w',  (1,0,1,1,0,1): 'x',
    (1,0,1,1,1,1): 'y',  (1,0,1,0,1,1): 'z',
    # Punctuation
    (0,0,0,0,0,0): ' ',  (0,1,0,0,0,0): ',',  (0,1,0,0,1,0): ';',
    (0,1,1,0,0,0): ':',  (0,1,1,0,1,0): '!',  (0,0,1,0,1,0): '?',
    (0,1,1,0,0,1): '-',  (0,1,0,0,1,1): '.',  (0,1,1,0,1,1): '"',
    (0,0,1,0,0,1): "'",
}

# Digits (used after the number indicator)
DIGIT_MAP: Dict[Tuple[int, ...], str] = {
    (1,0,0,0,0,0): '1',  (1,1,0,0,0,0): '2',  (1,0,0,1,0,0): '3',
    (1,0,0,1,1,0): '4',  (1,0,0,0,1,0): '5',  (1,1,0,1,0,0): '6',
    (1,1,0,1,1,0): '7',  (1,1,0,0,1,0): '8',  (0,1,0,1,0,0): '9',
    (0,1,0,1,1,0): '0',
}

NUMBER_INDICATOR  = (0,0,1,1,1,1)  # dots 3,4,5,6
CAPITAL_INDICATOR = (0,0,0,0,0,1)  # dot 6 only


# ══════════════════════════════════════════════════════════════════════════════
#  DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Dot:
    cx: float; cy: float; radius: float; area: float

@dataclass
class BrailleCell:
    col: int; row: int
    pattern: Tuple[int, ...] = field(default_factory=lambda: (0,)*6)
    bbox: Tuple[float,float,float,float] = (0,0,0,0)


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1  IMAGE LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_image(source: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Accept file path or numpy array → return BGR ndarray."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise ValueError(f"File not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return img
    if isinstance(source, np.ndarray):
        if source.ndim not in (2, 3):
            raise ValueError("Array must be 2-D or 3-D")
        return source.copy()
    raise ValueError(f"Unsupported type: {type(source)}")


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2  PREPROCESSOR
#  Grayscale → CLAHE → Gaussian blur → Adaptive threshold → Morph close
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (gray, binary) where binary has dots=255, background=0."""
    # Handle BGRA
    if bgr.ndim == 3 and bgr.shape[2] == 4:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr.copy()

    # CLAHE — handles uneven lighting, shadows, low contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)

    # Gaussian blur — removes JPEG/screenshot compression noise
    blurred = cv2.GaussianBlur(eq, (5, 5), 0)

    # Adaptive Gaussian threshold — NO fixed brightness value
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31, 10
    )

    # Morphological closing — fills tiny gaps inside dots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Auto-invert if background ended up white (>60% white pixels)
    if np.mean(binary > 127) > 0.6:
        binary = cv2.bitwise_not(binary)

    return gray, binary


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 3+4  DOT DETECTOR + FILTER
#  Contour analysis — no hardcoded pixel sizes
# ══════════════════════════════════════════════════════════════════════════════

def detect_dots(binary: np.ndarray) -> List[Dot]:
    """Detect and validate Braille dots from binary image."""
    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Collect raw blobs
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        blobs.append((area, cw, ch, cnt))

    if not blobs:
        return []

    # Dynamic area range from median (scale-invariant)
    areas = [b[0] for b in blobs]
    med = float(np.median(areas))
    min_a = max(med * 0.10, h * w * 0.000005)
    max_a = min(med * 4.50, h * w * 0.06)

    dots: List[Dot] = []
    for area, cw, ch, cnt in blobs:
        if not (min_a <= area <= max_a):
            continue

        # Circularity filter  (perfect circle = 1.0)
        perim = cv2.arcLength(cnt, True)
        if perim < 1e-3:
            continue
        if (4 * math.pi * area / perim**2) < 0.40:
            continue

        # Aspect ratio filter
        aspect = cw / ch if ch > 0 else 0
        if not (0.35 <= aspect <= 2.65):
            continue

        # Solidity filter
        hull_a = cv2.contourArea(cv2.convexHull(cnt))
        if hull_a < 1 or (area / hull_a) < 0.50:
            continue

        # Centroid
        M = cv2.moments(cnt)
        if M["m00"] < 1e-6:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        dots.append(Dot(cx=cx, cy=cy, radius=math.sqrt(area/math.pi), area=area))

    dots.sort(key=lambda d: (d.cy, d.cx))
    return dots


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 5+6  CELL SEGMENTER
# ══════════════════════════════════════════════════════════════════════════════

def _merge_close(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    sv = sorted(values)
    clusters: List[List[float]] = [[sv[0]]]
    for v in sv[1:]:
        if v - clusters[-1][-1] <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _estimate_spacings(dots: List[Dot]) -> Tuple[float, float]:
    """35th-percentile nearest-neighbour → robust intra-cell spacing estimate."""
    h_nn, v_nn = [], []
    for i, d in enumerate(dots):
        h_nn.append(min(abs(d.cx - dots[j].cx) for j in range(len(dots)) if j != i))
        v_nn.append(min(abs(d.cy - dots[j].cy) for j in range(len(dots)) if j != i))
    h = max(float(np.percentile(h_nn, 35)), float(np.mean([d.radius for d in dots])) * 2)
    v = max(float(np.percentile(v_nn, 35)), h * 0.5)
    return v, h


def _cluster_rows(dots: List[Dot], v_sp: float) -> List[List[Dot]]:
    tol = v_sp * 0.55
    sd = sorted(dots, key=lambda d: d.cy)
    rows: List[List[Dot]] = [[sd[0]]]
    cy = sd[0].cy
    for d in sd[1:]:
        if abs(d.cy - cy) <= tol:
            rows[-1].append(d)
            cy = float(np.mean([x.cy for x in rows[-1]]))
        else:
            rows.append([d])
            cy = d.cy
    return [sorted(r, key=lambda d: d.cx) for r in rows]


def segment_cells(dots: List[Dot]) -> List[BrailleCell]:
    if not dots:
        return []

    # ── BUG FIX 1: single-dot letters (e.g. A = only dot-1) ─────────────────
    if len(dots) == 1:
        d = dots[0]
        return [BrailleCell(col=0, row=0, pattern=(1,0,0,0,0,0),
                            bbox=(d.cx, d.cy, 0, 0))]

    v_sp, h_sp = _estimate_spacings(dots)
    rows = _cluster_rows(dots, v_sp)

    # Compute actual cy-centre of every dot-row
    row_cys = [float(np.mean([d.cy for d in r])) for r in rows]

    # Measure consecutive inter-row gaps; use their median as the canonical
    # intra-cell vertical step.  For a single-row image, fall back to v_sp*3.
    if len(row_cys) >= 2:
        gaps = [row_cys[k+1] - row_cys[k] for k in range(len(row_cys)-1)]
        median_gap = float(np.median(gaps))
    else:
        median_gap = v_sp * 3.0

    # Two consecutive dot-rows belong to the SAME Braille cell if their gap
    # is ≤ 1.6× the median gap.  Between-cell (between-line) gaps are larger.
    intra_thresh = median_gap * 1.6

    # Group dot-rows into Braille lines (up to 3 dot-rows per line)
    groups: List[List[List[Dot]]] = []
    i = 0
    while i < len(rows):
        grp = [rows[i]]
        for j in range(i+1, min(i+3, len(rows))):
            prev_cy = row_cys[j-1]
            curr_cy = row_cys[j]
            if abs(curr_cy - prev_cy) <= intra_thresh:
                grp.append(rows[j])
            else:
                break
        groups.append(grp)
        i += len(grp)

    cells: List[BrailleCell] = []
    for line_idx, grp in enumerate(groups):
        # Find column centres for this line
        all_cx = [d.cx for row in grp for d in row]
        col_centres = _merge_close(all_cx, h_sp * 0.55)

        # Pair columns into cells  (intra-dot gap < inter-cell gap)
        pairs: List[Tuple[float, Optional[float]]] = []
        c = 0
        while c < len(col_centres):
            left = col_centres[c]
            if c+1 < len(col_centres) and col_centres[c+1] - left <= h_sp * 1.8:
                pairs.append((left, col_centres[c+1]))
                c += 2
            else:
                pairs.append((left, None))
                c += 1

        # ── FIX 2: map each dot-row to correct Braille position 0/1/2 ──────────
        # Letters like K (dots 1,3) have an EMPTY middle row.
        # enumerate(grp) gives indices 0,1 but we need positions 0,2.
        # Scale-invariant unit step: inter-row gap ≈ 3.14 × mean_dot_radius.
        # This ratio is constant across all image scales (verified empirically).
        mean_r   = float(np.mean([d.radius for row in grp for d in row])) or 1.0
        row_unit = mean_r * 3.14   # 1 Braille row step in pixels

        grp_cys = [float(np.mean([d.cy for d in r])) for r in grp]
        ref_cy  = grp_cys[0]

        def cy_to_pos(cy: float) -> int:
            offset = cy - ref_cy
            pos = int(round(offset / row_unit)) if row_unit > 0.1 else 0
            return max(0, min(2, pos))

        for cell_idx, (lx, rx) in enumerate(pairs):
            pat = [0]*6
            x_tol = h_sp * 0.6
            for row_dots, rcy in zip(grp, grp_cys):
                ri = cy_to_pos(rcy)
                for d in row_dots:
                    if abs(d.cx - lx) < x_tol:
                        pat[ri] = 1          # positions 1, 2, 3
                    if rx is not None and abs(d.cx - rx) < x_tol:
                        pat[3+ri] = 1        # positions 4, 5, 6

            # Bounding box
            cell_dots = [
                d for row in grp for d in row
                if abs(d.cx - lx) < h_sp*0.8 or (rx and abs(d.cx - rx) < h_sp*0.8)
            ]
            if cell_dots:
                xs = [d.cx for d in cell_dots]; ys = [d.cy for d in cell_dots]
                bbox = (min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys))
            else:
                bbox = (lx, 0, 0, 0)

            cells.append(BrailleCell(col=cell_idx, row=line_idx,
                                     pattern=tuple(pat), bbox=bbox))
    return cells


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 7+8  DECODER + TEXT RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def decode_cells(cells: List[BrailleCell]) -> str:
    if not cells:
        return ""

    cells_sorted = sorted(cells, key=lambda c: (c.row, c.col))
    lines: Dict[int, str] = {}
    line_buf: List[str] = []
    prev_row = cells_sorted[0].row
    num_mode = cap_next = False

    for cell in cells_sorted:
        if cell.row != prev_row:
            lines[prev_row] = "".join(line_buf)
            line_buf = []; num_mode = cap_next = False
            prev_row = cell.row

        pat = cell.pattern

        if pat == NUMBER_INDICATOR:
            num_mode = True; continue
        if pat == CAPITAL_INDICATOR:
            cap_next = True; continue
        if pat == (0,0,0,0,0,0):
            line_buf.append(' '); num_mode = False; continue

        if num_mode:
            ch = DIGIT_MAP.get(pat)
            if ch:
                line_buf.append(ch); continue
            else:
                num_mode = False

        ch = BRAILLE_MAP.get(pat, '?')
        if cap_next and ch.isalpha():
            ch = ch.upper(); cap_next = False
        line_buf.append(ch)

    lines[prev_row] = "".join(line_buf)
    return "\n".join(lines[k].strip() for k in sorted(lines))


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API  (for SAMVAAD integration)
# ══════════════════════════════════════════════════════════════════════════════

class BrailleRecognizer:
    """
    Drop-in module for SAMVAAD.

    from samvaad_braille import BrailleRecognizer
    text = BrailleRecognizer().recognize("image.png")
    """

    def recognize(self, image: Union[str, Path, np.ndarray], auto_space: bool = True) -> str:
        """
        Recognize Braille text from an image.
        
        Parameters
        ----------
        image : str | Path | np.ndarray
            Input image
        auto_space : bool
            If True, automatically add spaces between words (default: True)
            Useful for Braille images that lack blank cells between words
            
        Returns
        -------
        str
            Decoded text
        """
        bgr   = load_image(image)
        _, bn = preprocess(bgr)
        dots  = detect_dots(bn)
        if not dots:
            return ""
        cells = segment_cells(dots)
        raw_text = decode_cells(cells)
        
        if auto_space:
            return auto_space_text(raw_text, remove_newlines=True)
        return raw_text

    def recognize_with_debug(self, image: Union[str, Path, np.ndarray]) -> Tuple[str, Dict[str, np.ndarray]]:
        """Returns (text, debug_images_dict)."""
        bgr   = load_image(image)
        gray, binary = preprocess(bgr)
        dots  = detect_dots(binary)
        cells = segment_cells(dots) if dots else []
        text  = decode_cells(cells)

        # Build overlays
        debug = {
            "original": bgr.copy(),
            "binary"  : cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR),
        }
        dots_img = bgr.copy()
        for d in dots:
            cv2.circle(dots_img, (int(d.cx), int(d.cy)),
                       max(3, int(d.radius)), (0,200,0), 2)
        debug["dots"] = dots_img

        cells_img = bgr.copy()
        for c in cells:
            x,y,w,h = c.bbox
            cv2.rectangle(cells_img, (int(x)-5, int(y)-5),
                          (int(x+w)+5, int(y+h)+5), (255,80,0), 2)
            cv2.putText(cells_img, "".join(map(str,c.pattern)),
                        (int(x), int(y)-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.35, (0,0,220), 1)
        debug["cells"] = cells_img
        return text, debug


# Convenience function
def recognize_braille(image: Union[str, Path, np.ndarray]) -> str:
    return BrailleRecognizer().recognize(image)


# ══════════════════════════════════════════════════════════════════════════════
#  RUNNER  (when you execute this file directly)
# ══════════════════════════════════════════════════════════════════════════════

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Expected output per filename stem (for --test mode)
_EXPECTED: Dict[str, str] = {}
for _c in "abcdefghijklmnopqrstuvwxyz": _EXPECTED[_c] = _c
for _i in range(10): _EXPECTED[str(_i)] = str(_i)

os.system("")  # enable ANSI colours on Windows
G = "\033[92m"; R = "\033[91m"; Y = "\033[93m"; C = "\033[96m"
B = "\033[1m";  Z = "\033[0m"


def _run_single(path: Path, debug: bool):
    rec = BrailleRecognizer()
    print(f"\n{B}Image:{Z} {path.name}")
    t0 = time.perf_counter()
    if debug:
        text, dbg = rec.recognize_with_debug(path)
        for name, img in dbg.items():
            out = path.parent / f"{path.stem}_debug_{name}.png"
            cv2.imwrite(str(out), img)
            print(f"  {C}Debug saved → {out.name}{Z}")
    else:
        text = rec.recognize(path)
    ms = (time.perf_counter() - t0) * 1000
    print(f"{B}Result:{Z} {G}{text!r}{Z}")
    print(f"{B}Time:  {Z} {ms:.0f} ms\n")


def _run_test(folder: Path, debug: bool):
    imgs = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not imgs:
        print(f"{Y}No images found in {folder}{Z}"); return

    rec = BrailleRecognizer()
    print(f"\n{B}{'='*62}{Z}")
    print(f"{B}  SAMVAAD – Braille Dataset Test{Z}")
    print(f"  Folder : {folder}   |   Images : {len(imgs)}")
    print(f"{B}{'='*62}{Z}")
    print(f"  {'File':<22} {'Expected':<10} {'Got':<22} Status   ms")
    print(f"  {'-'*22} {'-'*10} {'-'*22} {'-'*8} ---")

    passed = failed = info_ct = 0
    fails = []

    for p in imgs:
        exp = _EXPECTED.get(p.stem.lower())
        t0 = time.perf_counter()
        if debug:
            text, dbg = rec.recognize_with_debug(p)
            for name, img in dbg.items():
                cv2.imwrite(str(p.parent / f"{p.stem}_debug_{name}.png"), img)
        else:
            text = rec.recognize(p)
        ms = (time.perf_counter() - t0)*1000
        got = text.strip().lower()

        if exp is None:
            status = f"{C}INFO{Z}"; info_ct += 1
        elif got == exp:
            status = f"{G}PASS{Z}"; passed += 1
        else:
            status = f"{R}FAIL{Z}"; failed += 1
            fails.append((p.name, exp, got))

        print(f"  {p.name:<22} {repr(exp):<10} {repr(got):<22} {status}   {ms:4.0f}")

    total = passed + failed
    acc = (passed/total*100) if total else 0
    print(f"\n{B}{'='*62}{Z}")
    print(f"  Graded: {total}  |  {G}Pass: {passed}{Z}  |  {R}Fail: {failed}{Z}  |  {C}Info: {info_ct}{Z}")
    print(f"  Accuracy: {B}{acc:.1f}%{Z}")
    if fails:
        print(f"\n  {R}Failures:{Z}")
        for fn, ex, gt in fails:
            print(f"    {fn:<22} expected={ex!r}  got={gt!r}")
    print(f"{B}{'='*62}{Z}\n")


def _run_interactive(debug: bool):
    rec = BrailleRecognizer()
    print(f"\n{B}{'='*55}{Z}")
    print(f"{B}  SAMVAAD – Braille Recognition  (interactive){Z}")
    print(f"  Paste or type an image path.  'q' to quit.")
    print(f"{B}{'='*55}{Z}\n")
    while True:
        try:
            raw = input("Image path: ").strip().strip('"').strip("'")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!"); break
        if raw.lower() in ("q","quit","exit",""):
            print("Bye!"); break
        p = Path(raw)
        if not p.exists():
            print(f"  {R}Not found: {p}{Z}\n"); continue
        if p.suffix.lower() not in IMG_EXTS:
            print(f"  {Y}Not an image: {p}{Z}\n"); continue

        t0 = time.perf_counter()
        if debug:
            text, dbg = rec.recognize_with_debug(p)
            for name, img in dbg.items():
                out = p.parent / f"{p.stem}_debug_{name}.png"
                cv2.imwrite(str(out), img)
                print(f"  {C}Debug → {out.name}{Z}")
        else:
            text = rec.recognize(p)
        ms = (time.perf_counter() - t0)*1000
        print(f"\n  {B}Result:{Z} {G}{text!r}{Z}")
        print(f"  {B}Time:  {Z} {ms:.0f} ms\n")


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        prog="samvaad_braille",
        description="SAMVAAD Braille Recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python samvaad_braille.py
  python samvaad_braille.py D:\\SAMVAAD\\dataset\\G.png
  python samvaad_braille.py --test D:\\SAMVAAD\\dataset
  python samvaad_braille.py --test D:\\SAMVAAD\\dataset --debug
        """,
    )
    ap.add_argument("image",  nargs="?", type=Path, default=None,
                    help="Single image path")
    ap.add_argument("--test", "-t", metavar="FOLDER", type=Path, default=None,
                    help="Test all images in a folder")
    ap.add_argument("--debug", "-d", action="store_true",
                    help="Save debug overlay images")
    args = ap.parse_args()

    if args.test:
        if not args.test.is_dir():
            print(f"{R}Not a folder: {args.test}{Z}"); sys.exit(1)
        _run_test(args.test, args.debug)
    elif args.image:
        if not args.image.exists():
            print(f"{R}File not found: {args.image}{Z}"); sys.exit(1)
        _run_single(args.image, args.debug)
    else:
        _run_interactive(args.debug)