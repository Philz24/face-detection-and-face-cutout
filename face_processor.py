"""
Face detection, cropping, and transparent face+neck extraction for the
Color Analysis workflow.

Design goals:
- Work on many different portrait photos, not just one example
- Keep face + neck only
- Exclude most hair, shoulders, and background
- Produce a reusable transparent PNG
- Stay compatible with the existing workflow interface

Uses:
- dlib HOG face detector
- dlib 68-point facial landmarks
- Pillow for image transforms and alpha compositing
- OpenCV for mask refinement and light morphology
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import dlib
import numpy as np
from PIL import Image, ImageFilter

# dlib HOG-based frontal face detector (no model file required)
_detector = dlib.get_frontal_face_detector()

_SHAPE_PREDICTOR_PATH = os.environ.get(
    "SHAPE_PREDICTOR_PATH",
    os.path.expanduser("~/.local/share/color-analysis/shape_predictor_68_face_landmarks.dat"),
)

_predictor = None
if os.path.exists(_SHAPE_PREDICTOR_PATH):
    _predictor = dlib.shape_predictor(_SHAPE_PREDICTOR_PATH)


def detect_faces(image_path: str) -> list[dict]:
    """
    Detect all faces in an image.

    Returns a list of dicts with keys: x, y, w, h, confidence
    Only returns faces with confidence >= 0.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections, scores, _ = _detector.run(gray, 1, -0.5)

    faces = []
    for rect, score in zip(detections, scores):
        if score < 0:
            continue
        faces.append(
            {
                "x": max(0, rect.left()),
                "y": max(0, rect.top()),
                "w": rect.width(),
                "h": rect.height(),
                "confidence": float(score),
            }
        )

    return faces


def _shape_to_np(shape: dlib.full_object_detection) -> np.ndarray:
    """Convert dlib shape predictor output to a NumPy array of shape (68, 2)."""
    coords = np.zeros((68, 2), dtype=np.int32)
    for i in range(68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def _fit_on_canvas(img: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Resize without distortion and center on a transparent canvas."""
    canvas = Image.new("RGBA", target_size, (0, 0, 0, 0))
    fitted = img.copy()
    fitted.thumbnail(target_size, Image.LANCZOS)
    x = (target_size[0] - fitted.width) // 2
    y = (target_size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y), fitted)
    return canvas


def _clamp_points(points: np.ndarray, w: int, h: int) -> np.ndarray:
    """Clamp polygon points so they stay inside the image."""
    pts = points.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def _select_primary_face(faces: list[dict], image_size: tuple[int, int]) -> dict:
    """
    Pick the most likely main subject face.

    Better than pure confidence:
    - prefers larger faces
    - prefers faces closer to center
    - still uses confidence
    """
    if not faces:
        raise ValueError("No faces available to select")

    if len(faces) == 1:
        return faces[0]

    img_w, img_h = image_size
    cx = img_w / 2.0
    cy = img_h / 2.0
    max_area = max(f["w"] * f["h"] for f in faces)

    best = None
    best_score = -1e18

    for f in faces:
        area = f["w"] * f["h"]
        face_cx = f["x"] + f["w"] / 2.0
        face_cy = f["y"] + f["h"] / 2.0

        dx = (face_cx - cx) / max(img_w, 1)
        dy = (face_cy - cy) / max(img_h, 1)
        center_penalty = math.sqrt(dx * dx + dy * dy)

        area_score = area / max(max_area, 1)
        conf_score = f["confidence"]

        score = (2.0 * area_score) + (0.35 * conf_score) - (0.9 * center_penalty)

        if score > best_score:
            best_score = score
            best = f

    return best


def _detect_faces_in_pil(img: Image.Image) -> list[dlib.rectangle]:
    """Detect dlib face rectangles in a PIL image."""
    rgb = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detections, scores, _ = _detector.run(gray, 1, -0.5)

    rects = []
    for rect, score in zip(detections, scores):
        if score >= 0:
            rects.append(rect)
    return rects


def _find_best_rect_in_crop(img: Image.Image) -> dlib.rectangle | None:
    """Detect faces in a crop and return the best candidate."""
    rects = _detect_faces_in_pil(img)
    if not rects:
        return None

    w, h = img.size
    cx = w / 2.0
    cy = h / 2.0

    best = None
    best_score = -1e18

    for rect in rects:
        rw = rect.width()
        rh = rect.height()
        area = rw * rh
        rcx = (rect.left() + rect.right()) / 2.0
        rcy = (rect.top() + rect.bottom()) / 2.0

        dx = (rcx - cx) / max(w, 1)
        dy = (rcy - cy) / max(h, 1)
        center_penalty = math.sqrt(dx * dx + dy * dy)

        score = area - (0.20 * area * center_penalty)

        if score > best_score:
            best_score = score
            best = rect

    return best


def _get_landmarks(img: Image.Image, rect: dlib.rectangle) -> np.ndarray:
    """Predict 68 landmarks for rect in PIL image."""
    if _predictor is None:
        raise RuntimeError(
            "Shape predictor not available. "
            f"Expected file at: {_SHAPE_PREDICTOR_PATH}"
        )

    rgb = np.array(img.convert("RGB"))
    shape = _predictor(rgb, rect)
    return _shape_to_np(shape)


def _crop_anchor_region(img: Image.Image, face: dict) -> Image.Image:
    """
    Create an initial crop around the detected face.

    Conservative top padding because we do NOT want lots of hair.
    More bottom padding for chin/neck.
    """
    img_w, img_h = img.size
    x, y, w, h = face["x"], face["y"], face["w"], face["h"]

    pad_x = int(w * 0.26)
    pad_top = int(h * 0.26)
    pad_bot = int(h * 0.72)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_top)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_bot)

    return img.crop((x1, y1, x2, y2))


def _compute_eye_angle(landmarks: np.ndarray) -> float:
    """Compute angle in degrees between eye centers."""
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    return math.degrees(math.atan2(dy, dx))


def _rotate_rgba(img: Image.Image, angle_deg: float) -> Image.Image:
    """Rotate with transparent fill."""
    return img.rotate(
        angle_deg,
        resample=Image.BICUBIC,
        expand=True,
        fillcolor=(0, 0, 0, 0),
    )


def _quadratic_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 18) -> np.ndarray:
    """Sample n points along a quadratic Bézier curve."""
    ts = np.linspace(0.0, 1.0, n)
    pts = []
    for t in ts:
        pt = ((1 - t) ** 2) * p0 + (2 * (1 - t) * t) * p1 + (t ** 2) * p2
        pts.append(pt)
    return np.array(pts, dtype=np.int32)


def _face_metrics(landmarks: np.ndarray) -> dict[str, int | float]:
    """Shared facial measurements used across geometry and refinement steps."""
    brow_y = int(np.mean(landmarks[17:27, 1]))
    chin_y = int(landmarks[8, 1])
    face_h = max(1, chin_y - brow_y)
    face_w = max(1, int(landmarks[16, 0] - landmarks[0, 0]))
    center_x = int((landmarks[0, 0] + landmarks[16, 0]) / 2)
    return {
        "brow_y": brow_y,
        "chin_y": chin_y,
        "face_h": face_h,
        "face_w": face_w,
        "center_x": center_x,
    }


def _base_face_neck_geometry(size: tuple[int, int], landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build conservative face and neck polygons from landmarks."""
    w, h = size

    jaw = landmarks[0:17]
    chin = landmarks[8]

    metrics = _face_metrics(landmarks)
    brow_mid_y = int(metrics["brow_y"])
    face_height = float(metrics["face_h"])
    face_width = float(metrics["face_w"])
    mid_x = float(metrics["center_x"])
    left_eye = np.mean(landmarks[36:42], axis=0)
    right_eye = np.mean(landmarks[42:48], axis=0)
    eye_mid_y = int(np.mean([left_eye[1], right_eye[1]]))
    eye_to_brow = max(1.0, float(eye_mid_y - brow_mid_y))

    # Keep a bit more forehead while still staying conservative vs hairline.
    top_lift = max(eye_to_brow * 1.65, face_height * 0.22)
    top_y = int(brow_mid_y - top_lift)

    left_temple = np.array(
        [
            int(landmarks[17, 0] - 0.005 * face_width),
            int(brow_mid_y - 0.06 * face_height),
        ],
        dtype=np.int32,
    )
    right_temple = np.array(
        [
            int(landmarks[26, 0] + 0.005 * face_width),
            int(brow_mid_y - 0.06 * face_height),
        ],
        dtype=np.int32,
    )

    center_top = np.array([int(mid_x), top_y], dtype=np.int32)
    right_ctrl = np.array([int(landmarks[24, 0]), int(top_y + 0.10 * face_height)], dtype=np.int32)
    left_ctrl = np.array([int(landmarks[19, 0]), int(top_y + 0.10 * face_height)], dtype=np.int32)

    arc_right = _quadratic_bezier(right_temple, right_ctrl, center_top, n=15)
    arc_left = _quadratic_bezier(center_top, left_ctrl, left_temple, n=15)
    forehead_arc = np.vstack([arc_right, arc_left[1:]])

    face_poly = np.vstack([jaw, forehead_arc])
    face_poly = _clamp_points(face_poly, w, h)

    neck_top_left = jaw[5]
    neck_top_right = jaw[11]
    neck_bottom_y = int(chin[1] + 0.30 * face_height)
    neck_half_width = int(0.12 * face_width)

    neck_poly = np.array(
        [
            neck_top_left,
            neck_top_right,
            [int(mid_x + neck_half_width), neck_bottom_y],
            [int(mid_x - neck_half_width), neck_bottom_y],
        ],
        dtype=np.int32,
    )
    neck_poly = _clamp_points(neck_poly, w, h)

    return face_poly, neck_poly


def _landmark_skin_model(rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Estimate a skin-likelihood mask using cheek/forehead landmark samples."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    cr = ycrcb[:, :, 1].astype(np.float32)
    cb = ycrcb[:, :, 2].astype(np.float32)

    sample_idx = [1, 2, 3, 31, 32, 33, 34, 35, 13, 14, 15, 21, 22, 27]
    samples = landmarks[sample_idx]
    sample_vals = []
    for x, y in samples:
        x0, x1 = max(0, x - 2), min(rgb.shape[1], x + 3)
        y0, y1 = max(0, y - 2), min(rgb.shape[0], y + 3)
        patch = ycrcb[y0:y1, x0:x1]
        if patch.size:
            sample_vals.append(patch[:, :, 1:3].reshape(-1, 2))

    if not sample_vals:
        # global fallback bounds
        return ((cr > 132) & (cr < 178) & (cb > 84) & (cb < 140)).astype(np.uint8) * 255

    vals = np.vstack(sample_vals).astype(np.float32)
    mean = vals.mean(axis=0)
    std = np.maximum(vals.std(axis=0), 7.5)

    d = ((cr - mean[0]) / (1.7 * std[0])) ** 2 + ((cb - mean[1]) / (1.7 * std[1])) ** 2
    skin = (d <= 1.0).astype(np.uint8) * 255

    # union with broad generic skin prior for robustness across tones/lighting
    broad = ((cr > 125) & (cr < 182) & (cb > 77) & (cb < 145)).astype(np.uint8) * 255
    skin = cv2.bitwise_or(skin, broad)

    skin = cv2.medianBlur(skin, 5)
    return skin


def _facial_feature_keep_mask(shape: tuple[int, int], landmarks: np.ndarray) -> np.ndarray:
    """
    Build a mask of key facial features that must never be carved out.

    This prevents dark eyebrows/irises/lashes from being removed by skin-only gating.
    """
    h, w = shape
    keep = np.zeros((h, w), dtype=np.uint8)

    # Brows and eyes are the main failure points; include a small safety margin.
    for region in (landmarks[17:22], landmarks[22:27], landmarks[36:42], landmarks[42:48]):
        hull = cv2.convexHull(region.astype(np.int32))
        cv2.fillConvexPoly(keep, hull, 255)

    # Keep a thin bridge between brows/eyes so we don't create cut lines.
    bridge = np.array(
        [
            landmarks[21],
            landmarks[22],
            landmarks[27],
            landmarks[28],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(keep, bridge, 255)

    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    return keep


def _facial_hair_support_masks(shape: tuple[int, int], landmarks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Build jaw-constrained facial-hair support and central seed masks.
    """
    h, w = shape
    support = np.zeros((h, w), dtype=np.uint8)
    seed = np.zeros((h, w), dtype=np.uint8)

    face_w = max(1, int(landmarks[16, 0] - landmarks[0, 0]))
    brow_y = int(np.mean(landmarks[17:27, 1]))
    chin_y = int(landmarks[8, 1])
    face_h = max(1, int(chin_y - brow_y))
    center_x = int((landmarks[0, 0] + landmarks[16, 0]) / 2)
    mouth_center = np.mean(landmarks[48:68], axis=0)
    upper_lip_y = int(np.mean(landmarks[50:53, 1]))

    # Jaw-based beard polygon (points 2..14 + slight chin extension).
    jaw = landmarks[2:15].astype(np.int32)
    ext_left = np.array([jaw[0, 0] - int(0.02 * face_w), jaw[0, 1] + int(0.06 * face_h)], dtype=np.int32)
    ext_right = np.array([jaw[-1, 0] + int(0.02 * face_w), jaw[-1, 1] + int(0.06 * face_h)], dtype=np.int32)
    chin_ext = np.array([center_x, int(chin_y + 0.10 * face_h)], dtype=np.int32)
    beard_poly = np.vstack([ext_left, jaw, ext_right, chin_ext]).astype(np.int32)
    beard_poly = _clamp_points(beard_poly, w, h)
    cv2.fillPoly(support, [beard_poly], 255)

    # Separate moustache support around upper lip.
    cv2.ellipse(
        support,
        (int(mouth_center[0]), int(upper_lip_y - 0.03 * face_h)),
        (max(8, int(0.15 * face_w)), max(6, int(0.07 * face_h))),
        0,
        0,
        360,
        255,
        -1,
    )

    # Central seed used to validate beard components.
    cv2.ellipse(
        seed,
        (center_x, int(mouth_center[1] + 0.05 * face_h)),
        (max(8, int(0.14 * face_w)), max(8, int(0.16 * face_h))),
        0,
        0,
        360,
        255,
        -1,
    )
    cv2.ellipse(
        seed,
        (int(mouth_center[0]), int(upper_lip_y - 0.03 * face_h)),
        (max(6, int(0.10 * face_w)), max(4, int(0.05 * face_h))),
        0,
        0,
        360,
        255,
        -1,
    )
    # Side beard anchors to avoid clipping lower jaw corners.
    for p in (landmarks[48], landmarks[54], landmarks[5], landmarks[11]):
        cv2.ellipse(
            seed,
            (int(p[0]), int(p[1] + 0.05 * face_h)),
            (max(4, int(0.06 * face_w)), max(4, int(0.06 * face_h))),
            0,
            0,
            360,
            255,
            -1,
        )
    support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    seed = cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    return support, seed


def _restore_small_inner_fragments(keep: np.ndarray, prior_mask: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Fill tiny interior dropouts caused by conservative gating/morphology.

    Only restores small holes well inside the face envelope (not border/hair regions).
    """
    restored = keep.copy()
    face_width = max(1, int(landmarks[16, 0] - landmarks[0, 0]))
    interior = cv2.erode(
        prior_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, face_width // 11), max(9, face_width // 11))),
        iterations=1,
    )

    holes = cv2.bitwise_and(interior, cv2.bitwise_not(restored))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), connectivity=8)

    max_hole_area = max(40, int(0.0035 * face_width * face_width))
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area <= max_hole_area:
            restored[labels == i] = 255

    return restored


def _filter_beard_components(candidates: np.ndarray, support: np.ndarray, seed: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Keep only jaw-constrained beard components anchored to central seed."""
    keep = np.zeros_like(candidates)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats((candidates > 0).astype(np.uint8), connectivity=8)
    face_w = max(1, int(landmarks[16, 0] - landmarks[0, 0]))
    brow_y = int(np.mean(landmarks[17:27, 1]))
    chin_y = int(landmarks[8, 1])
    face_h = max(1, int(chin_y - brow_y))
    x_min = landmarks[2, 0] - int(0.14 * face_w)
    x_max = landmarks[14, 0] + int(0.14 * face_w)
    y_min = int(np.mean(landmarks[50:53, 1]) - 0.10 * face_h)
    y_max = chin_y + int(0.18 * face_h)
    min_area = max(25, int(0.0008 * face_w * face_w))

    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = centroids[i]
        if not (x_min <= cx <= x_max and y_min <= cy <= y_max):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        if cv2.countNonZero(cv2.bitwise_and(comp, seed)) == 0:
            continue
        comp = cv2.bitwise_and(comp, support)
        keep = cv2.bitwise_or(keep, comp)
    return keep


def _refine_mask_with_grabcut(rgb: np.ndarray, prior_mask: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Refine conservative mask with GrabCut, then re-apply anti-hair constraints.

    This keeps boundary detail on face/neck while rejecting background and hair.
    """
    h, w = prior_mask.shape
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # sure background: far outside prior
    dilated = cv2.dilate(prior_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=1)
    gc_mask[dilated == 0] = cv2.GC_BGD

    # sure foreground: shrunken central face + upper neck
    eroded = cv2.erode(prior_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    gc_mask[eroded > 0] = cv2.GC_PR_FGD

    metrics = _face_metrics(landmarks)
    brow_y = int(metrics["brow_y"])
    chin_y = int(metrics["chin_y"])
    face_h = int(metrics["face_h"])
    center_x = int(metrics["center_x"])

    core = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        core,
        (center_x, int(brow_y + 0.55 * face_h)),
        (max(8, int(0.20 * face_h)), max(10, int(0.42 * face_h))),
        0,
        0,
        360,
        255,
        -1,
    )
    cv2.rectangle(
        core,
        (center_x - max(5, int(0.08 * face_h)), int(chin_y - 0.02 * face_h)),
        (center_x + max(5, int(0.08 * face_h)), int(chin_y + 0.22 * face_h)),
        255,
        -1,
    )
    gc_mask[core > 0] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(rgb, gc_mask, None, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_MASK)

    fg = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Enforce conservative geometry envelope.
    fg = cv2.bitwise_and(fg, prior_mask)

    # Skin gating near forehead and temple bands to suppress hair.
    skin = _landmark_skin_model(rgb, landmarks)
    gate = np.zeros_like(fg)
    # Gate only the upper-forehead band; avoid carving near brow ridge.
    top_gate_y = max(0, int(brow_y - 0.05 * face_h))
    side_margin = int(max(6, 0.10 * (landmarks[16, 0] - landmarks[0, 0])))
    gate[:top_gate_y, :] = 255
    side_gate_bottom = max(0, int(chin_y - 0.20 * face_h))
    gate[:side_gate_bottom, : max(1, landmarks[1, 0] + side_margin)] = 255
    gate[:side_gate_bottom, min(w - 1, landmarks[15, 0] - side_margin) :] = 255

    feature_keep = _facial_feature_keep_mask((h, w), landmarks)
    facial_hair_support, facial_hair_seed = _facial_hair_support_masks((h, w), landmarks)

    # In gate regions keep only skin-likely pixels, except protected facial features.
    keep = cv2.bitwise_or(
        cv2.bitwise_and(fg, cv2.bitwise_not(gate)),
        cv2.bitwise_and(fg, cv2.bitwise_and(gate, skin)),
    )
    keep = cv2.bitwise_or(keep, cv2.bitwise_and(feature_keep, prior_mask))
    beard_candidates = cv2.bitwise_and(fg, facial_hair_support)
    beard_keep = _filter_beard_components(beard_candidates, facial_hair_support, facial_hair_seed, landmarks)
    keep = cv2.bitwise_or(keep, beard_keep)

    # Light morphology to remove speckles and hard corners.
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    keep = cv2.erode(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    keep = cv2.bitwise_or(keep, cv2.bitwise_and(feature_keep, prior_mask))
    keep = cv2.bitwise_or(keep, beard_keep)
    keep = _restore_small_inner_fragments(keep, prior_mask, landmarks)

    return keep


def _adaptive_round_contour(prior: np.ndarray, rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Smooth top and side outline where skin evidence supports the expansion.

    Combines forehead and cheek-side rounding into one pass to reduce overlap.
    """
    h, w = prior.shape
    metrics = _face_metrics(landmarks)
    face_width = int(metrics["face_w"])
    brow_y = int(metrics["brow_y"])
    chin_y = int(metrics["chin_y"])
    face_h = int(metrics["face_h"])
    center_x = int(metrics["center_x"])

    skin = _landmark_skin_model(rgb, landmarks)

    addition = np.zeros_like(prior)

    # Forehead cap.
    rounded_cap = np.zeros_like(prior)
    cap_center_y = int(brow_y - 0.07 * face_h)
    cap_rx = max(10, int(0.37 * face_width))
    cap_ry = max(8, int(0.24 * face_h))
    cv2.ellipse(rounded_cap, (center_x, cap_center_y), (cap_rx, cap_ry), 0, 0, 360, 255, -1)

    # Keep the rounded addition focused on forehead space.
    forehead_band = np.zeros_like(prior)
    y_top = max(0, int(brow_y - 0.42 * face_h))
    y_bottom = min(h, int(brow_y + 0.12 * face_h))
    forehead_band[y_top:y_bottom, :] = 255
    x_left = max(0, int(landmarks[2, 0] - 0.08 * face_width))
    x_right = min(w, int(landmarks[14, 0] + 0.08 * face_width))
    side_window = np.zeros_like(prior)
    side_window[:, x_left:x_right] = 255

    forehead_add = cv2.bitwise_and(rounded_cap, forehead_band)
    forehead_add = cv2.bitwise_and(forehead_add, side_window)
    addition = cv2.bitwise_or(addition, forehead_add)

    # Side cheek rounding.
    side_add = np.zeros_like(prior)
    y_top = max(0, int(brow_y + 0.02 * face_h))
    y_bottom = min(h, int(chin_y + 0.16 * face_h))
    vertical_band = np.zeros_like(prior)
    vertical_band[y_top:y_bottom, :] = 255

    left_mid = landmarks[2:7]
    right_mid = landmarks[10:15]
    left_center = (
        int(np.mean(left_mid[:, 0]) - 0.03 * face_width),
        int(np.mean(left_mid[:, 1]) + 0.02 * face_h),
    )
    right_center = (
        int(np.mean(right_mid[:, 0]) + 0.03 * face_width),
        int(np.mean(right_mid[:, 1]) + 0.02 * face_h),
    )
    rx = max(8, int(0.12 * face_width))
    ry = max(12, int(0.30 * face_h))
    cv2.ellipse(side_add, left_center, (rx, ry), 0, 0, 360, 255, -1)
    cv2.ellipse(side_add, right_center, (rx, ry), 0, 0, 360, 255, -1)

    side_window = np.zeros_like(prior)
    left_x0 = max(0, int(landmarks[0, 0] - 0.10 * face_width))
    left_x1 = min(w, int(landmarks[5, 0] + 0.04 * face_width))
    right_x0 = max(0, int(landmarks[11, 0] - 0.04 * face_width))
    right_x1 = min(w, int(landmarks[16, 0] + 0.10 * face_width))
    side_window[:, left_x0:left_x1] = 255
    side_window[:, right_x0:right_x1] = 255

    side_add = cv2.bitwise_and(side_add, vertical_band)
    side_add = cv2.bitwise_and(side_add, side_window)
    addition = cv2.bitwise_or(addition, side_add)

    addition = cv2.bitwise_and(addition, skin)
    addition = cv2.morphologyEx(
        addition,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )

    return cv2.bitwise_or(prior, addition)


def _expand_for_facial_hair(prior: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Expand lower-face support so beard edges are not clipped by a too-tight prior.
    """
    support, _ = _facial_hair_support_masks(prior.shape, landmarks)
    dilated = cv2.dilate(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
    extra = cv2.bitwise_and(dilated, cv2.bitwise_not(prior))
    beard_extra = cv2.bitwise_and(extra, support)
    return cv2.bitwise_or(prior, beard_extra)


def _build_face_neck_mask(size: tuple[int, int], landmarks: np.ndarray, rgb: np.ndarray) -> Image.Image:
    """
    Build face+neck alpha mask with geometric prior + GrabCut refinement.

    - conservative forehead/sides to drop hair
    - stable neck shape
    - edge-preserving segmentation for natural contour
    """
    w, h = size
    face_poly, neck_poly = _base_face_neck_geometry(size, landmarks)

    prior = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(prior, [face_poly], 255)
    cv2.fillPoly(prior, [neck_poly], 255)
    prior = _adaptive_round_contour(prior, rgb, landmarks)
    prior = _expand_for_facial_hair(prior, landmarks)

    refined = _refine_mask_with_grabcut(rgb, prior, landmarks)

    mask = Image.fromarray(refined, mode="L")
    # tiny feather to avoid jagged edge without obvious halo/vignette
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.45))
    return mask


def crop_face_transparent(
    image_path: str,
    face: dict,
    output_path: str,
    target_size: tuple[int, int] = (400, 533),
) -> str:
    """
    Produce a face + neck transparent PNG.

    Pipeline:
    1. Create an initial crop around the detected face
    2. Predict landmarks
    3. Align crop by eye angle
    4. Re-detect / re-landmark in the aligned crop
    5. Build conservative + refined face+neck mask
    6. Trim transparent margin and fit on target canvas
    """
    if _predictor is None:
        raise RuntimeError(
            "Shape predictor not available. "
            f"Expected file at: {_SHAPE_PREDICTOR_PATH}"
        )

    img = Image.open(image_path).convert("RGBA")

    cropped = _crop_anchor_region(img, face)
    crop_rect = _find_best_rect_in_crop(cropped)
    if crop_rect is None:
        raise ValueError("Could not detect face in cropped region")

    landmarks = _get_landmarks(cropped, crop_rect)

    eye_angle = _compute_eye_angle(landmarks)
    aligned = _rotate_rgba(cropped, -eye_angle)

    aligned_rect = _find_best_rect_in_crop(aligned)
    if aligned_rect is None:
        aligned = cropped
        aligned_rect = crop_rect

    aligned_landmarks = _get_landmarks(aligned, aligned_rect)
    aligned_rgb = np.array(aligned.convert("RGB"))
    mask = _build_face_neck_mask(aligned.size, aligned_landmarks, aligned_rgb)

    result = Image.new("RGBA", aligned.size, (0, 0, 0, 0))
    result.paste(aligned, (0, 0), mask)

    bbox = result.getbbox()
    raw_result = result.crop(bbox) if bbox else result

    if os.environ.get("SAVE_RAW_FACE_CUTOUT", "0") == "1":
        raw_path = os.path.splitext(output_path)[0] + "_raw.png"
        os.makedirs(os.path.dirname(raw_path) if os.path.dirname(raw_path) else ".", exist_ok=True)
        raw_result.save(raw_path, "PNG")

    final_result = _fit_on_canvas(raw_result, target_size)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    final_result.save(output_path, "PNG")
    return output_path


def crop_face(
    image_path: str,
    face: dict,
    output_path: str,
    padding_ratio: float = 0.4,
    target_size: tuple[int, int] = (400, 500),
) -> str:
    """Simple rectangular crop (no transparency). Kept for backwards compatibility."""
    img = Image.open(image_path)
    img_w, img_h = img.size

    x, y, w, h = face["x"], face["y"], face["w"], face["h"]

    pad_x = int(w * padding_ratio)
    pad_y_top = int(h * padding_ratio * 1.2)
    pad_y_bottom = int(h * padding_ratio * 1.5)

    crop_x1 = max(0, x - pad_x)
    crop_y1 = max(0, y - pad_y_top)
    crop_x2 = min(img_w, x + w + pad_x)
    crop_y2 = min(img_h, y + h + pad_y_bottom)

    cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    cropped = cropped.resize(target_size, Image.LANCZOS)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    cropped.save(output_path, quality=95)
    return output_path


def process_client_photo(
    image_path: str,
    output_dir: str,
    client_name: str = "client",
) -> list[str]:
    """
    Full pipeline: detect faces, select the primary face, create a transparent
    face+neck PNG, and save it to output_dir.

    Returns a list with the path to the transparent PNG.
    """
    faces = detect_faces(image_path)
    if not faces:
        raise ValueError(f"No faces detected in {image_path}")

    with Image.open(image_path) as img:
        img_w, img_h = img.size

    primary = _select_primary_face(faces, (img_w, img_h))

    safe_name = client_name.replace(" ", "_").lower()
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{safe_name}_face_transparent.png")

    crop_face_transparent(image_path, primary, out_path)
    return [out_path]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python face_processor.py <image_path> [output_dir]")
        sys.exit(1)

    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/color-analysis/faces"
    results = process_client_photo(img_path, out_dir)

    for r in results:
        print(f"Saved: {r}")