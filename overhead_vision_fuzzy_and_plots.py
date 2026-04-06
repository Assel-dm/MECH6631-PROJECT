"""
# "TUNE:" markers for parameters to adjust.
"""

import cv2
import numpy as np
import math
import time
import argparse
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import csv
import matplotlib.pyplot as plt

try:
    import serial
except Exception:
    serial = None

# ============================================================
# --------------------------- UTILS ---------------------------
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp a scalar value into the closed interval [lo, hi]"""
    return max(lo, min(hi, x))


def angle_wrap(a: float) -> float:
    """Wrap an angle to the interval [-pi, pi]"""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a

def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """Calculate the Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])

def clean_mask(mask: np.ndarray, ksize: int = 5, open_it: int = 1, close_it: int = 1) -> np.ndarray:
    """Clean a binary mask using morphological operations."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_it)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_it)
    return mask


class EMA:
    """exponential moving average filter"""
    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value

# ============================================================
# -----------------Membership functions-----------------------
# ============================================================

def grade(x: float, x0: float, x1: float) -> float:
    """
    Rising fuzzy membership function
    Returns:
    - 0 below x0
    - 1 above x1
    - linear interpolation in between
    """
    if x <= x0:
        return 0.0
    if x >= x1:
        return 1.0
    return (x - x0) / max(1e-9, (x1 - x0))


def reverse_grade(x: float, x0: float, x1: float) -> float:
    """Falling fuzzy membership function"""
    return 1.0 - grade(x, x0, x1)


def triangle(x: float, a: float, b: float, c: float) -> float:
    """Calculate the triangle membership function."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / max(1e-9, (b - a))
    return (c - x) / max(1e-9, (c - b))


# ============================================================
# ---------------------- HSV SEGMENTATION --------------------
# ============================================================


@dataclass
class HSVRange:
    h_lo: int
    h_hi: int


def mask_hsv(frame_bgr: np.ndarray,
             hue_ranges: List[HSVRange],
             s_min: int,
             v_min: int,
             morph_ksize: int = 5,
             morph_close_it: int = 2) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    base = ((s >= s_min) & (v >= v_min)).astype(np.uint8) * 255
    mask = np.zeros_like(base)
    for r in hue_ranges:
        mask_r = ((h >= r.h_lo) & (h <= r.h_hi)).astype(np.uint8) * 255
        mask = cv2.bitwise_or(mask, mask_r)

    mask = cv2.bitwise_and(mask, base)
    return clean_mask(mask, ksize=morph_ksize, open_it=1, close_it=morph_close_it)

# ============================================================
# ---------------------- ROBOT COMMANDER ---------------------
# ============================================================

class RobotCommander:
    def __init__(self, port: Optional[str], baud: int, dry_run: bool, print_rate_hz: float = 5.0):
        self.dry_run = dry_run or (port is None) or (serial is None)
        self.ser = None
        self._last_print = 0.0
        self._print_period = 1.0 / max(1e-3, print_rate_hz)
        if not self.dry_run:
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(2.0)

    def send(self, left: float, right: float, laser: bool):
        msg = f"L={left:+.2f} R={right:+.2f} LAS={1 if laser else 0}\n"
        if self.ser is not None:
            self.ser.write(msg.encode("ascii"))
        else:
            now = time.time()
            if now - self._last_print >= self._print_period:
                self._last_print = now
                print("[CMD]", msg.strip())

    def close(self):
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass

# ============================================================
# ----------------- PAIRING : BLOBS INTO ROBOTS ----------------
# ============================================================

@dataclass
class Blob:
    center: Tuple[float, float]
    area: float
    contour: np.ndarray


def extract_blobs(mask: np.ndarray, min_area_px: float) -> List[Blob]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: List[Blob] = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area_px:
            continue
        M = cv2.moments(c)
        if abs(M["m00"]) < 1e-6:
            continue
        cx = float(M["m10"] / M["m00"])
        cy = float(M["m01"] / M["m00"])
        blobs.append(Blob(center=(cx, cy), area=area, contour=c))
    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


@dataclass
class RobotDet:
    """One robot detection = (front BLUE blob, rear RED blob)"""
    front: Blob
    rear: Blob
    x: float
    y: float
    theta: float # heading = rear -> front (rad)
    sep_px: float # marker separation in pixels


def compute_pose(front: Blob, rear: Blob) -> RobotDet:
    """
    Compute robot pose from markers:
      - (x,y) = midpoint
      - theta = angle(rear->front)
      - sep_px = distance between markers (pixels)
    """
    fx, fy = front.center
    rx, ry = rear.center
    dx = fx - rx
    dy = fy - ry
    sep = math.hypot(dx, dy)
    th = math.atan2(dy, dx)
    x = 0.5 * (fx + rx)
    y = 0.5 * (fy + ry)
    return RobotDet(front=front, rear=rear, x=x, y=y, theta=th, sep_px=sep)


def estimate_marker_sep_px(front_blobs: List[Blob], rear_blobs: List[Blob]) -> Optional[float]:
    """
    Estimate marker separation in pixels:
      - For each front blob, find nearest rear blob distance
      - Return median of those distances 
    """
    if not front_blobs or not rear_blobs:
        return None
    dists = []
    for f in front_blobs:
        fx, fy = f.center
        best = None
        for r in rear_blobs:
            rx, ry = r.center
            d = math.hypot(fx - rx, fy - ry)
            best = d if best is None else min(best, d)
        if best is not None:
            dists.append(best)
    if not dists :
        return None  
    return float(np.median(np.array(dists, dtype=np.float32)))


def pair_markers(front_blobs: List[Blob], rear_blobs: List[Blob], expected_sep_px: Optional[float], sep_tol: float, max_pair_px: float) -> List[RobotDet]:
    """
    Pair BLUE and RED blobs into robot detections.

    Two-step logic:
      1) If expected_sep_px is known we use it.
      2) Otherwise -> fallback to mutual nearest-neighbor pairing.

    # TUNE: sep_tol and max_pair_px are the main knobs to adjust pairing strictness.
    """
    if not front_blobs or not rear_blobs:
        return []
    
    # --------------------------------------------------------
    # 1) If we know expected separation, pair using distance gate
    # --------------------------------------------------------
    if expected_sep_px is not None:
        dmin = (1.0 - sep_tol) * expected_sep_px
        dmax = (1.0 + sep_tol) * expected_sep_px
        candidates = []
        for i, f in enumerate(front_blobs):
            fx, fy = f.center
            for j, r in enumerate(rear_blobs):
                rx, ry = r.center
                d = math.hypot(fx - rx, fy - ry)
                if dmin <= d <= dmax:
                    candidates.append((abs(d - expected_sep_px), i, j))
        candidates.sort(key=lambda x: x[0])
        used_f, used_r = set(), set()
        dets = []
        for _, i, j in candidates:
            if i in used_f or j in used_r:
                continue
            used_f.add(i)
            used_r.add(j)
            dets.append(compute_pose(front_blobs[i], rear_blobs[j]))
        return dets

    # --------------------------------------------------------
    # 2) Fallback: mutual nearest neighbor pairing (bring-up mode)
    # --------------------------------------------------------
    blue_to_red = {}
    for i, f in enumerate(front_blobs):
        fx, fy = f.center
        best_j, best_d = None, None
        for j, r in enumerate(rear_blobs):
            rx, ry = r.center
            d = math.hypot(fx - rx, fy - ry)
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        blue_to_red[i] = (best_j, best_d)

    red_to_blue = {}
    for j, r in enumerate(rear_blobs):
        rx, ry = r.center
        best_i, best_d = None, None
        for i, f in enumerate(front_blobs):
            fx, fy = f.center
            d = math.hypot(fx - rx, fy - ry)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        red_to_blue[j] = (best_i, best_d)

    dets = []
    used_f, used_r = set(), set()
    for i, (j, d) in blue_to_red.items():
        if j is None or d is None or d > max_pair_px:
            continue
        i2, _ = red_to_blue.get(j, (None, None))
        if i2 != i or i in used_f or j in used_r:
            continue
        used_f.add(i)
        used_r.add(j)
        dets.append(compute_pose(front_blobs[i], rear_blobs[j]))
    return dets

# ============================================================
# --------------------------- TRACKING ------------------------
# ============================================================

@dataclass
class RobotTrack:
    track_id: int
    x: float
    y: float
    theta: float
    sep_px: float
    last_seen: float
    misses: int = 0


def update_tracks(tracks: List[RobotTrack], dets: List[RobotDet], now: float, max_match_dist_px: float, max_misses: int) -> List[RobotTrack]:
    """
    Greedy nearest-neighbor data association:
      - match detections to existing tracks by (x,y) distance
      - unmatched dets create new tracks
      - unmatched tracks increment misses, removed after max_misses

    # TUNE: max_match_dist_px and max_misses
    """
    pairs = []
    for ti, tr in enumerate(tracks):
        for di, det in enumerate(dets):
            d = math.hypot(det.x - tr.x, det.y - tr.y)
            if d <= max_match_dist_px:
                pairs.append((d, ti, di))
    pairs.sort(key=lambda x: x[0])

    matched_tracks, matched_dets = set(), set()
    for _, ti, di in pairs:
        if ti in matched_tracks or di in matched_dets:
            continue
        matched_tracks.add(ti)
        matched_dets.add(di)
        det = dets[di]
        tr = tracks[ti]
        tr.x, tr.y, tr.theta, tr.sep_px = det.x, det.y, det.theta, det.sep_px
        tr.last_seen = now
        tr.misses = 0

    for ti, tr in enumerate(tracks):
        if ti not in matched_tracks:
            tr.misses += 1

    tracks = [tr for tr in tracks if tr.misses <= max_misses]
    next_id = (max([t.track_id for t in tracks]) + 1) if tracks else 0
    for di, det in enumerate(dets):
        if di in matched_dets:
            continue
        tracks.append(RobotTrack(next_id, det.x, det.y, det.theta, det.sep_px, now, 0))
        next_id += 1
    return tracks

# ============================================================
# ------------------------ OBSTACLES (MAP) --------------------
# ============================================================

def build_robot_exclusion_mask(shape: Tuple[int, int], dets: List[RobotDet], radius_px: int, body_width_px: int) -> np.ndarray:
    """Mask out robot regions so they are NOT detected as obstacles."""
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    for d in dets:
        fx, fy = d.front.center
        rx, ry = d.rear.center
        cv2.circle(m, (int(fx), int(fy)), radius_px, 255, -1)
        cv2.circle(m, (int(rx), int(ry)), radius_px, 255, -1)
        cv2.line(m, (int(rx), int(ry)), (int(fx), int(fy)), 255, thickness=body_width_px)
    return m


def detect_obstacles_floor_model(frame_bgr: np.ndarray, robot_excl_mask: Optional[np.ndarray], k_sigma: Tuple[float, float, float], min_area_px: float):
    """
    Simple obstacle detection for overhead view:
      - Convert to Lab format
      - Compute floor mean/std over pixels NOT in robot_excl_mask
      - Mark pixels far from floor in any Lab channel
      - Find connected components as obstacles

    # TUNE: k_sigma and min_area_px
    """
    h, w = frame_bgr.shape[:2]
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    sample_ok = np.ones((h, w), dtype=np.uint8) * 255
    if robot_excl_mask is not None:
        sample_ok = cv2.bitwise_and(sample_ok, cv2.bitwise_not(robot_excl_mask))
    ys, xs = np.where(sample_ok > 0)
    if len(xs) < 2000:
        ys, xs = np.where(np.ones((h, w), dtype=np.uint8) > 0)
    pixels = lab[ys, xs, :]
    mu = pixels.mean(axis=0)
    sig = pixels.std(axis=0) + 1e-6
    z = np.abs((lab - mu) / sig)
    kL, ka, kb = k_sigma
    obs_mask = ((z[:, :, 0] > kL) | (z[:, :, 1] > ka) | (z[:, :, 2] > kb)).astype(np.uint8) * 255
    if robot_excl_mask is not None:
        obs_mask = cv2.bitwise_and(obs_mask, cv2.bitwise_not(robot_excl_mask))
    obs_mask = clean_mask(obs_mask, ksize=7, open_it=1, close_it=2)
    cnts, _ = cv2.findContours(obs_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obs = []
    for c in cnts:
        area = float(cv2.contourArea(c))
        if area < min_area_px:
            continue
        x, y, ww, hh = cv2.boundingRect(c)
        obs.append({"bbox": (x, y, ww, hh), "center_px": (x + ww / 2.0, y + hh / 2.0), "area": area})
    obs.sort(key=lambda o: o["area"], reverse=True)
    return obs, obs_mask

# ============================================================
# -------------------- OCCUPANCY GRID + A* -------------------
# ============================================================

def mask_to_occupancy(mask: np.ndarray, cell_px: int, inflate_px: int) -> np.ndarray:
    """Convert obstacle mask into a coarse occupancy grid with inflation."""
    h, w = mask.shape[:2]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inflate_px + 1, 2 * inflate_px + 1))
    inflated = cv2.dilate(mask, k, iterations=1)
    gh = (h + cell_px - 1) // cell_px
    gw = (w + cell_px - 1) // cell_px
    grid = np.zeros((gh, gw), dtype=np.uint8)
    for gy in range(gh):
        y0 = gy * cell_px
        y1 = min(h, (gy + 1) * cell_px)
        for gx in range(gw):
            x0 = gx * cell_px
            x1 = min(w, (gx + 1) * cell_px)
            grid[gy, gx] = 1 if np.any(inflated[y0:y1, x0:x1] > 0) else 0
    return grid


def pix_to_cell(x: float, y: float, cell_px: int) -> Tuple[int, int]:
    return int(y // cell_px), int(x // cell_px)


def cell_to_pix(gx: int, gy: int, cell_px: int) -> Tuple[float, float]:
    return (gx + 0.5) * cell_px, (gy + 0.5) * cell_px


def astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* on an 8-connected occupancy grid. grid=0 free, 1 occupied."""
    gh, gw = grid.shape
    sy, sx = start
    gy, gx = goal
    if not (0 <= sy < gh and 0 <= sx < gw and 0 <= gy < gh and 0 <= gx < gw):
        return None
    if grid[sy, sx] or grid[gy, gx]:
        return None
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    def h_cost(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])
    open_heap = [(0.0, (sy, sx))]
    came_from = {}
    gscore = {(sy, sx): 0.0}
    while open_heap:
        _, cur = heapq.heappop(open_heap)
        if cur == (gy, gx):
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        for dy, dx in nbrs:
            ny = cur[0] + dy
            nx = cur[1] + dx
            if not (0 <= ny < gh and 0 <= nx < gw) or grid[ny, nx] == 1:
                continue
            step = math.hypot(dy, dx)
            tentative = gscore[cur] + step
            if (ny, nx) not in gscore or tentative < gscore[(ny, nx)]:
                came_from[(ny, nx)] = cur
                gscore[(ny, nx)] = tentative
                heapq.heappush(open_heap, (tentative + h_cost((ny, nx), (gy, gx)), (ny, nx)))
    return None


def follow_waypoint(my_pose: Tuple[float, float, float], waypoint_xy: Tuple[float, float], stop_dist_px: float, k_ang: float, k_lin: float, v_max: float) -> Dict:
    x, y, th = my_pose
    wx, wy = waypoint_xy
    dx = wx - x
    dy = wy - y
    distv = math.hypot(dx, dy)
    desired = math.atan2(dy, dx)
    err = angle_wrap(desired - th)
    w = k_ang * err
    v = 0.0
    if distv > stop_dist_px:
        v = clamp(k_lin * (distv - stop_dist_px), 0.0, v_max) * max(0.0, math.cos(err))
    left = v - 0.7 * w
    right = v + 0.7 * w
    m = max(1.0, abs(left), abs(right))
    return {"left": float(clamp(left / m, -1, 1)), "right": float(clamp(right / m, -1, 1)), "laser": False, "dist_px": float(distv), "heading_err_rad": float(err)}

# ============================================================
# Geometry utilities
# ============================================================

def point_line_distance(p: Tuple[float, float],
                        a: Tuple[float, float],
                        b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = p

    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 < 1e-9:
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = clamp(t, 0.0, 1.0)

    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)


def obstacle_blocks_line(my_xy: Tuple[float, float],
                         enemy_xy: Tuple[float, float],
                         obs_bbox: Tuple[int, int, int, int],
                         margin_px: float = 10.0) -> bool:
    x, y, w, h = obs_bbox
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    r = 0.5 * math.hypot(w, h) + margin_px
    d = point_line_distance((cx, cy), my_xy, enemy_xy)
    return d <= r


def count_blocking_obstacles(my_xy: Tuple[float, float],
                             enemy_xy: Tuple[float, float],
                             obstacles: List[Dict],
                             margin_px: float = 10.0) -> int:
    c = 0
    for o in obstacles:
        if obstacle_blocks_line(my_xy, enemy_xy, o["bbox"], margin_px=margin_px):
            c += 1
    return c


def nearest_obstacle_distance(my_xy: Tuple[float, float],
                              obstacles: List[Dict]) -> float:
    if not obstacles:
        return 1e9
    return min(dist(my_xy, o["center_px"]) for o in obstacles)


def best_hiding_obstacle(my_xy: Tuple[float, float],
                         enemy_xy: Tuple[float, float],
                         obstacles: List[Dict]) -> Optional[Dict]:
    best = None
    best_score = -1e9
    for o in obstacles:
        blocks = obstacle_blocks_line(my_xy, enemy_xy, o["bbox"], margin_px=12.0)
        d_me = dist(my_xy, o["center_px"])
        d_en = dist(enemy_xy, o["center_px"])

        score = 0.0
        if blocks:
            score += 200.0
        score += 0.15 * d_en
        score -= 0.35 * d_me

        if score > best_score:
            best_score = score
            best = o
    return best


def hiding_point_behind_obstacle(my_xy: Tuple[float, float],
                                 enemy_xy: Tuple[float, float],
                                 obs_bbox: Tuple[int, int, int, int],
                                 stand_off_px: float = 35.0) -> Tuple[float, float]:
    x, y, w, h = obs_bbox
    cx = x + 0.5 * w
    cy = y + 0.5 * h

    vx = cx - enemy_xy[0]
    vy = cy - enemy_xy[1]
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return (cx, cy)

    ux = vx / n
    uy = vy / n
    radius = 0.5 * math.hypot(w, h)
    tx = cx + (radius + stand_off_px) * ux
    ty = cy + (radius + stand_off_px) * uy
    return (tx, ty)
# ============================================================
# ------------------------ Fuzzy features -------------------------
# ============================================================

@dataclass
class TacticalFeatures:
    enemy_dist_px: float
    enemy_bearing_deg: float
    nearest_obs_dist_px: float
    n_blocking_obs: int
    has_cover: float
    close_danger: float
    obstacle_pressure: float
    surprise_desire: float


@dataclass
class FuzzyDecision:
    tactic: str
    speed_scale: float
    lookahead_scale: float
    prefer_alternate_path: bool
    use_hiding_obstacle: bool
    debug: Dict[str, float]


def extract_tactical_features(my_track: RobotTrack, enemy_track: RobotTrack, obstacles: List[Dict]) -> TacticalFeatures:
    my_xy = (my_track.x, my_track.y)
    enemy_xy = (enemy_track.x, enemy_track.y)
    dx = enemy_track.x - my_track.x
    dy = enemy_track.y - my_track.y
    enemy_dist_px = math.hypot(dx, dy)
    desired = math.atan2(dy, dx)
    err = angle_wrap(desired - my_track.theta)
    enemy_bearing_deg = math.degrees(err)
    nearest_obs_dist_px = nearest_obstacle_distance(my_xy, obstacles)
    n_block = count_blocking_obstacles(my_xy, enemy_xy, obstacles)
    has_cover = grade(float(n_block), 0.5, 1.5)
    close_danger = reverse_grade(enemy_dist_px, 120.0, 260.0)
    obstacle_pressure = reverse_grade(nearest_obs_dist_px, 80.0, 220.0)
    clutter = grade(float(len(obstacles)), 1.0, 4.0)
    medium_range = triangle(enemy_dist_px, 120.0, 220.0, 380.0)
    surprise_desire = min(1.0, 0.6 * clutter + 0.7 * medium_range)
    return TacticalFeatures(enemy_dist_px, enemy_bearing_deg, nearest_obs_dist_px, n_block, has_cover, close_danger, obstacle_pressure, surprise_desire)


def fuzzy_decide_offense(feat: TacticalFeatures) -> FuzzyDecision:
    near_enemy = reverse_grade(feat.enemy_dist_px, 120.0, 220.0)
    med_enemy = triangle(feat.enemy_dist_px, 120.0, 240.0, 420.0)
    far_enemy = grade(feat.enemy_dist_px, 260.0, 420.0)
    attack_direct = max(near_enemy, 0.6 * far_enemy)
    attack_alt = min(med_enemy, feat.surprise_desire)
    cautious = feat.obstacle_pressure
    aggressive = max(near_enemy, far_enemy * 0.7)
    tactic = "ATTACK_ALT" if attack_alt > attack_direct else "ATTACK_DIRECT"
    return FuzzyDecision(tactic, clamp(0.75 + 0.45 * aggressive - 0.30 * cautious, 0.45, 1.25), clamp(1.10 - 0.35 * cautious + 0.20 * far_enemy, 0.75, 1.45), tactic == "ATTACK_ALT", False, {"attack_direct": attack_direct, "attack_alt": attack_alt})


def fuzzy_decide_defense(feat: TacticalFeatures) -> FuzzyDecision:
    enemy_too_close = reverse_grade(feat.enemy_dist_px, 90.0, 180.0)
    enemy_close = reverse_grade(feat.enemy_dist_px, 150.0, 280.0)
    can_hide = feat.has_cover
    pressure = max(feat.close_danger, 0.7 * feat.obstacle_pressure)
    flee_strength = enemy_too_close
    hide_strength = min(enemy_close, can_hide)
    orbit_strength = min(can_hide, 1.0 - enemy_too_close)
    if flee_strength > max(hide_strength, orbit_strength):
        tactic = "FLEE"
    elif orbit_strength > hide_strength:
        tactic = "ORBIT_HIDE"
    else:
        tactic = "HIDE"
    return FuzzyDecision(tactic, clamp(0.45 + 0.90 * enemy_close, 0.35, 1.35), clamp(1.20 - 0.55 * pressure, 0.70, 1.25), tactic in ["HIDE", "ORBIT_HIDE"], tactic in ["HIDE", "ORBIT_HIDE"], {"flee_strength": flee_strength, "hide_strength": hide_strength, "orbit_strength": orbit_strength})

# ============================================================
# ------------------------ STRATEGIES -------------------------
# ============================================================

def pick_nearest_enemy(tracks: List[RobotTrack], my_id: Optional[int]) -> Optional[RobotTrack]:
    if my_id is None:
        return None
    me = next((t for t in tracks if t.track_id == my_id), None)
    if me is None:
        return None
    enemies = [t for t in tracks if t.track_id != my_id]
    if not enemies:
        return None
    enemies.sort(key=lambda e: math.hypot(e.x - me.x, e.y - me.y))
    return enemies[0]


def offense_strategy_fuzzy(tracks: List[RobotTrack], my_id: Optional[int], occ_grid: np.ndarray, cell_px: int, lookahead_cells: int, laser_close_px: float, laser_align_deg: float, ctrl: Dict, obstacles: List[Dict]):
    me = next((t for t in tracks if t.track_id == my_id), None)
    target = pick_nearest_enemy(tracks, my_id)
    if me is None or target is None:
        return {"left": 0.0, "right": 0.0, "laser": False}, None, None
    feat = extract_tactical_features(me, target, obstacles)
    decision = fuzzy_decide_offense(feat)
    start = pix_to_cell(me.x, me.y, cell_px)
    goal = pix_to_cell(target.x, target.y, cell_px)
    path = astar(occ_grid, start, goal)
    if not path or len(path) < 2:
        return {"left": 0.0, "right": 0.0, "laser": False}, path, decision
    dynamic_lookahead = max(1, int(round(lookahead_cells * decision.lookahead_scale)))
    idx = max(1, min(len(path) - 2, int(round(0.55 * (len(path) - 1))))) if decision.prefer_alternate_path and len(path) >= 4 else min(len(path) - 1, dynamic_lookahead)
    wy, wx = path[idx]
    waypoint = cell_to_pix(wx, wy, cell_px)
    ctrl_local = dict(ctrl)
    ctrl_local["v_max"] = ctrl["v_max"] * decision.speed_scale
    cmd = follow_waypoint((me.x, me.y, me.theta), waypoint, **ctrl_local)
    dx = target.x - me.x
    dy = target.y - me.y
    desired = math.atan2(dy, dx)
    err = angle_wrap(desired - me.theta)
    cmd["laser"] = bool(math.hypot(dx, dy) < laser_close_px and abs(math.degrees(err)) < laser_align_deg)
    return cmd, path, decision


def defense_strategy_fuzzy(tracks: List[RobotTrack], my_id: Optional[int], occ_grid: np.ndarray, cell_px: int, lookahead_cells: int, samples: int, ctrl: Dict, obstacles: List[Dict]):
    me = next((t for t in tracks if t.track_id == my_id), None)
    enemy = pick_nearest_enemy(tracks, my_id)
    if me is None or enemy is None:
        return {"left": 0.0, "right": 0.0, "laser": False}, None, None, None
    feat = extract_tactical_features(me, enemy, obstacles)
    decision = fuzzy_decide_defense(feat)
    ctrl_local = dict(ctrl)
    ctrl_local["v_max"] = ctrl["v_max"] * decision.speed_scale
    dynamic_lookahead = max(1, int(round(lookahead_cells * decision.lookahead_scale)))
    hiding_goal_xy = None
    path = None
    if decision.use_hiding_obstacle:
        obs = best_hiding_obstacle((me.x, me.y), (enemy.x, enemy.y), obstacles)
        if obs is not None:
            hiding_goal_xy = hiding_point_behind_obstacle((enemy.x, enemy.y), obs["bbox"], stand_off_px=35.0)
            path = astar(occ_grid, pix_to_cell(me.x, me.y, cell_px), pix_to_cell(hiding_goal_xy[0], hiding_goal_xy[1], cell_px))
    if path is None or len(path) < 2 or decision.tactic == "FLEE":
        gh, gw = occ_grid.shape
        rng = np.random.default_rng(0)
        best_score = -1e9
        start = pix_to_cell(me.x, me.y, cell_px)
        for _ in range(samples):
            gy = int(rng.integers(0, gh)); gx = int(rng.integers(0, gw))
            if occ_grid[gy, gx] == 1:
                continue
            px, py = cell_to_pix(gx, gy, cell_px)
            score = math.hypot(px - enemy.x, py - enemy.y) - 0.15 * math.hypot(px - me.x, py - me.y)
            if decision.tactic in ["HIDE", "ORBIT_HIDE"]:
                score += count_blocking_obstacles((px, py), (enemy.x, enemy.y), obstacles) * 80.0
            cand = astar(occ_grid, start, (gy, gx))
            if cand is not None and score > best_score:
                best_score = score
                path = cand
        if path is None:
            return {"left": 0.0, "right": 0.0, "laser": False}, None, decision, hiding_goal_xy
    idx = min(len(path) - 1, dynamic_lookahead)
    wy, wx = path[idx]
    waypoint = cell_to_pix(wx, wy, cell_px)
    cmd = follow_waypoint((me.x, me.y, me.theta), waypoint, **ctrl_local)
    cmd["laser"] = False
    return cmd, path, decision, hiding_goal_xy

# ============================================================
# ------------------------- ID DANCE --------------------------
# ============================================================

@dataclass
class DanceSegment:
    duration_s: float
    left: float
    right: float
    laser: bool
    expected_omega_sign: int


def default_id_dance() -> List[DanceSegment]:
    return [DanceSegment(0.30, +0.45, -0.45, False, +1), DanceSegment(0.30, -0.45, +0.45, False, -1), DanceSegment(0.30, +0.45, -0.45, False, +1), DanceSegment(0.30, 0.00, 0.00, False, 0)]


class DanceIdentifier:
    def __init__(self, dance: List[DanceSegment], omega_clip: float = 8.0):
        self.dance = dance
        self.omega_clip = float(omega_clip)
        self.reset()

    def reset(self):
        self.segment_idx = 0
        self.segment_t0 = None
        self.prev_theta = {}
        self.prev_time = {}
        self.score = {}
        self.energy = {}

    def start(self, now: float):
        self.reset(); self.segment_t0 = now; self.segment_idx = 0

    def is_active(self) -> bool:
        return self.segment_t0 is not None and self.segment_idx < len(self.dance)

    def current_segment(self, now: float) -> Optional[DanceSegment]:
        if not self.is_active():
            return None
        seg = self.dance[self.segment_idx]
        if now - self.segment_t0 >= seg.duration_s:
            self.segment_idx += 1
            self.segment_t0 = now
            if self.segment_idx >= len(self.dance):
                return None
            seg = self.dance[self.segment_idx]
        return seg

    def update_scores(self, tracks: List[RobotTrack], now: float, expected_sign: int):
        for tr in tracks:
            tid = tr.track_id
            if tid not in self.prev_theta:
                self.prev_theta[tid] = tr.theta; self.prev_time[tid] = now; self.score.setdefault(tid, 0.0); self.energy.setdefault(tid, 0.0); continue
            dt = now - self.prev_time[tid]
            if dt <= 1e-6:
                continue
            dth = angle_wrap(tr.theta - self.prev_theta[tid])
            omega = clamp(dth / dt, -self.omega_clip, self.omega_clip)
            if expected_sign == 0:
                self.score[tid] += -abs(omega)
            else:
                self.score[tid] += expected_sign * omega; self.energy[tid] += abs(omega)
            self.prev_theta[tid] = tr.theta; self.prev_time[tid] = now

    def pick_best(self, min_energy: float) -> Optional[int]:
        if not self.score:
            return None
        candidates = [(tid, sc) for tid, sc in self.score.items() if self.energy.get(tid, 0.0) >= min_energy]
        if not candidates:
            candidates = list(self.score.items())
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]




# ============================================================
# -------------------- DIAGNOSTICS / PLOTS -------------------
# ============================================================

TACTIC_MAP = {
    "NONE": 0,
    "ATTACK_DIRECT": 1,
    "ATTACK_ALT": 2,
    "HIDE": 3,
    "ORBIT_HIDE": 4,
    "FLEE": 5,
}

PHASE_MAP = {
    "IDENTIFY": 0,
    "RUN": 1,
}


def estimate_signed_speeds(left_cmd: float, right_cmd: float):
    """
    Very rough speed proxies from normalized left/right commands.
    These are not physical calibrations, but they are still very useful for analysis plots.
    """
    v_lin = 0.5 * (left_cmd + right_cmd)
    v_ang = 0.5 * (right_cmd - left_cmd)
    return v_lin, v_ang


def init_run_log():
    """Initialize all logged channels."""
    return {
        "t": [],
        "phase": [],
        "phase_id": [],
        "pwm_left": [],
        "pwm_right": [],
        "laser": [],
        "v_lin_proxy": [],
        "v_ang_proxy": [],
        "enemy_dist_px": [],
        "enemy_bearing_deg": [],
        "nearest_obs_dist_px": [],
        "blocking_obs_count": [],
        "num_obstacles": [],
        "tactic": [],
        "tactic_id": [],
        "speed_scale": [],
        "lookahead_scale": [],
        "reaction_time_ms": [],
        "loop_dt_ms": [],
        "fps": [],
        "detect_ms": [],
        "obstacles_ms": [],
        "astar_ms": [],
        "control_ms": [],
        "total_ms": [],
        "my_x": [],
        "my_y": [],
        "enemy_x": [],
        "enemy_y": [],
    }


def append_run_log(log, **kwargs):
    """Append one sample to the log dictionary."""
    for k in log.keys():
        log[k].append(kwargs.get(k, np.nan))


def _nan_array(values):
    out = []
    for v in values:
        if v is None:
            out.append(np.nan)
        else:
            out.append(v)
    return np.array(out, dtype=float)


def save_run_csv(log, output_prefix="run_diagnostics"):
    """Save raw log to CSV"""
    out_dir = Path.cwd()
    csv_path = out_dir / f"{output_prefix}.csv"
    keys = list(log.keys())

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        n = len(log[keys[0]])
        for i in range(n):
            writer.writerow([log[k][i] for k in keys])

    return csv_path


def save_run_diagnostics(log, output_prefix="run_diagnostics"):
    """
    Save CSV + a set of figures after the OpenCV session is over
    Returns a list of generated file paths
    """
    out_dir = Path.cwd()
    generated = []

    csv_path = save_run_csv(log, output_prefix=output_prefix)
    generated.append(csv_path)

    t = _nan_array(log["t"])
    if len(t) == 0:
        print("No diagnostics data to save.")
        return generated

    t = t - t[0]

    # 1) PWM / command history
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, _nan_array(log["pwm_left"]), label="PWM Left")
    plt.plot(t, _nan_array(log["pwm_right"]), label="PWM Right")
    plt.plot(t, _nan_array(log["laser"]), label="Laser", linestyle="--")
    plt.xlabel("Time [s]")
    plt.ylabel("Normalized command")
    plt.title("PWM / Command History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p = out_dir / f"{output_prefix}_pwm.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    generated.append(p)
    plt.close(fig)

    # 2) Speed proxies
    fig = plt.figure(figsize=(10, 6))
    plt.plot(t, _nan_array(log["v_lin_proxy"]), label="Linear speed proxy")
    plt.plot(t, _nan_array(log["v_ang_proxy"]), label="Angular speed proxy")
    plt.xlabel("Time [s]")
    plt.ylabel("Proxy speed")
    plt.title("Robot Speed Proxies from PWM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    p = out_dir / f"{output_prefix}_speed_proxy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    generated.append(p)
    plt.close(fig)

    # 3) Combat geometry
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, _nan_array(log["enemy_dist_px"]), label="Enemy distance")
    ax1.plot(t, _nan_array(log["nearest_obs_dist_px"]), label="Nearest obstacle distance")
    ax1.set_ylabel("Distance [px]")
    ax1.set_title("Combat Geometry vs Time")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, _nan_array(log["enemy_bearing_deg"]), label="Enemy bearing")
    ax2.set_ylabel("Bearing [deg]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, _nan_array(log["blocking_obs_count"]), label="Blocking obstacles")
    ax3.plot(t, _nan_array(log["num_obstacles"]), label="Total obstacles")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Count")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    p = out_dir / f"{output_prefix}_geometry.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    generated.append(p)
    plt.close(fig)

    # 4) Strategy timeline
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.step(t, _nan_array(log["tactic_id"]), where="post")
    ax1.set_ylabel("Tactic ID")
    ax1.set_title("Strategy / Tactical Decisions")
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, _nan_array(log["speed_scale"]), label="Speed scale")
    ax2.plot(t, _nan_array(log["lookahead_scale"]), label="Lookahead scale")
    ax2.set_ylabel("Scale")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.step(t, _nan_array(log["phase_id"]), where="post", label="Phase")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Phase ID")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    p = out_dir / f"{output_prefix}_strategy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    generated.append(p)
    plt.close(fig)

    # 5) Performance
    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(t, _nan_array(log["fps"]), label="FPS")
    ax1.set_ylabel("FPS")
    ax1.set_title("Runtime Performance")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(t, _nan_array(log["detect_ms"]), label="Detect ms")
    ax2.plot(t, _nan_array(log["obstacles_ms"]), label="Obstacles ms")
    ax2.plot(t, _nan_array(log["astar_ms"]), label="A* ms")
    ax2.plot(t, _nan_array(log["control_ms"]), label="Control ms")
    ax2.plot(t, _nan_array(log["total_ms"]), label="Total loop ms")
    ax2.set_ylabel("Time [ms]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(t, _nan_array(log["reaction_time_ms"]), label="Reaction time ms")
    ax3.plot(t, _nan_array(log["loop_dt_ms"]), label="Loop dt ms")
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Time [ms]")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    p = out_dir / f"{output_prefix}_performance.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    generated.append(p)
    plt.close(fig)

    # 6) XY trajectories
    fig = plt.figure(figsize=(8, 8))
    plt.plot(_nan_array(log["my_x"]), _nan_array(log["my_y"]), label="My robot trajectory")
    plt.plot(_nan_array(log["enemy_x"]), _nan_array(log["enemy_y"]), label="Enemy trajectory")
    plt.xlabel("X [px]")
    plt.ylabel("Y [px]")
    plt.title("XY Trajectories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_yaxis()
    p = out_dir / f"{output_prefix}_xy.png"
    fig.tight_layout()
    fig.savefig(p, dpi=180)
    generated.append(p)
    plt.close(fig)

    print("Saved diagnostics:")
    for item in generated:
        print(" -", item)

    return generated

# ============================================================
# ---------------------------- MAIN ---------------------------
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--mode", type=str, default="offense", choices=["offense", "defense"])
    ap.add_argument("--log_period", type=float, default=2.0)

    # ---------------- # TUNE: MARKER HSV ----------------
    ap.add_argument("--blue_h_lo", type=int, default=95)
    ap.add_argument("--blue_h_hi", type=int, default=140)
    ap.add_argument("--blue_s_min", type=int, default=30)
    ap.add_argument("--blue_v_min", type=int, default=20)
    ap.add_argument("--red_h1_lo", type=int, default=0)
    ap.add_argument("--red_h1_hi", type=int, default=10)
    ap.add_argument("--red_h2_lo", type=int, default=172)
    ap.add_argument("--red_h2_hi", type=int, default=179)
    ap.add_argument("--red_s_min", type=int, default=90)
    ap.add_argument("--red_v_min", type=int, default=40)
    ap.add_argument("--blob_min_area", type=float, default=150.0)

    # ---------------- # TUNE: PAIRING ----------------
    ap.add_argument("--marker_sep_px", type=float, default=-1.0)
    ap.add_argument("--pair_sep_tol", type=float, default=0.55)
    ap.add_argument("--max_pair_px", type=float, default=1200.0)
    # ---------------- # TUNE: SCALE ----------------
    ap.add_argument("--marker_sep_in", type=float, default=9.0)
    ap.add_argument("--robot_radius_in", type=float, default=4.0)
    ap.add_argument("--scale_alpha", type=float, default=0.2)

    # ---------------- # TUNE: OBSTACLES ----------------
    ap.add_argument("--k_sigma", type=float, nargs=3, default=(3.0, 3.0, 3.0))
    ap.add_argument("--obs_min_area", type=float, default=1800.0)
    ap.add_argument("--robot_excl_radius_px", type=int, default=30)
    ap.add_argument("--robot_body_width_px", type=int, default=45)
    # ---------------- # TUNE: GRID ----------------
    ap.add_argument("--cell_px", type=int, default=10)
    ap.add_argument("--inflate_px", type=int, default=25)

    # ---------------- # TUNE: CONTROLLER ----------------
    ap.add_argument("--ctrl_stop_px", type=float, default=18.0)
    ap.add_argument("--ctrl_k_ang", type=float, default=2.2)
    ap.add_argument("--ctrl_k_lin", type=float, default=0.010)
    ap.add_argument("--ctrl_v_max", type=float, default=0.7)

    # ---------------- # TUNE: STRATEGY ----------------
    ap.add_argument("--lookahead_cells", type=int, default=6)
    ap.add_argument("--laser_close_px", type=float, default=120.0)
    ap.add_argument("--laser_align_deg", type=float, default=12.0)
    ap.add_argument("--def_samples", type=int, default=80)

    # ---------------- Serial ----------------
    ap.add_argument("--serial_port", type=str, default=None)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--dry_run", action="store_true")

    # ---------------- ID dance ----------------
    ap.add_argument("--id_dance", action="store_true")
    ap.add_argument("--id_attempts", type=int, default=2)
    ap.add_argument("--id_min_energy", type=float, default=1.5)

    # ---------------- # DEBUG ----------------
    ap.add_argument("--show_masks", action="store_true")
    ap.add_argument("--show_obs_mask", action="store_true")
    ap.add_argument("--draw_path", action="store_true")

    # ---------------- Plots ----------------
    ap.add_argument("--show_fuzzy", action="store_true")
    ap.add_argument("--save_plots", action="store_true", help="Save CSV and figures after closing OpenCV windows")
    ap.add_argument("--plot_prefix", type=str, default="run_diagnostics")
    return ap.parse_args()



def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.cam)
    #cap = cv2.VideoCapture("http://10.207.120.184:8080/video") #use phone "IP webcam application" (change IP)

    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    commander = RobotCommander(port=args.serial_port, baud=args.baud, dry_run=args.dry_run)
    tracks = []
    my_id = None
    expected_sep_px = None if args.marker_sep_px <= 0 else float(args.marker_sep_px)
    scale_ema = EMA(alpha=args.scale_alpha)
    px_per_in = None
    identifier = DanceIdentifier(default_id_dance())
    phase = "IDENTIFY" if args.id_dance else "RUN"
    attempts_left = int(args.id_attempts)
    last_log = time.time()
    blue_ranges = [HSVRange(args.blue_h_lo, args.blue_h_hi)]
    red_ranges = [HSVRange(args.red_h1_lo, args.red_h1_hi), HSVRange(args.red_h2_lo, args.red_h2_hi)]
    ctrl = dict(stop_dist_px=args.ctrl_stop_px, k_ang=args.ctrl_k_ang, k_lin=args.ctrl_k_lin, v_max=args.ctrl_v_max)

    # Diagnostics state
    run_log = init_run_log()
    last_loop_time = None

    try:
        while True:
            loop_t0 = time.perf_counter()
            now = time.time()
            loop_dt_ms = np.nan if last_loop_time is None else (now - last_loop_time) * 1000.0
            fps = np.nan if last_loop_time is None or (now - last_loop_time) <= 1e-9 else 1.0 / (now - last_loop_time)
            last_loop_time = now

            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            # Detection / pairing timing
            t_detect0 = time.perf_counter()
            # 1) HSV segmentation
            blue_mask = mask_hsv(frame, blue_ranges, s_min=args.blue_s_min, v_min=args.blue_v_min)
            red_mask = mask_hsv(frame, red_ranges, s_min=args.red_s_min, v_min=args.red_v_min)
            # 2) Blobs
            front_blobs = extract_blobs(blue_mask, args.blob_min_area)
            rear_blobs = extract_blobs(red_mask, args.blob_min_area)
            # 3) Estimate marker separation (if not provided)
            if expected_sep_px is None:
                expected_sep_px = estimate_marker_sep_px(front_blobs, rear_blobs)
            # 4) Pair blobs into robots
            dets = pair_markers(front_blobs, rear_blobs, expected_sep_px, args.pair_sep_tol, args.max_pair_px)
            detect_ms = (time.perf_counter() - t_detect0) * 1000.0

            # 5) Scale estimate (px/in) + dynamic inflation
            inflate_px_dynamic = args.inflate_px
            if dets:
                sep_px_med = float(np.median([d.sep_px for d in dets]))
                px_per_in_meas = sep_px_med / args.marker_sep_in
                px_per_in = scale_ema.update(px_per_in_meas)
                inflate_px_dynamic = int(round(args.robot_radius_in * px_per_in))
            
            # 6) Track robots
            tracks = update_tracks(tracks, dets, now, 90.0, 12)

            # 7) Obstacles
            t_obs0 = time.perf_counter()
            robot_excl = build_robot_exclusion_mask((h, w), dets, args.robot_excl_radius_px, args.robot_body_width_px)
            obstacles, obs_mask = detect_obstacles_floor_model(frame, robot_excl, tuple(args.k_sigma), args.obs_min_area)
            # 8) Grid + planning
            occ_grid = mask_to_occupancy(obs_mask, args.cell_px, inflate_px_dynamic)
            obstacles_ms = (time.perf_counter() - t_obs0) * 1000.0

            # 9) Phase machine
            cmd = {"left": 0.0, "right": 0.0, "laser": False}
            path = None
            fuzzy_decision = None
            hiding_goal_xy = None
            astar_ms = 0.0

            if phase == "IDENTIFY":
                if not tracks:
                    commander.send(0.0, 0.0, False)
                else:
                    if not identifier.is_active():
                        if attempts_left <= 0:
                            phase = "RUN"
                        else:
                            attempts_left -= 1
                            identifier.start(now)
                    seg = identifier.current_segment(now)
                    if seg is not None:
                        commander.send(seg.left, seg.right, seg.laser)
                        identifier.update_scores(tracks, now, seg.expected_omega_sign)
                        cmd = {"left": seg.left, "right": seg.right, "laser": seg.laser}
                    else:
                        my_id = identifier.pick_best(args.id_min_energy)
                        commander.send(0.0, 0.0, False)
                        phase = "RUN"

            # Strategy / command timing
            t_ctrl0 = time.perf_counter()
            if phase == "RUN":
                if my_id is None:
                    cmd = {"left": 0.0, "right": 0.0, "laser": False}
                else:
                    t_astar0 = time.perf_counter()
                    if args.mode == "offense":
                        cmd, path, fuzzy_decision = offense_strategy_fuzzy(
                            tracks, my_id, occ_grid, args.cell_px, args.lookahead_cells,
                            args.laser_close_px, args.laser_align_deg, ctrl, obstacles
                        )
                    else:
                        cmd, path, fuzzy_decision, hiding_goal_xy = defense_strategy_fuzzy(
                            tracks, my_id, occ_grid, args.cell_px, args.lookahead_cells,
                            args.def_samples, ctrl, obstacles
                        )
                    astar_ms = (time.perf_counter() - t_astar0) * 1000.0
                commander.send(cmd["left"], cmd["right"], cmd["laser"])
            control_ms = (time.perf_counter() - t_ctrl0) * 1000.0

            # Combat geometry for logs
            my_track = next((t for t in tracks if t.track_id == my_id), None)
            enemy_track = pick_nearest_enemy(tracks, my_id)
            feat = None
            if my_track is not None and enemy_track is not None:
                feat = extract_tactical_features(my_track, enemy_track, obstacles)

            # Approximate reaction time = frame-to-command latency
            reaction_time_ms = (time.perf_counter() - loop_t0) * 1000.0
            total_ms = reaction_time_ms

            #Logging
            if now - last_log >= args.log_period:
                last_log = now
                line = f"[LOG] phase={phase} robots={len(tracks)} blue={len(front_blobs)} red={len(rear_blobs)} obs={len(obstacles)} my_id={my_id} sep_px={expected_sep_px} scale_px_per_in={px_per_in}"
                if fuzzy_decision is not None:
                    line += f" tactic={fuzzy_decision.tactic} speed_scale={fuzzy_decision.speed_scale:.2f}"
                print(line)

            # Append diagnostics sample
            tactic = fuzzy_decision.tactic if fuzzy_decision is not None else "NONE"
            speed_scale = fuzzy_decision.speed_scale if fuzzy_decision is not None else np.nan
            lookahead_scale = fuzzy_decision.lookahead_scale if fuzzy_decision is not None else np.nan
            v_lin_proxy, v_ang_proxy = estimate_signed_speeds(cmd["left"], cmd["right"])

            append_run_log(
                run_log,
                t=now,
                phase=phase,
                phase_id=PHASE_MAP.get(phase, np.nan),
                pwm_left=cmd["left"],
                pwm_right=cmd["right"],
                laser=float(cmd["laser"]),
                v_lin_proxy=v_lin_proxy,
                v_ang_proxy=v_ang_proxy,
                enemy_dist_px=(feat.enemy_dist_px if feat is not None else np.nan),
                enemy_bearing_deg=(feat.enemy_bearing_deg if feat is not None else np.nan),
                nearest_obs_dist_px=(feat.nearest_obs_dist_px if feat is not None else np.nan),
                blocking_obs_count=(feat.n_blocking_obs if feat is not None else np.nan),
                num_obstacles=len(obstacles),
                tactic=tactic,
                tactic_id=TACTIC_MAP.get(tactic, np.nan),
                speed_scale=speed_scale,
                lookahead_scale=lookahead_scale,
                reaction_time_ms=reaction_time_ms,
                loop_dt_ms=loop_dt_ms,
                fps=fps,
                detect_ms=detect_ms,
                obstacles_ms=obstacles_ms,
                astar_ms=astar_ms,
                control_ms=control_ms,
                total_ms=total_ms,
                my_x=(my_track.x if my_track is not None else np.nan),
                my_y=(my_track.y if my_track is not None else np.nan),
                enemy_x=(enemy_track.x if enemy_track is not None else np.nan),
                enemy_y=(enemy_track.y if enemy_track is not None else np.nan),
            )

            # Visualization
            out = frame.copy()
            for d in dets:
                fx, fy = d.front.center
                rx, ry = d.rear.center
                cv2.circle(out, (int(fx), int(fy)), 7, (0, 255, 0), 2)
                cv2.circle(out, (int(rx), int(ry)), 7, (0, 255, 0), 2)
                cv2.line(out, (int(rx), int(ry)), (int(fx), int(fy)), (0, 255, 0), 2)
            for t in tracks:
                color = (0, 255, 255) if (my_id is not None and t.track_id == my_id) else (255, 255, 0)
                cv2.circle(out, (int(t.x), int(t.y)), 10, color, 2)
                x2 = int(t.x + 35 * math.cos(t.theta))
                y2 = int(t.y + 35 * math.sin(t.theta))
                cv2.arrowedLine(out, (int(t.x), int(t.y)), (x2, y2), color, 2)
                cv2.putText(out, f"id={t.track_id}", (int(t.x + 12), int(t.y - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            for o in obstacles[:12]:
                x, y, ww, hh = o["bbox"]
                cv2.rectangle(out, (x, y), (x + ww, y + hh), (0, 255, 255), 2)
            if args.draw_path and path:
                for (cy, cx) in path[::2]:
                    px, py = cell_to_pix(cx, cy, args.cell_px)
                    cv2.circle(out, (int(px), int(py)), 2, (255, 0, 255), -1)
            if hiding_goal_xy is not None:
                cv2.circle(out, (int(hiding_goal_xy[0]), int(hiding_goal_xy[1])), 6, (0, 128, 255), -1)
                cv2.putText(out, "hide_goal", (int(hiding_goal_xy[0] + 8), int(hiding_goal_xy[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 128, 255), 1)

            header = f"phase={phase} mode={args.mode} L={cmd['left']:+.2f} R={cmd['right']:+.2f} laser={cmd['laser']} my_id={my_id}"
            cv2.putText(out, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
            cv2.putText(out, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 1)
            sep_txt = "marker_sep_px: estimating..." if expected_sep_px is None else f"marker_sep_px~{expected_sep_px:.1f}"
            cv2.putText(out, sep_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2)
            if px_per_in is not None:
                sc_txt = f"scale~{px_per_in:.1f} px/in inflate={inflate_px_dynamic}px"
                cv2.putText(out, sc_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2)
            if args.show_fuzzy and fuzzy_decision is not None:
                ftxt = f"tactic={fuzzy_decision.tactic} speedx={fuzzy_decision.speed_scale:.2f} lookaheadx={fuzzy_decision.lookahead_scale:.2f}"
                cv2.putText(out, ftxt, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

            cv2.imshow("MECH663 Overhead + Fuzzy", out)
            if args.show_obs_mask:
                cv2.imshow("obs_mask", obs_mask)
            if args.show_masks:
                cv2.imshow("blue_mask", blue_mask)
                cv2.imshow("red_mask", red_mask)

            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord("q")):
                break

    finally:
        cap.release()
        commander.close()
        cv2.destroyAllWindows()

    if args.save_plots:
        save_run_diagnostics(run_log, output_prefix=args.plot_prefix)



if __name__ == "__main__":
    main()
