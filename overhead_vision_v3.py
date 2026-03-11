"""
# TUNE: Search for "# TUNE:" markers for parameters to adjust.
# DEBUG: Search for "# DEBUG:" markers for debug windows / prints.
"""

import cv2
import numpy as np
import math
import time
import argparse
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

try:
    import serial  # pip install pyserial
except Exception:
    serial = None


# ============================================================
# --------------------------- UTILS ---------------------------
# ============================================================

def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


def angle_wrap(a: float) -> float:
    """Wrap angle (rad) to [-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a < -math.pi:
        a += 2 * math.pi
    return a


def clean_mask(mask: np.ndarray, ksize: int = 5, open_it: int = 1, close_it: int = 1) -> np.ndarray:
    """
    Binary mask cleanup
    """
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=open_it)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=close_it)
    return mask


class EMA:
    """Tiny exponential moving average (EMA) helper."""
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
# ---------------------- HSV SEGMENTATION --------------------
# ============================================================

@dataclass
class HSVRange:
    """Hue range in OpenCV HSV (H in [0..179], S,V in [0..255])."""
    h_lo: int
    h_hi: int


def mask_hsv(frame_bgr: np.ndarray,
             hue_ranges: List[HSVRange],
             s_min: int,
             v_min: int,
             morph_ksize: int = 5,
             morph_close_it: int = 2) -> np.ndarray:
    """
    Create a binary mask for pixels within hue_ranges AND with S >= s_min AND V >= v_min.
    """
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
    mask = clean_mask(mask, ksize=morph_ksize, open_it=1, close_it=morph_close_it)
    return mask


# ============================================================
# ---------------------- ROBOT COMMANDER ---------------------
# ============================================================

class RobotCommander:
    """
    Send left/right/laser commands to the robot.

    # TUNE: adapt this format to the Arduino expectation and future improvement.
    """
    def __init__(self, port: Optional[str], baud: int, dry_run: bool, print_rate_hz: float = 5.0):
        self.dry_run = dry_run or (port is None) or (serial is None)
        self.ser = None
        self._last_print = 0.0
        self._print_period = 1.0 / max(1e-3, print_rate_hz)

        if not self.dry_run:
            self.ser = serial.Serial(port, baud, timeout=0)
            time.sleep(2.0)

    def send(self, left: float, right: float, laser: bool):
        msg = f"L={left:+.2f} R={right:+.2f} LAS={1 if laser else 0}\\n"
        if self.ser is not None:
            self.ser.write(msg.encode("ascii"))
        else:
            # Dry-run: print commands at a limited rate (avoid terminal spam)
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
    center: Tuple[float, float]   # (x,y) pixels
    area: float                   # contour area (px^2)
    contour: np.ndarray           # raw contour


def extract_blobs(mask: np.ndarray, min_area_px: float) -> List[Blob]:
    """
    Extract connected components (contours) from a binary mask.

    # TUNE: min_area_px allows to reject noise.
    """
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

    # Large first (useful for debug)
    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs


@dataclass
class RobotDet:
    """One robot detection = (front BLUE blob, rear RED blob)"""
    front: Blob
    rear: Blob
    x: float
    y: float
    theta: float     # heading = rear -> front (rad)
    sep_px: float    # marker separation in pixels


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

    if not dists:
        return None

    return float(np.median(np.array(dists, dtype=np.float32)))


def pair_markers(front_blobs: List[Blob],
                 rear_blobs: List[Blob],
                 expected_sep_px: Optional[float],
                 sep_tol: float,
                 max_pair_px: float) -> List[RobotDet]:
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
                    # smaller cost is better
                    candidates.append((abs(d - expected_sep_px), i, j))

        candidates.sort(key=lambda x: x[0])

        used_f, used_r = set(), set()
        dets: List[RobotDet] = []
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

    dets: List[RobotDet] = []
    used_f, used_r = set(), set()
    for i, (j, d) in blue_to_red.items():
        if j is None or d is None or d > max_pair_px:
            continue
        i2, _ = red_to_blue.get(j, (None, None))
        if i2 != i:
            continue
        if i in used_f or j in used_r:
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
    """A tracked robot hypothesis with a stable ID."""
    track_id: int
    x: float
    y: float
    theta: float
    sep_px: float
    last_seen: float
    misses: int = 0


def update_tracks(tracks: List[RobotTrack],
                  dets: List[RobotDet],
                  now: float,
                  max_match_dist_px: float,
                  max_misses: int) -> List[RobotTrack]:
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

    matched_tracks = set()
    matched_dets = set()

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
        tracks.append(RobotTrack(
            track_id=next_id,
            x=det.x, y=det.y, theta=det.theta,
            sep_px=det.sep_px,
            last_seen=now,
            misses=0
        ))
        next_id += 1

    return tracks


# ============================================================
# ------------------------ OBSTACLES (MAP) --------------------
# ============================================================

def build_robot_exclusion_mask(shape: Tuple[int, int],
                               dets: List[RobotDet],
                               radius_px: int,
                               body_width_px: int) -> np.ndarray:
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


def detect_obstacles_floor_model(frame_bgr: np.ndarray,
                                robot_excl_mask: Optional[np.ndarray],
                                k_sigma: Tuple[float, float, float],
                                min_area_px: float) -> Tuple[List[Dict], np.ndarray]:
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

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

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
            if not (0 <= ny < gh and 0 <= nx < gw):
                continue
            if grid[ny, nx] == 1:
                continue

            step = math.hypot(dy, dx)
            tentative = gscore[cur] + step
            if (ny, nx) not in gscore or tentative < gscore[(ny, nx)]:
                came_from[(ny, nx)] = cur
                gscore[(ny, nx)] = tentative
                f = tentative + h_cost((ny, nx), (gy, gx))
                heapq.heappush(open_heap, (f, (ny, nx)))

    return None


# ============================================================
# ------------------------ STRATEGIES -------------------------
# ============================================================

def pick_nearest_enemy(tracks: List[RobotTrack], my_id: Optional[int]) -> Optional[RobotTrack]:
    """Pick the closest track that is NOT my_id."""
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


def follow_waypoint(my_pose: Tuple[float, float, float],
                    waypoint_xy: Tuple[float, float],
                    stop_dist_px: float,
                    k_ang: float,
                    k_lin: float,
                    v_max: float) -> Dict:
    """Waypoint follower -> differential drive command."""
    x, y, th = my_pose
    wx, wy = waypoint_xy

    dx = wx - x
    dy = wy - y
    dist = math.hypot(dx, dy)

    desired = math.atan2(dy, dx)
    err = angle_wrap(desired - th)

    w = k_ang * err
    v = 0.0
    if dist > stop_dist_px:
        v = clamp(k_lin * (dist - stop_dist_px), 0.0, v_max) * max(0.0, math.cos(err))

    left = v - 0.7 * w
    right = v + 0.7 * w

    m = max(1.0, abs(left), abs(right))
    left /= m
    right /= m

    return {
        "left": float(clamp(left, -1, 1)),
        "right": float(clamp(right, -1, 1)),
        "laser": False,
        "dist_px": float(dist),
        "heading_err_rad": float(err),
    }


def offense_strategy(tracks: List[RobotTrack],
                    my_id: Optional[int],
                    occ_grid: np.ndarray,
                    cell_px: int,
                    lookahead_cells: int,
                    laser_close_px: float,
                    laser_align_deg: float,
                    ctrl: Dict) -> Tuple[Dict, Optional[List[Tuple[int, int]]]]:
    """Offense: A* to nearest enemy + laser gate."""
    me = next((t for t in tracks if t.track_id == my_id), None)
    target = pick_nearest_enemy(tracks, my_id)
    if me is None or target is None:
        return {"left": 0.0, "right": 0.0, "laser": False}, None

    start = pix_to_cell(me.x, me.y, cell_px)
    goal = pix_to_cell(target.x, target.y, cell_px)
    path = astar(occ_grid, start, goal)

    if not path or len(path) < 2:
        return {"left": 0.0, "right": 0.0, "laser": False}, path

    idx = min(len(path) - 1, lookahead_cells)
    wy, wx = path[idx]
    waypoint = cell_to_pix(wx, wy, cell_px)

    cmd = follow_waypoint((me.x, me.y, me.theta), waypoint, **ctrl)

    dx = target.x - me.x
    dy = target.y - me.y
    desired = math.atan2(dy, dx)
    err = angle_wrap(desired - me.theta)

    close = math.hypot(dx, dy) < laser_close_px
    aligned = abs(math.degrees(err)) < laser_align_deg
    cmd["laser"] = bool(close and aligned)

    return cmd, path


def defense_strategy(tracks: List[RobotTrack],
                    my_id: Optional[int],
                    occ_grid: np.ndarray,
                    cell_px: int,
                    lookahead_cells: int,
                    samples: int,
                    ctrl: Dict) -> Tuple[Dict, Optional[List[Tuple[int, int]]]]:
    """Defense: pick a far free cell (sampling) + A*."""
    me = next((t for t in tracks if t.track_id == my_id), None)
    enemy = pick_nearest_enemy(tracks, my_id)
    if me is None or enemy is None:
        return {"left": 0.0, "right": 0.0, "laser": False}, None

    gh, gw = occ_grid.shape
    rng = np.random.default_rng(0)

    best_goal = None
    best_score = -1e9
    start = pix_to_cell(me.x, me.y, cell_px)

    for _ in range(samples):
        gy = int(rng.integers(0, gh))
        gx = int(rng.integers(0, gw))
        if occ_grid[gy, gx] == 1:
            continue

        px, py = cell_to_pix(gx, gy, cell_px)
        dE = math.hypot(px - enemy.x, py - enemy.y)
        dM = math.hypot(px - me.x, py - me.y)
        score = dE - 0.15 * dM

        if score > best_score:
            path = astar(occ_grid, start, (gy, gx))
            if path is None:
                continue
            best_score = score
            best_goal = (gy, gx)

    if best_goal is None:
        return {"left": 0.0, "right": 0.0, "laser": False}, None

    path = astar(occ_grid, start, best_goal)
    if not path or len(path) < 2:
        return {"left": 0.0, "right": 0.0, "laser": False}, path

    idx = min(len(path) - 1, lookahead_cells)
    wy, wx = path[idx]
    waypoint = cell_to_pix(wx, wy, cell_px)

    cmd = follow_waypoint((me.x, me.y, me.theta), waypoint, **ctrl)
    cmd["laser"] = False
    return cmd, path


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
    """Default yaw signature used to identify our robot."""
    return [
        DanceSegment(0.30, +0.45, -0.45, False, +1),
        DanceSegment(0.30, -0.45, +0.45, False, -1),
        DanceSegment(0.30, +0.45, -0.45, False, +1),
        DanceSegment(0.30,  0.00,  0.00, False,  0),
    ]


class DanceIdentifier:
    """Score tracks based on the sign of their observed angular velocity (dtheta/dt)."""
    def __init__(self, dance: List[DanceSegment], omega_clip: float = 8.0):
        self.dance = dance
        self.omega_clip = float(omega_clip)
        self.reset()

    def reset(self):
        self.segment_idx = 0
        self.segment_t0 = None
        self.prev_theta: Dict[int, float] = {}
        self.prev_time: Dict[int, float] = {}
        self.score: Dict[int, float] = {}
        self.energy: Dict[int, float] = {}

    def start(self, now: float):
        self.reset()
        self.segment_t0 = now
        self.segment_idx = 0

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
                self.prev_theta[tid] = tr.theta
                self.prev_time[tid] = now
                self.score.setdefault(tid, 0.0)
                self.energy.setdefault(tid, 0.0)
                continue

            dt = now - self.prev_time[tid]
            if dt <= 1e-6:
                continue

            dth = angle_wrap(tr.theta - self.prev_theta[tid])
            omega = clamp(dth / dt, -self.omega_clip, self.omega_clip)

            if expected_sign == 0:
                self.score[tid] += -abs(omega)
            else:
                self.score[tid] += expected_sign * omega
                self.energy[tid] += abs(omega)

            self.prev_theta[tid] = tr.theta
            self.prev_time[tid] = now

    def pick_best(self, min_energy: float) -> Optional[int]:
        if not self.score:
            return None
        candidates = [(tid, sc) for tid, sc in self.score.items()
                      if self.energy.get(tid, 0.0) >= min_energy]
        if not candidates:
            candidates = list(self.score.items())
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]


# ============================================================
# ---------------------------- MAIN ---------------------------
# ============================================================

def parse_args():
    """All runtime parameters are here. To adjust."""
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
    ap.add_argument("--marker_sep_px", type=float, default=-1.0, help="If <=0, auto-estimate")
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

    return ap.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.cam)
    #cap = cv2.VideoCapture("http://10.207.120.184:8080/video") #use phone "IP webcam application" (change IP)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    commander = RobotCommander(port=args.serial_port, baud=args.baud, dry_run=args.dry_run)

    tracks: List[RobotTrack] = []
    my_id: Optional[int] = None
    expected_sep_px: Optional[float] = None if args.marker_sep_px <= 0 else float(args.marker_sep_px)

    scale_ema = EMA(alpha=args.scale_alpha)
    px_per_in: Optional[float] = None

    identifier = DanceIdentifier(default_id_dance())
    phase = "IDENTIFY" if args.id_dance else "RUN"
    attempts_left = int(args.id_attempts)

    last_log = time.time()

    blue_ranges = [HSVRange(args.blue_h_lo, args.blue_h_hi)]
    red_ranges = [HSVRange(args.red_h1_lo, args.red_h1_hi), HSVRange(args.red_h2_lo, args.red_h2_hi)]

    ctrl = dict(
        stop_dist_px=args.ctrl_stop_px,
        k_ang=args.ctrl_k_ang,
        k_lin=args.ctrl_k_lin,
        v_max=args.ctrl_v_max
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            h, w = frame.shape[:2]

            # 1) HSV segmentation
            blue_mask = mask_hsv(frame, blue_ranges, s_min=args.blue_s_min, v_min=args.blue_v_min)
            red_mask  = mask_hsv(frame, red_ranges,  s_min=args.red_s_min,  v_min=args.red_v_min)

            # 2) Blobs
            front_blobs = extract_blobs(blue_mask, min_area_px=args.blob_min_area)
            rear_blobs  = extract_blobs(red_mask,  min_area_px=args.blob_min_area)

            # 3) Estimate marker separation (if not provided)
            if expected_sep_px is None:
                expected_sep_px = estimate_marker_sep_px(front_blobs, rear_blobs)

            # 4) Pair blobs into robots
            dets = pair_markers(front_blobs, rear_blobs,
                                expected_sep_px=expected_sep_px,
                                sep_tol=args.pair_sep_tol,
                                max_pair_px=args.max_pair_px)

            # 5) Scale estimate (px/in) + dynamic inflation
            inflate_px_dynamic = args.inflate_px
            if dets:
                sep_px_med = float(np.median([d.sep_px for d in dets]))
                px_per_in_meas = sep_px_med / args.marker_sep_in
                px_per_in = scale_ema.update(px_per_in_meas)
                inflate_px_dynamic = int(round(args.robot_radius_in * px_per_in))

            # 6) Track robots
            tracks = update_tracks(tracks, dets, now, max_match_dist_px=90.0, max_misses=12)

            # 7) Obstacles
            robot_excl = build_robot_exclusion_mask((h, w), dets,
                                                   radius_px=args.robot_excl_radius_px,
                                                   body_width_px=args.robot_body_width_px)
            obstacles, obs_mask = detect_obstacles_floor_model(frame, robot_excl,
                                                              k_sigma=tuple(args.k_sigma),
                                                              min_area_px=args.obs_min_area)

            # 8) Grid + planning
            occ_grid = mask_to_occupancy(obs_mask, cell_px=args.cell_px, inflate_px=inflate_px_dynamic)

            # 9) Phase machine
            cmd = {"left": 0.0, "right": 0.0, "laser": False}
            path = None

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
                        identifier.update_scores(tracks, now, expected_sign=seg.expected_omega_sign)
                        cmd = {"left": seg.left, "right": seg.right, "laser": seg.laser}
                    else:
                        my_id = identifier.pick_best(min_energy=args.id_min_energy)
                        commander.send(0.0, 0.0, False)
                        phase = "RUN"

            if phase == "RUN":
                if my_id is None:
                    cmd = {"left": 0.0, "right": 0.0, "laser": False}
                else:
                    if args.mode == "offense":
                        cmd, path = offense_strategy(
                            tracks, my_id, occ_grid, args.cell_px,
                            lookahead_cells=args.lookahead_cells,
                            laser_close_px=args.laser_close_px,
                            laser_align_deg=args.laser_align_deg,
                            ctrl=ctrl
                        )
                    else:
                        cmd, path = defense_strategy(
                            tracks, my_id, occ_grid, args.cell_px,
                            lookahead_cells=args.lookahead_cells,
                            samples=args.def_samples,
                            ctrl=ctrl
                        )
                commander.send(cmd["left"], cmd["right"], cmd["laser"])

            # 10) Logging
            if now - last_log >= args.log_period:
                last_log = now
                print(f"[LOG] phase={phase} robots={len(tracks)} blue={len(front_blobs)} red={len(rear_blobs)} "
                      f"obs={len(obstacles)} my_id={my_id} sep_px={expected_sep_px} scale_px_per_in={px_per_in}")

            # 11) Visualization
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

            header = f"phase={phase} mode={args.mode}  L={cmd['left']:+.2f} R={cmd['right']:+.2f} laser={cmd['laser']}  my_id={my_id}"
            cv2.putText(out, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 3)
            cv2.putText(out, header, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 1)

            if expected_sep_px is None:
                sep_txt = "marker_sep_px: estimating..."
            else:
                sep_txt = f"marker_sep_px~{expected_sep_px:.1f}"
            cv2.putText(out, sep_txt, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2)

            if px_per_in is not None:
                sc_txt = f"scale~{px_per_in:.1f} px/in  inflate={inflate_px_dynamic}px"
                cv2.putText(out, sc_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 2)

            cv2.imshow("MECH663 Overhead (clean)", out)
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


if __name__ == "__main__":
    main()
