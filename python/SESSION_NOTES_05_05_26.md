# Session Notes — 5 May 2026

## Project Overview
Arduino Giga R1 WiFi robot with RPLiDAR. Python SLAM pipeline (`hybrid_slam.py`, `live_slam.py`) drives a proportional bearing controller (`real_nav.py`) to navigate to a user-specified goal from the robot's start position (SLAM origin).

**Hardware constants:**
- Wheel radius: 0.034 m
- Encoder counts/rev: 152
- Wheel base: 0.15 m
- Arduino IP: 192.168.0.197 (changed this session — was 192.168.0.199)
- Laptop IP: 192.168.0.200, Port: 4010

---

## Changes Made This Session

### 1. Arduino IP updated — `live_slam.py` line 50
**Old:** `ARDUINO_IP = "192.168.0.199"`  
**New:** `ARDUINO_IP = "192.168.0.197"`  
Arduino was assigned a new IP by the router. `parameters.py` had already been updated by the user but `live_slam.py` was hardcoded separately.

---

### 2. Early goal-termination fix — `real_nav.py`
**Problem:** Robot stopped at ~30 cm. ICP on the first few scans produced a large spurious translation jump, EKF absorbed it, controller reported goal reached immediately, sent speed=0.

**Fix:** Added an odometry-based minimum travel guard. The goal check and the speed=0 command are both suppressed until odometry reports the robot has physically covered at least 50% of the goal distance.

```python
_goal_dist_total = math.hypot(goal_x, goal_y)
_min_travel      = _goal_dist_total * 0.5
# In loop:
if _odo_travel < _min_travel and speed == 0:
    speed = base_speed   # keep driving regardless of EKF
# Goal check:
if _odo_travel >= _min_travel and ctrl.at_goal(ex, ey):
    ...
```

---

### 3. Motor stall fix — `real_nav.py`
**Problem:** Robot would stop moving but timeout wouldn't trigger. Audible motor hum. Speed was being reduced to `MIN_SPEED=28` inside the 0.40 m slow zone, which was below the robot's stall threshold.

**Changes:**
- `SLOW_RADIUS_M`: 0.40 → **0.15 m** (decel zone nearly eliminated)
- `MIN_SPEED`: 28 → **45** (above stall threshold)

---

### 4. Exception handler added — `real_nav.py`
Added a bare `except Exception` block so any Python crash inside the drive loop prints a traceback rather than silently stopping the robot.

---

### 5. ICP consistency threshold tightened — `hybrid_slam.py`
**Problem:** ICP occasionally converged to a mirror-symmetric wrong rotation (~47° off). `CONSISTENCY_THRESH = 135°` let it through. EKF absorbed a −20° heading flip, causing controller to steer in the wrong direction for the entire run.

**Change:** `CONSISTENCY_THRESH`: 135° → **25°**  
Any ICP result whose rotation disagrees with odometry by more than 25° is rejected. The trade-off is more ICP rejections on turns, but the EKF heading stays stable.

---

### 6. Slip factor calibration — `hybrid_slam.py`
**Problem:** Robot physically covered only ~29 cm when odometry reported ~0.93 m. Encoder is on the motor shaft and counts rotations regardless of wheel slip.

**Calibration:** 10 runs of `--goal 1.0 0.0 --speed 100`, physical back-wheel arc measured manually. Mean slip = 0.328 (only ~33% of encoder counts translate to ground displacement).

**After joint optimisation with STEER_SCALE (see below):**  
`SLIP_FACTOR` = **0.220**

```python
SLIP_FACTOR   = 0.220
ENCODER_SCALE = 2 * math.pi * WHEEL_RADIUS / ENCODER_COUNTS_PER_REV * SLIP_FACTOR
```

---

### 7. Steering scale calibration — `hybrid_slam.py` + `live_slam.py`
**Problem:** EKF heading and Y position were consistently wrong on diagonal runs. Post-run plots showed EKF θ diverging sharply from odometry. Physical Y displacement was much less than commanded.

**Root cause:** The servo linkage delivers only a fraction of the commanded steering angle to the wheels. The model assumed commanded angle = actual wheel angle.

**Calibration:** 5 runs of `--goal 1.0 1.0 --speed 100` from a wall corner. Back-wall and side-wall distances measured after each run. Joint grid search over `SLIP_FACTOR` × `STEER_SCALE` minimised RMSE across all 5 runs.

**Result:** `STEER_SCALE` = **0.400**, RMSE ≈ 4 cm

```python
STEER_SCALE = 0.400
```

Applied in **every** location where steering angle feeds into heading:
- `hybrid_slam.py` — `compute_odometry_trajectory()` 
- `hybrid_slam.py` — ICP consistency check (`dtheta_odom`)
- `hybrid_slam.py` — `EKF.predict()`
- `live_slam.py` — `_odo_step()`
- `live_slam.py` — ICP consistency check (`dtheta_odom`)

```python
phi = math.radians(steering_deg * STEER_SCALE)
```

`STEER_SCALE` is defined in `hybrid_slam.py` and imported into `live_slam.py`.

---

### 8. Goal radius reduced — `real_nav.py`
`GOAL_RADIUS_M`: 0.15 → **0.05 m**  
With short runs (~1 m), stopping 15 cm early looked poor visually.

---

### 9. Live video recording — `real_nav.py`
Added optional `--save-video` flag. When set, each matplotlib frame is captured via `fig.canvas.buffer_rgba()` and written to an mp4 using `cv2.VideoWriter` at 5 fps.

**Usage:**
```bash
python real_nav.py --goal 1.0 1.0 --speed 100 --save-video
```

Output: `results/nav_real/real_nav_{timestamp}.mp4`

---

## Current Calibrated Constants (`hybrid_slam.py`)

```python
WHEEL_RADIUS           = 0.034
ENCODER_COUNTS_PER_REV = 152
WHEEL_BASE             = 0.15
SLIP_FACTOR            = 0.220       # calibrated: encoder→ground displacement
ENCODER_SCALE          = 2 * pi * WHEEL_RADIUS / ENCODER_COUNTS_PER_REV * SLIP_FACTOR
STEER_SCALE            = 0.400       # calibrated: commanded→actual wheel angle
ICP_STRIDE             = 4
ICP_ERROR_THRESHOLD    = 0.10
MIN_SCAN_POINTS        = 20
MIN_DS                 = 0.02
CONSISTENCY_THRESH     = radians(25)
```

---

## Key Behavioural Notes

- **ICP acceptance rate is low (~1/16) on straight runs** — expected. The aperture problem prevents ICP from converging well on straight corridors with 50 rays. EKF falls back to pure odometry, which is accurate enough for runs ≤ ~1.5 m at speed 100.
- **Calibration is speed-dependent.** `SLIP_FACTOR` and `STEER_SCALE` were calibrated at `--speed 100`. At lower speeds, slip changes and the model will be less accurate.
- **ICP corrections become more valuable on longer runs or sharp turns** — scan-to-scan geometry changes more, giving ICP a proper signal to anchor rotation.
- **`CONSISTENCY_THRESH = 25°`** may reject valid ICP on sharp turns. If future runs involve large-angle turns, consider loosening to ~40°.

---

## Files Modified This Session

| File | Changes |
|---|---|
| `live_slam.py` | Arduino IP, STEER_SCALE import + applied in `_odo_step` and ICP consistency |
| `hybrid_slam.py` | SLIP_FACTOR, STEER_SCALE, CONSISTENCY_THRESH; STEER_SCALE applied in odometry, ICP, EKF.predict |
| `real_nav.py` | Early-stop guard, stall fix, exception handler, GOAL_RADIUS_M, SLOW_RADIUS_M, MIN_SPEED, --save-video |
