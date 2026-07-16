# Localization and Navigation with Geometric ICP

Real-world and replayable SLAM/navigation pipeline for a WiFi-connected Arduino robot with RPLiDAR.

This project combines:
- Odometry from encoder + steering
- ICP-based scan matching
- EKF sensor fusion
- Goal-directed navigation with safety limits
- Live plotting, session recording, and post-run analysis

## Source Attribution

This project builds on course material and prior work from:

- NYU ROB-GY 6213: Robot Localization & Navigation
- Chris Clark
- https://github.com/cmclarkk/NYU_ROB_GY_6213

## System Overview

Data flow:

1. Arduino reads encoder + steering + LiDAR rays.
2. Arduino sends UDP packets to the laptop.
3. Python ingests packets and updates odometry.
4. ICP aligns scan pairs for geometric correction.
5. EKF fuses odometry + ICP into a corrected pose estimate.
6. Navigation controller commands speed/steering toward goal.

## Repository Structure

```text
arduino/arduino/robot_arduino_code/
  robot_arduino_code.ino      # Robot firmware: WiFi UDP, motor/servo, encoder, LiDAR

python/
  live_slam.py                # Live SLAM visualizer
  real_nav.py                 # Goal navigation on hardware (with safety and logging)
  auto_drive_slam.py          # Predefined autonomous drive sequences + SLAM analysis
  hybrid_slam.py              # Core SLAM logic (ICP + EKF + map building)
  extended_kalman_filter.py
  motion_models.py
  parameters.py
  ...

data/
  *.pkl                       # Recorded sessions

results/
  geo_on/, geo_off/, nav_real/  # Saved trajectories, maps, diagnostics, navigation outputs
```

## Hardware Setup

Expected hardware in code:
- Arduino Giga R1 WiFi
- RPLiDAR
- Differential drive base + steering servo
- Wheel encoder

Network assumptions in Python scripts:
- Robot/Arduino UDP endpoint: 192.168.0.197:4010
- Laptop local bind address: 192.168.0.200:4010

If your network differs, update IP values in python/live_slam.py and firmware settings in robot_arduino_code.ino.

## Python Dependencies

Install with your environment manager (pip/conda). Core packages used by scripts:
- numpy
- scipy
- matplotlib
- opencv-python
- pyserial
- fastapi
- nicegui

Minimal install example:

```bash
pip install numpy scipy matplotlib opencv-python pyserial fastapi nicegui
```

## Running the Pipeline

Run commands from the python directory.

### 1) Live SLAM visualization

```bash
python live_slam.py
```

Shows odometry vs EKF+ICP trajectory, current LiDAR scan, and ICP acceptance statistics.

### 2) Autonomous sequence + SLAM logging

```bash
python auto_drive_slam.py
python auto_drive_slam.py --sequence right_circle
python auto_drive_slam.py --sequence figure_eight
```

Saves session logs in data/ and comparison figures/maps in results/.

### 3) Real goal navigation

```bash
python real_nav.py --goal 1.5 0.0
python real_nav.py --goal 1.0 1.0 --speed 55
python real_nav.py --goal 1.2 0.3 --timeout 90
```

Optional run recording:

```bash
python real_nav.py --goal 1.0 1.0 --save-video
```

## Safety Features

Built-in safeguards in navigation scripts include:
- Ctrl-C emergency stop handling
- Timeout stop
- Max-distance emergency stop
- No-signal stop on firmware side

Always test with wheels lifted or in a controlled space after parameter changes.

## Data and Results

- data/ contains raw session pickles for replay and analysis.
- results/ contains generated trajectory plots, heading/error diagnostics, map views, and real navigation outputs.
- geo_on and geo_off folders support comparisons of geometric initialization settings.

## Notes

- Calibration constants (slip factor, steering scale, ICP thresholds) strongly affect performance and are speed/platform dependent.
- Session notes with tuning rationale are documented in python/SESSION_NOTES_05_05_26.md.
