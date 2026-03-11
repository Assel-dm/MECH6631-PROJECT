# MECH6631-PROJECT

This project implements a real-time **overhead camera** pipeline for the MECH663 robot WAR.

## What it does
- Detects **robot markers** from an overhead camera:
  - **Blue = front**, **Red = rear** (HSV segmentation)
- Pairs blue/red markers into **robot poses** (x, y, heading) and keeps **stable IDs** with tracking
- Automatically identifies **our robot** using an **ID dance** (short motion signature sent via Bluetooth)
- Detects **obstacles** using a simple **floor color model** (Lab space) and builds an **obstacle mask**
- Converts the obstacle mask into an **occupancy grid** (with obstacle inflation for robot clearance)
- Plans paths with **A\*** and outputs **differential-drive commands**:
  - **Offense**: shortest path to the nearest enemy + laser when close/aligned
  - **Defense**: move to a free area far from the nearest enemy 

## How to run (Python)
Use --dry_run to print commands instead of sending them to the robot
```bash
python overhead_vision_v3.py --cam 0 --dry_run --show_masks --show_obs_mask --draw_path
