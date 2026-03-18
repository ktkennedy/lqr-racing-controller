# LQR Racing Controller

Speed-scheduled LQR preview controller for autonomous racing, based on a bicycle vehicle model.

Ported from the C++ ROS controller used by HMCL-UNIST (USAFE Racing Team) at KIAPI 2024.

![demo](demo.gif)

## Features

- Discrete-time LQR with Bilinear (Tustin) discretization
- Speed-scheduled Q/R weights (20~110+ km/h)
- Bicycle model with actuator lag modeling
- Steering rate limiting
- Real-time matplotlib visualization

## Run

```bash
pip install numpy matplotlib scipy
python3 lqr_controller.py
```

## Controls

| Key | Action |
|-----|--------|
| Space | Pause / Resume |
| Up/Down | Adjust target speed +/-10 km/h |
| R | Reset simulation |
| Q / Esc | Quit |

## Vehicle Parameters

| Parameter | Value |
|-----------|-------|
| Wheelbase | 2.72 m |
| Front axle to CG | 1.58 m |
| Rear axle to CG | 1.59 m |
| Mass | 1700 kg |
| Control rate | 20 Hz |
| Actuator lag | 0.14 s |
| Steering rate limit | 0.5 rad/s |

## Speed-Scheduled Weights

| Speed (km/h) | Q_ey | Q_epsi | R |
|---------------|------|--------|---|
| > 108 | 0.8 | 1.3 | 7800 |
| > 100 | 1.0 | 1.5 | 7500 |
| > 85 | 1.0 | 2.5 | 7500 |
| > 78 | 4.2 | 3.0 | 7000 |
| > 55 | 2.0 | 6.0 | 6000 |
| > 30 | 4.0 | 7.0 | 4500 |
| > 20 | 3.0 | 7.0 | 2500 |

Higher speed = higher R (conservative steering). Lower speed = higher Q (aggressive tracking).

## Architecture

```
                    ┌─────────────────┐
  Track (oval) ───> │  Find closest   │
                    │  waypoint       │
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
  Vehicle state ──> │  Compute errors │ ey, epsi
                    │  (lateral/heading)│
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
  Speed ──────────> │  Schedule Q/R   │
                    │  weights        │
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
                    │  Build A,B      │
                    │  (Bilinear/Tustin)│
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
                    │  Solve DARE     │ scipy
                    │  K = (R+B'PB)^-1│
                    │      B'PA       │
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
                    │  u = -Kx        │
                    │  + rate limit   │
                    │  + actuator lag │
                    └────────┬────────┘
                             v
                    ┌─────────────────┐
                    │  Bicycle model  │
                    │  update x,y,yaw │
                    └─────────────────┘
```
