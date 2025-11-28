  
# Particle Filter System Simulation

A ROS 2 simulation for tracking Unmanned Aerial Vehicles (UAVs). Simulates a target drone, produces noisy sensor data (DOA and PointPillars), and estimates the drone's state with a particle filter.

## 🚀 Features

- **Drone Simulator:** simulates a UAV flying a randomized path toward a target.
- **Sensor Simulator:** generates realistic noisy readings:
  - **DOA (Direction of Arrival):** azimuth and elevation angles.
  - **PointPillars:** 3D position estimates (X, Y, Z).
- **Particle Filter Tracker:** fuses multisensor data to estimate position and velocity in real time.
- **Visualization:** real-time 3D plotting of true path, sensor readings, and estimated path.

## 📋 Prerequisites

- **OS:** Ubuntu 22.04 (Jammy) or 20.04 (Focal)
- **ROS 2:** Humble or Foxy
- **Python packages:** `numpy`, `matplotlib`

## 🛠️ Installation

1. Create a workspace (if you don't have one):

   ```bash
   mkdir -p ~/particle_filter_ws/src
   cd ~/particle_filter_ws/src
   ```

2. Add the required packages to `src/`:

   - `counter_uav_system` (this repository)
   - `uav_interfaces` (custom message definitions)

   Example structure:

   ```
   ~/particle_filter_ws/
   └── src/
       ├── counter_uav_system/
       │   ├── package.xml
       │   ├── setup.py
       │   └── ...
       └── uav_interfaces/
           ├── CMakeLists.txt
           ├── package.xml
           └── msg/
   ```

3. Build the workspace:

   ```bash
   cd ~/particle_filter_ws

   # Clean previous builds (optional)
   rm -rf build/ install/ log/

   # Build with symlink install for easy Python edits
   colcon build --symlink-install
   ```

4. Source the workspace:

   ```bash
   source install/setup.bash
   ```

## 🏃 Usage

Run three separate nodes in three separate terminals.

Terminal 1 — Drone Simulator (true drone):

```bash
source install/setup.bash
ros2 run counter_uav_system drone_sim_node
```

Terminal 2 — Sensor Simulator (adds noise):

```bash
source install/setup.bash
ros2 run counter_uav_system sensor_sim_node
```

Terminal 3 — Particle Filter (tracker + viz):

```bash
source install/setup.bash
ros2 run counter_uav_system particle_filter_node
```

## 🔧 Troubleshooting

- "ImportError: numpy.core.multiarray failed to import" or Matplotlib crashes  
  If you installed `numpy`/`matplotlib` with `pip` and they conflict with the system versions, run the node ignoring local site packages:

  ```bash
  PYTHONNOUSERSITE=1 ros2 run counter_uav_system particle_filter_node
  ```

- "ModuleNotFoundError: No module named 'uav_interfaces'"  
  Ensure `uav_interfaces` is present in `src/`, then:

  ```bash
  colcon build
  source install/setup.bash
  ```
