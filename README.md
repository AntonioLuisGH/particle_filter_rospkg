# Counter UAV System (AAU Project)

**Department of Electronic Systems, Aalborg University (AAU)**

A ROS 2 architecture for tracking Unmanned Aerial Vehicles (UAVs). This project fuses noisy sensor data to estimate a target's position and drives a custom 2-DOF Pan-Tilt turret to aim a Conical LiDAR at the target in real-time.

## 🚀 System Overview

The system consists of two main subsystems:
1. **Simulation & Estimation:** Simulates a target drone and sensors, using a Particle Filter to estimate state.
2. **Actuation (Pan-Tilt):** Controls a physical or simulated turret to align with the estimated coordinates.

### Key Features
- **Drone Simulator:** Simulates a UAV flying randomized paths.
- **Sensor Simulator:** Generates realistic noisy readings (DOA Azimuth/Elevation and PointPillars 3D coordinates).
- **Particle Filter Tracker:** Fuses multisensor data to estimate position and velocity.
- **Pan-Tilt Control:** Automatically aims a turret at the estimated 3D coordinates.
- **Visualization:** Real-time 3D plotting of trajectories and URDF visualization of the turret state.

---

## 📡 Subsystem 1: Particle Filter & Simulation

This module handles the generation of data and the estimation of the drone's state.

- **Inputs:** DOA (Acoustic) and PointPillars (LiDAR) data.
- **Output:** Estimated 3D position $(x, y, z)$ published to `/cmd_point`.
- **Visualization:** Matplotlib 3D plots showing True Path vs. Estimated Path.

---

## 🦾 Subsystem 2: Pan-Tilt Control System

This module controls the physical aiming hardware. It subscribes to the `/cmd_point` topic from the particle filter and performs Inverse Kinematics (IK) to align the sensor.

- **Function:** Converts 3D target points into Azimuth (Pan) and Elevation (Tilt) angles.
- **Hardware:** Designed for a 2-DOF turret using Dynamixel servomotors and an Arduino controller.
- **Offsets:** The driver automatically accounts for the physical mounting offsets of the LiDAR sensor to ensure accurate aiming.

---

## 🛠️ Installation

### Prerequisites
- **OS:** Ubuntu 22.04 (Jammy)
- **ROS 2:** Humble
- **Python packages:** `numpy`, `matplotlib`, `pyserial`

### Workspace Setup

1. **Create a workspace:**
   ```bash
   mkdir -p ~/counter_uav_ws/src
   cd ~/counter_uav_ws/src
   ```

2. **Clone/Add required packages to `src/`:**
   Ensure the following packages are present in your source folder:
   - `counter_uav_system` (Simulation & Filter)
   - `uav_interfaces` (Custom messages)
   - `pan_tilt_control` (Turret driver & description)

3. **Build the workspace:**
   ```bash
   cd ~/counter_uav_ws
   rm -rf build/ install/ log/
   colcon build --symlink-install
   ```

4. **Source the workspace:**
   ```bash
   source install/setup.bash
   ```

---

## 🏃 Usage

To run the full system, you will need multiple terminals.

### Step 1: Start the Simulation & Tracker

**Terminal 1 — Drone Simulator (True State):**
```bash
ros2 run counter_uav_system drone_sim_node
```

**Terminal 2 — Sensor Simulator (Noisy Data):**
```bash
ros2 run counter_uav_system sensor_sim_node
```

**Terminal 3 — Particle Filter (Estimation):**
This node estimates the position and publishes the target to `/cmd_point`.
```bash
ros2 run counter_uav_system particle_filter_node
```

### Step 2: Start the Pan-Tilt System

**Terminal 4 — Pan-Tilt Driver & Visualizer:**
This launches the driver, robot state publisher, and RViz.
```bash
ros2 launch pan_tilt_control pan_tilt.launch.py
# Optional argument: serial_port:=/dev/ttyACM0 (Default is /dev/ttyUSB0)
```

---

## ⚙️ Configuration

- **Turret Geometry:** If you modify the physical 3D printed parts (heights or sensor offsets), update the constants in `pan_tilt_control/driver_node.py` and the visual model in `pan_tilt_control/urdf/pan_tilt.urdf`.
- **Servo Limits:** Safety limits for the servo rotation are also defined in `driver_node.py`.

---

## 🔧 Troubleshooting

**`ImportError: numpy.core.multiarray failed to import`**
If using a virtual environment or conflicting pip versions:
```bash
PYTHONNOUSERSITE=1 ros2 run counter_uav_system particle_filter_node
```

**`ModuleNotFoundError: No module named 'uav_interfaces'`**
Ensure the package is built and sourced:
```bash
colcon build --packages-select uav_interfaces
source install/setup.bash
```

**`Permission denied /dev/ttyUSB0`**
Add your user to the dialout group:
```bash
sudo usermod -a -G dialout $USER
```