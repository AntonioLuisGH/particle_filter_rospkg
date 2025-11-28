#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState

class UAVDynamicModel:
    "Dynamic model for UAV - only used for particle prediction"
    def __init__(self):
        self.max_horizontal_speed = 22.0
        self.max_vertical_speed = 5.0
        self.velocity_decay = 0.98
        self.position_noise_std = 0.1 # changed from 0.1
        self.velocity_noise_std = 0.05 # changed from 0.05

    def constrain_velocity(self, velocity):
        "Constrain velocities to physically plausible limits"
        horizontal_speed = np.linalg.norm(velocity[:2])
        if horizontal_speed > self.max_horizontal_speed:
            scale = self.max_horizontal_speed / horizontal_speed
            velocity[0] *= scale
            velocity[1] *= scale
        
        if abs(velocity[2]) > self.max_vertical_speed:
            velocity[2] = np.sign(velocity[2]) * self.max_vertical_speed
        return velocity









class ParticleFilter:
    "particle filter (XYZ position)"
    def __init__(self, system_position, num_particles=1000):
        self.system_position = system_position
        self.num_particles = num_particles
        self.dynamic_model = UAVDynamicModel()
        
        # Minimal state: each particle has [x, y, z] position only
        self.particles = None  # shape: (num_particles, 3)
        self.velocities = None  # shape: (num_particles, 3)
        self.weights = None
        
    def initialize_particles(self, initial_range=(-70, 70)):
        "Initialize particles with random positions around system"
        # Position particles randomly in 3D space around system
        self.particles = np.random.uniform(-initial_range[1], initial_range[1], 
                                         (self.num_particles, 3))
        self.particles[:, 2] = np.abs(self.particles[:, 2])  # Keep z positive
        
        # Initialize velocities
        self.velocities = np.random.uniform(-1, 1, (self.num_particles, 3))
        for i in range(self.num_particles):
            self.velocities[i] = self.dynamic_model.constrain_velocity(self.velocities[i])
            
        # Initialize uniform weights
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def cartesian_to_angles(self, position):
        "Convert Cartesian position to angles relative to system origin"
        relative_pos = position - self.system_position
        distance = np.linalg.norm(relative_pos)
        
        if distance > 0:
            azimuth = np.arctan2(relative_pos[1], relative_pos[0])
            elevation = np.arcsin(relative_pos[2] / distance)
        else:
            azimuth = 0.0
            elevation = 0.0
            
        return azimuth, elevation
        
    def predict(self, dt=0.1):
        "Predict particle movement using kinematic model"
        if self.particles is None:
            self.initialize_particles()
            
        # Add process noise to velocities
        velocity_noise = np.random.normal(0, self.dynamic_model.velocity_noise_std, 
                                        (self.num_particles, 3))
        self.velocities += velocity_noise
        
        # Apply velocity constraints and decay
        for i in range(self.num_particles):
            self.velocities[i] = self.dynamic_model.constrain_velocity(self.velocities[i])
            self.velocities[i] *= self.dynamic_model.velocity_decay
        
        # Update positions
        self.particles += self.velocities * dt
        
        # Add position noise
        position_noise = np.random.normal(0, self.dynamic_model.position_noise_std, 
                                        (self.num_particles, 3))
        self.particles += position_noise
        
        # Ensure particles stay above ground
        self.particles[:, 2] = np.maximum(self.particles[:, 2], 0.1)
            
    def update_with_doa(self, doa_data, doa_std_rad=0.1):
        "Update weights based on DOA measurement - compute angles on-demand"
        doa_azimuth = np.deg2rad(doa_data[0])
        doa_elevation = np.deg2rad(doa_data[1])
        
        for i in range(self.num_particles):
            # Convert particle position to predicted angles
            pred_azimuth, pred_elevation = self.cartesian_to_angles(self.particles[i])
            
            # Calculate angular errors
            azimuth_error = self.angle_difference(pred_azimuth, doa_azimuth)
            elevation_error = self.angle_difference(pred_elevation, doa_elevation)
            
            # Combined angular likelihood
            azimuth_likelihood = np.exp(-0.5 * (azimuth_error / doa_std_rad) ** 2)
            elevation_likelihood = np.exp(-0.5 * (elevation_error / doa_std_rad) ** 2)
            
            angular_likelihood = azimuth_likelihood * elevation_likelihood
            
            # Update weight
            self.weights[i] *= angular_likelihood
        
        # Normalize weights
        self.normalize_weights()

    def update_with_pp(self, pp_position, position_std=0.5):
        "Update weights based on PointPillars position measurement"
        for i in range(self.num_particles):
            # Calculate position error directly
            position_error = np.linalg.norm(self.particles[i] - pp_position)
            position_likelihood = np.exp(-0.5 * (position_error / position_std) ** 2)
            
            # Update weight
            self.weights[i] *= position_likelihood
        
        # Normalize weights
        self.normalize_weights()

    def normalize_weights(self):
        "Normalize weights with robustness check"
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            # Reset if all weights become zero (numerical issues)
            self.weights = np.ones(self.num_particles) / self.num_particles

    def angle_difference(self, angle1, angle2):
        "Calculate smallest difference between two angles"
        diff = angle1 - angle2
        return np.arctan2(np.sin(diff), np.cos(diff))

    def resample(self):
        "Systematic resampling to concentrate on high-probability particles"
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0  # Ensure numerical stability
        
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        indices = np.zeros(self.num_particles, dtype=int)
        
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
                
        # Resample particles and velocities
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        
        # Add small noise to resampled particles to maintain diversity
        resample_noise = np.random.normal(0, 0.01, self.particles.shape)
        self.particles += resample_noise
        
        # Reset weights to uniform
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def process_measurement(self, sensor_type, sensor_data, dt=0.2):
        "Main processing function for any sensor type"
        if self.particles is None:
            self.initialize_particles()
        
        # 1. PREDICT: Move particles according to dynamics
        self.predict(dt)
        
        # 2. UPDATE: Update weights based on sensor type
        if sensor_type == 'doa':
            self.update_with_doa(sensor_data)
        elif sensor_type == 'pp':
            self.update_with_pp(sensor_data)
        
        # 3. RESAMPLE: Keep good particles, discard bad ones
        self.resample()
    
    def estimate_state(self):
        "Get position and velocity estimates using weighted average"
        if self.particles is not None and self.weights is not None:
            # Use weighted average instead of single best particle
            position = np.average(self.particles, axis=0, weights=self.weights)
            velocity = np.average(self.velocities, axis=0, weights=self.weights)
            
            # Convert to angles for output
            azimuth, elevation = self.cartesian_to_angles(position)
            
            return position, velocity, azimuth, elevation
        return np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0, 0.0
    










class ParticleFilterNode(Node):
    "ROS node with minimal state particle filter"
    def __init__(self):
        super().__init__('particle_filter_node')
        
        # System position (sensor location)
        self.system_position = np.array([0, 0, 0])
        
        # Initialize particle filter
        self.pf = ParticleFilter(self.system_position, num_particles=500)
        
        # Data storage for visualization
        self.true_positions = []
        self.estimated_positions = []
        self.doa_measurements = []
        self.pp_measurements = []
        
        # Latest sensor data
        self.latest_doa_data = None
        self.latest_pp_data = None
        self.latest_true_state = None
        
        # Subscribers
        self.doa_sub = self.create_subscription(
            DOAData,
            '/sensors/doa',
            self.doa_callback,
            10
        )
        
        self.pp_sub = self.create_subscription(
            PointPillarsData,
            '/sensors/point_pillars',
            self.pp_callback,
            10
        )
        
        self.true_state_sub = self.create_subscription(
            DroneState,
            '/drone/true_state',
            self.true_state_callback,
            10
        )
        
        # Publisher for filter results
        self.filter_state_pub = self.create_publisher(ParticleFilterState, '/filter/state', 10)
        
        # Visualization setup
        self.setup_plot()
        
        # Main processing timer
        self.timer = self.create_timer(0.1, self.process_measurements)
        
        self.get_logger().info("Minimal state particle filter node started")

    def setup_plot(self):
        "Initialize 3D plot for visualization"
        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-50, 50])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([0, 50])
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Particle Filter - 3D Tracking')

    def doa_callback(self, msg):
        "Store latest DOA data"
        self.latest_doa_data = np.array([msg.azimuth, msg.elevation])
        self.get_logger().debug(f"Received DOA: az={msg.azimuth:.1f}°, el={msg.elevation:.1f}°")

    def pp_callback(self, msg):
        "Store latest PointPillars data"
        self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.get_logger().debug(f"Received PP: ({self.latest_pp_data[0]:.1f}, {self.latest_pp_data[1]:.1f}, {self.latest_pp_data[2]:.1f})")

    def true_state_callback(self, msg):
        "Store latest true state for visualization only"
        self.latest_true_state = {
            'position': np.array([msg.true_position.x, msg.true_position.y, msg.true_position.z])
        }
        self.true_positions.append(self.latest_true_state['position'].copy())

    def process_measurements(self):
        "Main processing - filter incoming sensor data"
        has_new_data = False
        
        # Process DOA data
        if self.latest_doa_data is not None:
            self.pf.process_measurement('doa', self.latest_doa_data)
            
            # Convert DOA to 3D point for visualization (at fixed distance)
            azimuth_rad = np.deg2rad(self.latest_doa_data[0])
            elevation_rad = np.deg2rad(self.latest_doa_data[1])
            x = 30 * np.cos(elevation_rad) * np.cos(azimuth_rad)
            y = 30 * np.cos(elevation_rad) * np.sin(azimuth_rad)
            z = 30 * np.sin(elevation_rad)
            self.doa_measurements.append(np.array([x, y, z]))
            
            has_new_data = True
            self.latest_doa_data = None  # Clear after processing
            
        # Process PointPillars data
        if self.latest_pp_data is not None:
            self.pf.process_measurement('pp', self.latest_pp_data)
            self.pp_measurements.append(self.latest_pp_data.copy())
            has_new_data = True
            self.latest_pp_data = None  # Clear after processing
        
        if has_new_data:
            # Get estimate
            est_position, est_velocity, est_azimuth, est_elevation = self.pf.estimate_state()
            self.estimated_positions.append(est_position.copy())
            
            # Publish filter result
            self.publish_filter_state(est_position, est_velocity)
            
            # Update visualization
            self.update_plot()

    def publish_filter_state(self, position, velocity):
        "Publish filtered state"
        filter_msg = ParticleFilterState()
        filter_msg.header.stamp = self.get_clock().now().to_msg()
        filter_msg.header.frame_id = "world"
        
        # Position
        filter_msg.estimated_position.x = float(position[0])
        filter_msg.estimated_position.y = float(position[1])
        filter_msg.estimated_position.z = float(position[2])
        
        # Velocity
        filter_msg.estimated_velocity.x = float(velocity[0])
        filter_msg.estimated_velocity.y = float(velocity[1])
        filter_msg.estimated_velocity.z = float(velocity[2])
        
        self.filter_state_pub.publish(filter_msg)

    def update_plot(self):
        "Update visualization with current data"
        self.ax.clear()
        
        self.ax.set_xlim([-50, 50])
        self.ax.set_ylim([-50, 50])
        self.ax.set_zlim([0, 50])
        self.ax.set_title('Particle Filter - 3D Tracking')
        
        # Plot system origin
        self.ax.scatter(*self.system_position, c='black', s=100, marker='*', label='System Origin')
        
        # Plot true trajectory if available
        if len(self.true_positions) > 1:
            true_traj = np.array(self.true_positions)
            self.ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], 
                        'g-', linewidth=2, label='True Trajectory')
        
        # Plot estimated trajectory
        if len(self.estimated_positions) > 1:
            est_traj = np.array(self.estimated_positions)
            self.ax.plot(est_traj[:, 0], est_traj[:, 1], est_traj[:, 2], 
                        'b-', linewidth=2, label='Estimated Trajectory')
        
        # Plot particles
        if self.pf.particles is not None:
            self.ax.scatter(self.pf.particles[:, 0], self.pf.particles[:, 1], self.pf.particles[:, 2], 
                          c='red', alpha=0.3, s=10, label='Particles')
        
        # Plot current estimate
        if len(self.estimated_positions) > 0:
            current_est = self.estimated_positions[-1]
            self.ax.scatter(*current_est, c='blue', s=100, marker='s', 
                          label='Current Estimate')
        
        # Plot current true position if available
        if self.latest_true_state is not None:
            self.ax.scatter(*self.latest_true_state['position'], c='green', s=100, 
                          marker='o', label='True Position')
        
        self.ax.legend()
        plt.draw()
        plt.pause(0.01)

def main():
    rclpy.init()
    node = ParticleFilterNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == "__main__":
    main()