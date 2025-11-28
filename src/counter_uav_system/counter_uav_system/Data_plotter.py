#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from geometry_msgs.msg import Point, Vector3
from uav_interfaces.msg import DroneState, ParticleFilterState
import collections
import time

class ParticleFilterVisualizer(Node):
    def __init__(self):
        super().__init__('particle_filter_visualizer')
        
        # Data storage - store as separate coordinate lists
        self.true_x = collections.deque(maxlen=1000)
        self.true_y = collections.deque(maxlen=1000)
        self.true_z = collections.deque(maxlen=1000)
        self.est_x = collections.deque(maxlen=1000)
        self.est_y = collections.deque(maxlen=1000)
        self.est_z = collections.deque(maxlen=1000)
        
        self.true_vx = collections.deque(maxlen=1000)
        self.true_vy = collections.deque(maxlen=1000)
        self.true_vz = collections.deque(maxlen=1000)
        self.est_vx = collections.deque(maxlen=1000)
        self.est_vy = collections.deque(maxlen=1000)
        self.est_vz = collections.deque(maxlen=1000)
        
        self.true_timestamps = collections.deque(maxlen=1000)
        self.est_timestamps = collections.deque(maxlen=1000)
        
        # Error metrics
        self.position_errors = collections.deque(maxlen=1000)
        self.velocity_errors = collections.deque(maxlen=1000)
        self.error_timestamps = collections.deque(maxlen=1000)
        
        # Synchronization flags
        self.has_true_data = False
        self.has_est_data = False
        
        # Subscribers
        self.true_state_sub = self.create_subscription(
            DroneState,
            '/drone/true_state',
            self.true_state_callback,
            10
        )
        
        self.filter_state_sub = self.create_subscription(
            ParticleFilterState,
            '/filter/state',
            self.filter_state_callback,
            10
        )
        
        # Setup plotting
        self.setup_plots()
        
        self.get_logger().info("Particle Filter Visualizer started")

    def setup_plots(self):
        """Initialize all plots with subplots"""
        plt.ion()
        self.fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=self.fig)
        
        # 3D Trajectory Plot
        self.ax_3d = self.fig.add_subplot(gs[0:2, 0:2], projection='3d')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Trajectory - True vs Estimated')
        
        # Position Error Over Time
        self.ax_error = self.fig.add_subplot(gs[0, 2])
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Position Error (m)')
        self.ax_error.set_title('Position Estimation Error')
        self.ax_error.grid(True)
        
        # Velocity Error Over Time
        self.ax_vel_error = self.fig.add_subplot(gs[1, 2])
        self.ax_vel_error.set_xlabel('Time (s)')
        self.ax_vel_error.set_ylabel('Velocity Error (m/s)')
        self.ax_vel_error.set_title('Velocity Estimation Error')
        self.ax_vel_error.grid(True)
        
        # XYZ Position Components
        self.ax_xyz = self.fig.add_subplot(gs[2, :])
        self.ax_xyz.set_xlabel('Time (s)')
        self.ax_xyz.set_ylabel('Position (m)')
        self.ax_xyz.set_title('XYZ Position Components - True vs Estimated')
        self.ax_xyz.grid(True)
        
        plt.tight_layout()

    def true_state_callback(self, msg):
        """Store true drone state"""
        timestamp = time.time()
        
        self.true_x.append(msg.true_position.x)
        self.true_y.append(msg.true_position.y)
        self.true_z.append(msg.true_position.z)
        
        self.true_vx.append(msg.true_velocity.x)
        self.true_vy.append(msg.true_velocity.y)
        self.true_vz.append(msg.true_velocity.z)
        
        self.true_timestamps.append(timestamp)
        self.has_true_data = True

    def filter_state_callback(self, msg):
        """Store filter estimates"""
        timestamp = time.time()
        
        self.est_x.append(msg.estimated_position.x)
        self.est_y.append(msg.estimated_position.y)
        self.est_z.append(msg.estimated_position.z)
        
        self.est_vx.append(msg.estimated_velocity.x)
        self.est_vy.append(msg.estimated_velocity.y)
        self.est_vz.append(msg.estimated_velocity.z)
        
        self.est_timestamps.append(timestamp)
        self.has_est_data = True
        
        # Calculate error if we have corresponding true data
        if len(self.true_x) > 0 and len(self.est_x) > 0:
            # Use the latest true data point for error calculation
            true_idx = min(len(self.true_x) - 1, len(self.est_x) - 1)
            est_idx = len(self.est_x) - 1
            
            pos_error = np.sqrt(
                (self.true_x[true_idx] - self.est_x[est_idx])**2 +
                (self.true_y[true_idx] - self.est_y[est_idx])**2 +
                (self.true_z[true_idx] - self.est_z[est_idx])**2
            )
            
            vel_error = np.sqrt(
                (self.true_vx[true_idx] - self.est_vx[est_idx])**2 +
                (self.true_vy[true_idx] - self.est_vy[est_idx])**2 +
                (self.true_vz[true_idx] - self.est_vz[est_idx])**2
            )
            
            self.position_errors.append(pos_error)
            self.velocity_errors.append(vel_error)
            self.error_timestamps.append(timestamp)

    def calculate_statistics(self):
        """Calculate performance statistics"""
        if len(self.position_errors) == 0:
            return "Waiting for data..."
        
        stats = []
        stats.append(f"Total Samples: {len(self.position_errors)}")
        
        if len(self.position_errors) > 0:
            pos_errors_array = np.array(self.position_errors)
            vel_errors_array = np.array(self.velocity_errors)
            
            stats.append(f"Position RMSE: {np.sqrt(np.mean(pos_errors_array**2)):.3f} m")
            stats.append(f"Velocity RMSE: {np.sqrt(np.mean(vel_errors_array**2)):.3f} m/s")
            stats.append(f"Avg Position Error: {np.mean(pos_errors_array):.3f} m")
            stats.append(f"Avg Velocity Error: {np.mean(vel_errors_array):.3f} m/s")
            stats.append(f"Max Position Error: {np.max(pos_errors_array):.3f} m")
            if len(self.position_errors) > 0:
                stats.append(f"Current Error: {self.position_errors[-1]:.3f} m")
        
        return "\n".join(stats)

    def sync_data_for_plotting(self, true_data, est_data, true_times, est_times):
        """Synchronize data for plotting by finding common time points"""
        if len(true_data) == 0 or len(est_data) == 0:
            return [], [], []
        
        # Use the dataset with fewer points as reference
        if len(true_data) <= len(est_data):
            reference_data = true_data
            reference_times = true_times
            other_data = est_data
            other_times = est_times
        else:
            reference_data = est_data
            reference_times = est_times
            other_data = true_data
            other_times = true_times
        
        synced_ref = []
        synced_other = []
        synced_times = []
        
        for i in range(len(reference_data)):
            # For each reference point, find the closest other point in time
            ref_time = reference_times[i]
            
            # Find the closest other timestamp
            time_diffs = [abs(ref_time - t) for t in other_times]
            if time_diffs:
                closest_idx = np.argmin(time_diffs)
                
                # Only use if time difference is reasonable (less than 1 second)
                if time_diffs[closest_idx] < 1.0:
                    synced_ref.append(reference_data[i])
                    synced_other.append(other_data[closest_idx])
                    synced_times.append(ref_time - min(reference_times[0], other_times[0]))
        
        return synced_ref, synced_other, synced_times

    def update_plots(self):
        """Update all plots with new data"""
        if not self.has_true_data or not self.has_est_data:
            # Show waiting message
            self.ax_3d.clear()
            self.ax_3d.text(0.5, 0.5, 0.5, "Waiting for data...", 
                           ha='center', va='center', transform=self.ax_3d.transAxes)
            self.ax_3d.set_title('3D Trajectory - Waiting for data')
            plt.draw()
            plt.pause(0.01)
            return
        
        # Clear all plots
        self.ax_3d.clear()
        self.ax_error.clear()
        self.ax_vel_error.clear()
        self.ax_xyz.clear()
        
        # Set titles and labels
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_title('3D Trajectory - True vs Estimated')
        
        self.ax_error.set_xlabel('Time (s)')
        self.ax_error.set_ylabel('Position Error (m)')
        self.ax_error.set_title('Position Estimation Error')
        self.ax_error.grid(True)
        
        self.ax_vel_error.set_xlabel('Time (s)')
        self.ax_vel_error.set_ylabel('Velocity Error (m/s)')
        self.ax_vel_error.set_title('Velocity Estimation Error')
        self.ax_vel_error.grid(True)
        
        self.ax_xyz.set_xlabel('Time (s)')
        self.ax_xyz.set_ylabel('Position (m)')
        self.ax_xyz.set_title('XYZ Position Components - True vs Estimated')
        self.ax_xyz.grid(True)
        
        # Convert to lists for plotting
        true_x_list = list(self.true_x)
        true_y_list = list(self.true_y)
        true_z_list = list(self.true_z)
        est_x_list = list(self.est_x)
        est_y_list = list(self.est_y)
        est_z_list = list(self.est_z)
        
        # Plot 3D trajectories (plot whatever data we have)
        if len(true_x_list) > 1:
            self.ax_3d.plot(true_x_list, true_y_list, true_z_list, 'g-', linewidth=2, label='True Trajectory')
        if len(est_x_list) > 1:
            self.ax_3d.plot(est_x_list, est_y_list, est_z_list, 'b-', linewidth=2, label='Estimated Trajectory')
        
        # Plot current positions
        if len(true_x_list) > 0:
            self.ax_3d.scatter([true_x_list[-1]], [true_y_list[-1]], [true_z_list[-1]], 
                             c='green', s=100, marker='o', label='Current True')
        if len(est_x_list) > 0:
            self.ax_3d.scatter([est_x_list[-1]], [est_y_list[-1]], [est_z_list[-1]], 
                             c='blue', s=100, marker='s', label='Current Estimate')
        
        # Plot errors
        if len(self.position_errors) > 0 and len(self.error_timestamps) > 0:
            error_times_relative = [t - self.error_timestamps[0] for t in self.error_timestamps]
            self.ax_error.plot(error_times_relative, self.position_errors, 'r-', linewidth=2, label='Position Error')
            self.ax_vel_error.plot(error_times_relative, self.velocity_errors, 'r-', linewidth=2, label='Velocity Error')
        
        # Plot XYZ components with synchronized data
        if len(self.true_timestamps) > 0 and len(self.est_timestamps) > 0:
            # Synchronize X data
            true_x_sync, est_x_sync, time_x_sync = self.sync_data_for_plotting(
                true_x_list, est_x_list, list(self.true_timestamps), list(self.est_timestamps))
            
            if len(true_x_sync) > 0 and len(est_x_sync) > 0:
                self.ax_xyz.plot(time_x_sync, true_x_sync, 'r-', linewidth=1, label='True X', alpha=0.7)
                self.ax_xyz.plot(time_x_sync, est_x_sync, 'r--', linewidth=2, label='Est X')
            
            # Synchronize Y data
            true_y_sync, est_y_sync, time_y_sync = self.sync_data_for_plotting(
                true_y_list, est_y_list, list(self.true_timestamps), list(self.est_timestamps))
            
            if len(true_y_sync) > 0 and len(est_y_sync) > 0:
                self.ax_xyz.plot(time_y_sync, true_y_sync, 'g-', linewidth=1, label='True Y', alpha=0.7)
                self.ax_xyz.plot(time_y_sync, est_y_sync, 'g--', linewidth=2, label='Est Y')
            
            # Synchronize Z data
            true_z_sync, est_z_sync, time_z_sync = self.sync_data_for_plotting(
                true_z_list, est_z_list, list(self.true_timestamps), list(self.est_timestamps))
            
            if len(true_z_sync) > 0 and len(est_z_sync) > 0:
                self.ax_xyz.plot(time_z_sync, true_z_sync, 'b-', linewidth=1, label='True Z', alpha=0.7)
                self.ax_xyz.plot(time_z_sync, est_z_sync, 'b--', linewidth=2, label='Est Z')
        
        # Add legends only if we have data
        if len(true_x_list) > 0 or len(est_x_list) > 0:
            self.ax_3d.legend()
        
        if len(self.position_errors) > 0:
            self.ax_error.legend()
            self.ax_vel_error.legend()
        
        if any([len(self.true_x) > 0, len(self.true_y) > 0, len(self.true_z) > 0,
                len(self.est_x) > 0, len(self.est_y) > 0, len(self.est_z) > 0]):
            self.ax_xyz.legend()
        
        # Update statistics
        stats_text = self.calculate_statistics()
        self.ax_error.text(0.02, 0.98, stats_text, 
                          transform=self.ax_error.transAxes, 
                          verticalalignment='top', 
                          fontsize=8,
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Auto-scale
        self.ax_error.relim()
        self.ax_error.autoscale_view()
        self.ax_vel_error.relim()
        self.ax_vel_error.autoscale_view()
        self.ax_xyz.relim()
        self.ax_xyz.autoscale_view()
        
        plt.draw()
        plt.pause(0.01)

    def save_plots(self, filename="particle_filter_analysis.png"):
        """Save current plots to file"""
        self.update_plots()  # Ensure plots are up to date
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        self.get_logger().info(f"Plots saved to {filename}")

def main():
    rclpy.init()
    node = ParticleFilterVisualizer()
    
    try:
        # Use manual plotting updates instead of FuncAnimation
        print("Particle Filter Visualizer running...")
        print("Press Ctrl+C to save plots and exit")
        
        last_update = 0
        update_interval = 0.5  # Update plots every 0.5 seconds
        
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
            # Update plots at fixed interval
            current_time = time.time()
            if current_time - last_update > update_interval:
                node.update_plots()
                last_update = current_time
                
    except KeyboardInterrupt:
        print("\nSaving final plots...")
        node.save_plots()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.close('all')

if __name__ == "__main__":
    main()