#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
from uav_interfaces.msg import DroneState
import time

class UAVDynamicModel:
    """Simple dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        self.max_speed = 500.0
        self.max_acceleration = 100.0
        self.max_vertical_speed = 200.0
        self.max_vertical_acceleration = 50.0
        self.velocity_decay = 1
        self.position_noise_std = 0.1

    def constrain_velocity(self, velocity):
        horizontal_speed = np.linalg.norm(velocity[:2])
        vertical_speed = abs(velocity[2])
        
        if horizontal_speed > self.max_speed:
            scale = self.max_speed / horizontal_speed
            velocity[0] *= scale
            velocity[1] *= scale
        
        if vertical_speed > self.max_vertical_speed:
            velocity[2] = np.sign(velocity[2]) * self.max_vertical_speed
            
        return velocity
    
    def constrain_acceleration(self, acceleration):
        horizontal_accel = np.linalg.norm(acceleration[:2])
        vertical_accel = abs(acceleration[2])
        
        if horizontal_accel > self.max_acceleration:
            scale = self.max_acceleration / horizontal_accel
            acceleration[0] *= scale
            acceleration[1] *= scale
        
        if vertical_accel > self.max_vertical_acceleration:
            acceleration[2] = np.sign(acceleration[2]) * self.max_vertical_acceleration
            
        return acceleration
    
    def update_state(self, position, velocity, dt=0.1):
        random_accel = np.random.normal(0, 0.5, 3)
        random_accel = self.constrain_acceleration(random_accel)
        
        velocity = velocity * self.velocity_decay + random_accel * dt
        velocity = self.constrain_velocity(velocity)
        
        new_position = position + velocity * dt
        position_noise = np.random.normal(0, self.position_noise_std, 3)
        new_position += position_noise

        return new_position, velocity

class DroneSimNode(Node):
    def __init__(self):
        super().__init__('drone_sim_node')
        
        # Publishers
        self.drone_state_pub = self.create_publisher(DroneState, '/drone/true_state', 10)
        
        # Timer for simulation updates
        self.timer = self.create_timer(0.1, self.update_simulation)  # 10 Hz
        
        # Drone state variables
        self.dynamic_model = UAVDynamicModel()
        self.origin = np.array([0, 0, 0])
        self.despawn_distance = 0.5  # meters
        self.respawn_delay = 5.0  # seconds
        self.spawn_distance = 100.0  # meters
        
        # Initialize drone state
        self.true_position, self.true_velocity = self.spawn_drone()
        self.drone_active = True
        self.despawn_time = None

    def spawn_drone(self):
        """Spawn drone at random location 100m from origin"""
        # Generate random direction vector
        direction = np.random.uniform(-1, 1, 3)
        direction = direction / np.linalg.norm(direction)  # Normalize
        
        # Position at 100m distance
        spawn_position = direction * self.spawn_distance
        
        # Velocity towards origin
        velocity_direction = -direction  # Towards origin
        initial_speed = np.random.uniform(3, 8)  # Random speed between 3-8 m/s
        spawn_velocity = velocity_direction * initial_speed
        
        return spawn_position, spawn_velocity

    def update_drone_dynamics(self, true_position, true_velocity, dt=0.1):
        """Update drone position and maintain direction towards origin"""
        true_position, true_velocity = self.dynamic_model.update_state(true_position, true_velocity, dt)
        
        # Maintain consistent direction toward origin
        direction_to_origin = self.origin - true_position
        direction_to_origin = direction_to_origin / np.linalg.norm(direction_to_origin)
        true_velocity = direction_to_origin * np.linalg.norm(true_velocity)
        
        return true_position, true_velocity

    def check_despawn_condition(self, position):
        """Check if drone has reached origin (within despawn distance)"""
        distance_to_origin = np.linalg.norm(position)
        return distance_to_origin <= self.despawn_distance

    def update_simulation(self):
        """Update drone state and handle spawn/despawn logic"""
        current_time = time.time()
        
        if self.drone_active:
            # Update drone dynamics
            self.true_position, self.true_velocity = self.update_drone_dynamics(
                self.true_position, self.true_velocity
            )
            
            # Check if drone should despawn
            if self.check_despawn_condition(self.true_position):
                self.drone_active = False
                self.despawn_time = current_time
                return
            
            # Publish active drone state
            self.publish_drone_state()
            
        else:
            # Drone is despawning, check if respawn time has elapsed
            if current_time - self.despawn_time >= self.respawn_delay:
                self.true_position, self.true_velocity = self.spawn_drone()
                self.drone_active = True
                self.despawn_time = None
                self.publish_drone_state()

    def publish_drone_state(self):
        """Publish current drone state"""
        msg = DroneState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        msg.true_position.x = float(self.true_position[0])
        msg.true_position.y = float(self.true_position[1])
        msg.true_position.z = float(self.true_position[2])
        
        msg.true_velocity.x = float(self.true_velocity[0])
        msg.true_velocity.y = float(self.true_velocity[1])
        msg.true_velocity.z = float(self.true_velocity[2])
        
        self.drone_state_pub.publish(msg)

def main():
    rclpy.init()
    node = DroneSimNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()