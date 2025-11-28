#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header
from uav_interfaces.msg import DroneState

class UAVDynamicModel:
    """Simple dynamic model for DJI Mavic-like UAV"""
    def __init__(self):
        self.max_speed = 500.0 # cm/s
        self.max_acceleration = 100.0 
        self.max_vertical_speed = 200.0
        self.max_vertical_acceleration = 50.0
        self.velocity_decay = 1
        self.position_noise_std = 0.1

#    def __init__(self):
#        self.max_speed = 5
#        self.max_acceleration = 10.0
#        self.max_vertical_speed = 22
#        self.max_vertical_acceleration = 5.0
#        self.velocity_decay = 0.98
#        self.position_noise_std = 0.1
#        self.position_noise_std = 0.05




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
        
        # Initialize drone state
        self.dynamic_model = UAVDynamicModel()
        self.target_point = np.array([-100, 5, 0])
        self.true_position, self.true_velocity = self.initialize_drone_position(self.target_point)
        
        self.get_logger().info("Drone simulation node started")

    def initialize_drone_position(self, target_point, initial_distance_range=(30, 70)):
        """Initialize drone position and velocity towards target point"""
        true_position = np.random.uniform(initial_distance_range[0], initial_distance_range[1], 3)
        direction_to_target = target_point - true_position
        direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
        true_velocity = direction_to_target * 5  # Scale to desired speed
        
        return true_position, true_velocity

    def update_drone_dynamics(self, true_position, true_velocity, target_point, dt=0.1):
        """Update drone position and maintain direction towards target"""
        true_position, true_velocity = self.dynamic_model.update_state(true_position, true_velocity, dt)
        
        # Maintain consistent direction toward target
        direction_to_target = target_point - true_position
        direction_to_target = direction_to_target / np.linalg.norm(direction_to_target)
        true_velocity = direction_to_target * np.linalg.norm(true_velocity)
        
        return true_position, true_velocity

    def update_simulation(self):
        """Update drone state and publish"""
        # Update dynamics
        self.true_position, self.true_velocity = self.update_drone_dynamics(
            self.true_position, self.true_velocity, self.target_point
        )
        
        # Create and publish message
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
        
        self.get_logger().debug(f"Published drone state: {self.true_position}")

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
