#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData

class SensorSimNode(Node):
    def __init__(self):
        super().__init__('sensor_sim_node')
        
        # Subscriber to true drone state
        self.drone_state_sub = self.create_subscription(
            DroneState,
            '/drone/true_state',
            self.drone_state_callback,
            10
        )
        
        # Publishers for sensor data
        self.doa_pub = self.create_publisher(DOAData, '/sensors/doa', 10)
        self.pp_pub = self.create_publisher(PointPillarsData, '/sensors/point_pillars', 10)
        
        # System position (sensor origin)
        self.system_position = np.array([0, 0, 0])
        
        # Noise parameters
        self.doa_noise_level = 5.0  # changed from 2 degrees
        self.pp_noise_std = 1 # changed from 0.5 meters
        
        self.get_logger().info("Sensor simulation node started")

    def generate_doa_data(self, true_position, noise_level=2.0):
        """Generate simulated DOA sensor data with noise"""
        vector = true_position - self.system_position
        distance = np.linalg.norm(vector)
        
        if distance > 0:
            direction = vector / distance
            azimuth = np.arctan2(direction[1], direction[0])
            elevation = np.arcsin(direction[2])
            
            azimuth += np.random.normal(0, np.deg2rad(noise_level))
            elevation += np.random.normal(0, np.deg2rad(noise_level))
            
            return np.array([np.rad2deg(azimuth), np.rad2deg(elevation)])
        else:
            return np.array([0, 90])

    def generate_pp_position(self, true_position, noise_std=0.5):
        """Generate simulated PointPillars position data with noise"""
        return true_position + np.random.normal(0, noise_std, 3)

    def drone_state_callback(self, msg):
        """Callback when new drone state is received"""
        # Extract true position
        true_position = np.array([
            msg.true_position.x,
            msg.true_position.y, 
            msg.true_position.z
        ])
        
        # Generate DOA data
        doa_angles = self.generate_doa_data(true_position, self.doa_noise_level)
        
        # Publish DOA data
        doa_msg = DOAData()
        doa_msg.header.stamp = msg.header.stamp
        doa_msg.header.frame_id = "sensor_frame"
        doa_msg.azimuth = float(doa_angles[0])
        doa_msg.elevation = float(doa_angles[1])
        self.doa_pub.publish(doa_msg)
        
        # Generate and publish PointPillars data
        pp_position = self.generate_pp_position(true_position, self.pp_noise_std)
        
        pp_msg = PointPillarsData()
        pp_msg.header.stamp = msg.header.stamp
        pp_msg.header.frame_id = "sensor_frame"
        pp_msg.position.x = float(pp_position[0])
        pp_msg.position.y = float(pp_position[1])
        pp_msg.position.z = float(pp_position[2])
        pp_msg.confidence = 0.95  # Simulated confidence
        self.pp_pub.publish(pp_msg)
        
        self.get_logger().debug(f"Published sensor data - DOA: {doa_angles}")

def main():
    rclpy.init()
    node = SensorSimNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()