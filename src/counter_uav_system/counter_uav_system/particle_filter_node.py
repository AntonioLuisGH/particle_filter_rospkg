#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Point
from uav_interfaces.msg import DroneState, DOAData, PointPillarsData, ParticleFilterState
from visualization_msgs.msg import Marker

class UAVDynamicModel:
    def __init__(self):
        self.max_speed = 50.0 
        self.velocity_decay = 0.9
        self.position_noise_std = 0.1
    
    def constrain_velocity(self, velocity):
        return np.clip(velocity, -self.max_speed, self.max_speed)
    
    def update_state(self, position, velocity, dt=0.1):
        velocity = self.constrain_velocity(velocity * self.velocity_decay)
        new_position = position + velocity * dt
        return new_position, velocity

class SensorParticleFilter:
    def __init__(self, system_position, num_particles=1000):
        self.system_position = system_position
        self.num_particles = num_particles
        self.dynamic_model = UAVDynamicModel()
        self.particles = None
        self.velocities = None
        self.weights = None
        
    def initialize_particles(self, initial_positions, initial_velocities):
        self.particles = np.array(initial_positions)
        self.velocities = np.array(initial_velocities)
        self.weights = np.ones(self.num_particles) / self.num_particles
        
    def predict(self, dt=0.1):
        noise = np.random.normal(0, 0.1, (self.num_particles, 3))
        self.particles += (self.velocities * dt) + noise
            
    def resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        step = 1.0 / self.num_particles
        start = np.random.uniform(0, step)
        positions = np.arange(self.num_particles) * step + start
        
        indices = np.zeros(self.num_particles, dtype=int)
        i, j = 0, 0
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        
        self.particles = self.particles[indices]
        self.velocities = self.velocities[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_position(self):
        if self.particles is not None:
            return np.average(self.particles, weights=self.weights, axis=0)
        return np.array([0., 0., 0.])
    
    def estimate_velocity(self):
        if self.velocities is not None:
            return np.average(self.velocities, weights=self.weights, axis=0)
        return np.array([0., 0., 0.])

class DOAParticleFilter(SensorParticleFilter):
    def __init__(self, system_position, num_particles=500):
        super().__init__(system_position, num_particles)
    def initialize_from_doa(self, doa_data): pass
    def update_weights(self, doa_data): pass

class PPParticleFilter(SensorParticleFilter):
    def __init__(self, system_position, num_particles=500):
        super().__init__(system_position, num_particles)
        self.position_std = 1.0
        
    def initialize_from_pp(self, pp_position):
        positions = []
        velocities = []
        for _ in range(self.num_particles):
            pos = pp_position + np.random.normal(0, 2.0, 3)
            vel = np.random.uniform(-1, 1, 3)
            positions.append(pos)
            velocities.append(vel)
        self.initialize_particles(positions, velocities)

    def update_weights(self, pp_position):
        dists = np.linalg.norm(self.particles - pp_position, axis=1)
        self.weights *= np.exp(-0.5 * (dists / self.position_std) ** 2)
        self.weights += 1.e-300 
        self.weights /= np.sum(self.weights)

class MultiSensorParticleFilter:
    def __init__(self, system_position=np.array([0, 0, 0])):
        self.pp_filter = PPParticleFilter(system_position)

    def process_pp_measurement(self, pp_data):
        if self.pp_filter.particles is None:
            self.pp_filter.initialize_from_pp(pp_data)
        else:
            self.pp_filter.predict()
            self.pp_filter.update_weights(pp_data)
            self.pp_filter.resample()

    def estimate_position(self):
        return self.pp_filter.estimate_position()

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')
        
        self.declare_parameter('global_frame', 'world')
        self.global_frame = self.get_parameter('global_frame').get_parameter_value().string_value
        
        self.particle_filter = MultiSensorParticleFilter()
        self.latest_pp_data = None
        self.latest_true_state = None
        
        # Subscribers
        self.pp_sub = self.create_subscription(
            PointPillarsData, '/sensors/point_pillars', self.pp_callback, 10)
        
        self.true_state_sub = self.create_subscription(
            DroneState, '/drone/true_state', self.true_state_callback, 10)
        
        # Publishers
        self.vis_pub = self.create_publisher(Marker, '/filter/visualization_marker', 10)
        self.aiming_pub = self.create_publisher(Point, '/cmd_point', 10) # <--- RESTORED THIS
        
        self.timer = self.create_timer(0.05, self.process_measurements)
        self.get_logger().info(f"Particle Filter Node started in frame: {self.global_frame}")

    def pp_callback(self, msg):
        self.latest_pp_data = np.array([msg.position.x, msg.position.y, msg.position.z])

    def true_state_callback(self, msg):
        self.latest_true_state = np.array([msg.true_position.x, msg.true_position.y, msg.true_position.z])

    def process_measurements(self):
        if self.latest_pp_data is None:
            self.get_logger().warn("WAITING FOR DATA: No PointPillars data received yet on /sensors/point_pillars", throttle_duration_sec=2.0)
        else:
            self.particle_filter.process_pp_measurement(self.latest_pp_data)
            
        estimated_position = self.particle_filter.estimate_position()
        
        # 1. Publish Aiming Command (RESTORED)
        aiming_msg = Point()
        aiming_msg.x = float(estimated_position[0])
        aiming_msg.y = float(estimated_position[1])
        aiming_msg.z = float(estimated_position[2])
        self.aiming_pub.publish(aiming_msg)
        
        # 2. Publish Visualization
        self.publish_markers(estimated_position, self.latest_true_state)

    def publish_markers(self, estimated_pos, true_pos):
        marker = Marker()
        marker.header.frame_id = self.global_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "filter_estimate"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(estimated_pos[0])
        marker.pose.position.y = float(estimated_pos[1])
        marker.pose.position.z = float(estimated_pos[2])
        marker.scale.x = 0.5 
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        # Green color
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0 
        
        self.vis_pub.publish(marker)

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

if __name__ == "__main__":
    main()