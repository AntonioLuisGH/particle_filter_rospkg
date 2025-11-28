from setuptools import setup
import os
from glob import glob

package_name = 'counter_uav_system'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='UAV Counter System Simulation',
    license='TODO: License declaration',
    
    # You can simply replace tests_require with extras_require here:
    extras_require={
        'test': [
            'pytest',
        ],
    },
    # tests_require=['pytest'],  <-- You can remove this or keep it as a fallback
    
    entry_points={
        'console_scripts': [
            'drone_sim_node = counter_uav_system.drone_sim_node:main',
            'sensor_sim_node = counter_uav_system.sensor_sim_node:main',
            'particle_filter_node = counter_uav_system.particle_filter_node:main',
            'data_plotter = counter_uav_system.Data_plotter:main',
        ],
    },
)