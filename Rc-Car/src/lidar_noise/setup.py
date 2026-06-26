from setuptools import setup

package_name = 'lidar_noise'

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
    maintainer='sabeehmusharraf',
    maintainer_email='sabeeh.musharraf@arrk-engineering.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    #tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 'Lidar_noise = lidar_noise.Lidar_noise:main',
        ],
        'console_scripts': [ 'Lidar_scan_split = lidar_noise.Lidar_scan_split:main',
        ],
    },
)