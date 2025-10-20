# Chimera
For Stage simulator installation please run the following command
```bash
sudo apt-get install git cmake g++ libjpeg8-dev libpng-dev libglu1-mesa-dev libltdl-dev libfltk1.1-dev
cd WORKSPACE/src/Stage/
mkdir build && cd build
cmake ..
make && sudo make install
```
The last step to install Stage would be compiling it from the workspace folder

```bash
cd WORKSPACE/
colcon build --packages-select Stage
```
Now compile the stage_ros2 for humble version

```bash
colcon build --packages-select stage_ros2
```
Now we installed the package which interlinks Stage and ROS2 Humble.
