# Install script for directory: /home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/arrk-adas/Desktop/Rc-Car-Chimera/install/Stage")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "RELEASE")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/stage/worlds" TYPE FILE FILES
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/amcl-sonar.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/autolab.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/camera.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/everything.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/lsp_test.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/mbicp.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/nd.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/roomba.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/simple.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/test.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/uoa_robotics_lab.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/vfh.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wavefront-remote.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wavefront.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wifi.cfg"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/SFU.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/autolab.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/camera.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/circuit.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/everything.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/fasr.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/fasr2.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/fasr_plan.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/large.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/lsp_test.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/mbicp.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer_flocking.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer_follow.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer_walle.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/roomba.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/sensor_noise_demo.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/sensor_noise_module_demo.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/simple.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/uoa_robotics_lab.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wifi.world"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/beacons.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/chatterbox.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/hokuyo.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/irobot.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/map.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/objects.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pantilt.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/sick.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/ubot.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/uoa_robotics_lab_models.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/walle.inc"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/cfggen.sh"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/test.sh"
    "/home/arrk-adas/Desktop/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/worldgen.sh"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/arrk-adas/Desktop/Rc-Car-Chimera/build/Stage/worlds/benchmark/cmake_install.cmake")
  include("/home/arrk-adas/Desktop/Rc-Car-Chimera/build/Stage/worlds/bitmaps/cmake_install.cmake")
  include("/home/arrk-adas/Desktop/Rc-Car-Chimera/build/Stage/worlds/wifi/cmake_install.cmake")

endif()

