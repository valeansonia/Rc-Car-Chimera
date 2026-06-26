# Install script for directory: /media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/install/Stage")
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
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/amcl-sonar.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/autolab.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/camera.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/everything.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/lsp_test.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/mbicp.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/nd.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/roomba.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/simple.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/test.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/uoa_robotics_lab.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/vfh.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wavefront-remote.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wavefront.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wifi.cfg"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/SFU.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/autolab.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/camera.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/circuit.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/everything.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/fasr.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/fasr2.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/fasr_plan.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/large.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/lsp_test.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/mbicp.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer_flocking.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer_follow.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer_walle.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/roomba.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/sensor_noise_demo.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/sensor_noise_module_demo.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/simple.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/uoa_robotics_lab.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/wifi.world"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/beacons.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/chatterbox.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/hokuyo.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/irobot.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/map.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/objects.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pantilt.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/pioneer.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/sick.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/ubot.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/uoa_robotics_lab_models.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/walle.inc"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/cfggen.sh"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/test.sh"
    "/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/Rc-Car/src/Stage/worlds/worldgen.sh"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/build/Stage/worlds/benchmark/cmake_install.cmake")
  include("/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/build/Stage/worlds/bitmaps/cmake_install.cmake")
  include("/media/arrk-adas/RC-Env/RC-car-project/Rc-Car-Chimera/build/Stage/worlds/wifi/cmake_install.cmake")

endif()

