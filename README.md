# Mobile Robot Controlled by Hand Gestures Using ToF Sensor

A research project demonstrating intuitive hand gesture control for mobile robots using Time-of-Flight (ToF) sensors and computer vision. 

Code of this project is in the file spresense and the Powerpoint for presentation is uploaded as pdf with pictures of the whole system.

**Developed by:** Zekang Yu   
**Institution:** Tadokoro Lab, Sony SS Report 2022-02-28  


---

## Overview

This project implements a mobile robot control system that enables natural, contactless interaction through hand gestures. By combining ToF sensor technology with MediaPipe hand recognition and ROS-based robot control, the system provides an intuitive interface for robot navigation and obstacle avoidance.

---

## Motivation

The system addresses three key application areas:

- **Industry:** Lighten worker burden and advance intelligent manufacturing through gesture-controlled systems
- **Public Health:** Reduce direct contact between people and machines using contactless control
- **Human-Robot Interaction:** Create robots capable of natural interaction with humans

---

## Key Features

### Hand Gesture Recognition
- Real-time hand tracking and joint detection using MediaPipe
- Multiple gesture types supported:
  - **Open hand movement:** Intuitive directional control
  - **Closed hand/hand removal:** Emergency stop
  - **Finger rotation (clockwise/counter-clockwise):** In-place robot rotation
  - **Finger tilt:** Directional movement with angular control

### Movement Control
- **Linear Movement:** Vector-based velocity control calculated from hand position relative to screen center
- **Angular Movement:** Rotation control based on finger orientation
- **Velocity Normalization:** Smooth translation of hand movements to robot commands

### Obstacle Avoidance
- Dual LiDAR system for 360° environmental awareness
- Real-time collision avoidance with repulsive velocity calculations
- Seamless integration of hand commands with safety constraints

### Robust Performance
- Functions in low-light and dark environments
- High-speed, low-power operation using VSP (Visual Sensing Processor)

---

## System Architecture

### Hardware Components
- **VSP (Visual Sensing Processor):** High-speed, low-power visual processing
- **ToF Sensor:** Depth image and point cloud data capture
- **Dual LiDAR System:** Environmental scanning and obstacle detection
- **Mecanum Wheels:** Omnidirectional movement capability
- **Main Processing Unit:** ROS-based control system

### Software Stack
- **ROS (Robot Operating System):** System integration and communication
- **MediaPipe:** Hand recognition and gesture classification
- **OpenCV:** Image processing and visualization
- **ROS-bridge API:** Point cloud data publishing

### System Flow
```
ToF Sensor → Depth Image → Hand Recognition → Hand Command
                                                      ↓
LiDAR → Scan Points → Collision Avoidance ← Hand Command
                            ↓
                      Final Command → Robot
```

---

## Robot Design

The mobile robot features:
- Mecanum wheel configuration for omnidirectional movement
- Dual LiDAR placement for comprehensive obstacle detection
- Top-mounted ToF sensor for optimal hand tracking
- VSP module for efficient processing
- Integrated battery and PC mounting position

---

## Control Algorithms

### Linear Velocity Calculation
```
Vector AB: From screen center (B) to hand center (A)
Vx = -(|AB| × sin(α)) / N
Vy = -(|AB| × cos(α)) / N
```
Where α is the vector angle and N is the normalization factor.

### Angular Velocity Calculation
```
Vector F: From index finger base (D) to fingertip (C)
Vz = β
```
Where β is the finger orientation angle.

---

## Applications

### Demonstrated Use Cases
1. **Contactless Warehouse Delivery:** Gesture-controlled material transport
2. **Library Book Transfer:** Autonomous mobile robot navigation in public spaces
3. **Autonomous Tennis Ball Collection:** Sports equipment management

---

## Experimental Results

The system successfully demonstrated:
- ✅ Circular path navigation using hand position control
- ✅ Clockwise and counter-clockwise turning via finger movements
- ✅ Precise angular control through finger orientation
- ✅ Real-time obstacle avoidance and collision prevention
- ✅ Operation in dark environments

---

## Technical Specifications

- **Hand Recognition Library:** MediaPipe Hands
- **Gesture Recognition:** Custom model training based on MediaPipe landmarks
- **Communication:** ROS-bridge API
- **Sensor Data:** Point cloud and depth image streams
- **Control Method:** Vector-based velocity commands with normalization

---

## References

- MediaPipe Hands: [https://google.github.io/mediapipe/solutions/hands.html](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html)

- Industry Applications: Various robotics and automation sources

---

## Future Work

Potential extensions of this system include:
- Multi-robot coordination through gesture control
- Advanced gesture vocabulary for complex commands
- Integration with additional sensors for enhanced perception
- Machine learning-based gesture customization

---

## Acknowledgments

Special thanks to the instructors and Tadokoro Lab for guidance and support throughout this research project.

---

## License

This project was developed as part of academic research at Tadokoro Lab. Please contact the authors for licensing information.
