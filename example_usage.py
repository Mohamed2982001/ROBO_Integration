"""
example_usage.py
Run: python example_usage.py <robot_ip>
The ESP32 now joins your local Wi-Fi (router) instead of hosting its own
access point, so make sure your computer is on the same network. Find the
robot's IP in the Arduino Serial Monitor after boot - it prints
"Robot IP address: x.x.x.x". Consider reserving that IP as static in your
router's DHCP settings so it doesn't change between reboots.
"""

import sys
import time
from face_tracker import FaceTracker, FaceTrackerConfig
from robot_client import Robot

# if len(sys.argv) < 2:
#     print("Usage: python example_usage.py <robot_ip>")
#     sys.exit(1)

# HOST = sys.argv[1]

# with Robot(HOST) as bot:               # stop() is called automatically on exit
#     print("Ping:", bot.ping())

# the ip address of the robot and the camera url they are changed based on the wifi network and the camera used. The camera url is the url of the camera stream. in the app 
bot = Robot("192.168.1.18")
# tracker = FaceTracker(bot, camera_url="http://admin:12345678@192.168.1.2:8081/video")
# tracker.run()   # blocks; press ESC in the preview window to stop

#     # Chassis: basic + diagonal directions
bot.set_speed(200)                 # 0-255
# bot.forward(); time.sleep(0.5)
# bot.strafe_right(); time.sleep(0.5)
# bot.stop()
# bot.diagonal_backward_left(); time.sleep(0.5)
# bot.rotate_left(); time.sleep(0.5)
# bot.stop()
try:
    print("Robot script running... Press Ctrl+C at any time to interrupt!")
    
    # Chassis: basic + diagonal directions
    bot.forward()
    time.sleep(2)
    
    bot.strafe_right()
    time.sleep(2)
    bot.backward()
    time.sleep(2)
    bot.strafe_left()
    time.sleep(2)
    bot.stop()

except KeyboardInterrupt:
    print("\n[!] Execution interrupted by user (Ctrl+C)!")

finally:
    # This block GUARANTEES the robot doesn't get stuck running away or frozen mid-wave
    print("Cleaning up: Stopping chassis and centering all servos...")
    bot.stop()              # Fixed the missing () so the wheels actually stop!
    bot.center_all_servos() # Safely returns shoulders, elbow, and head back to home
    print("Robot safely disarmed.")

# bot.move("forward_right")
# time.sleep(0.3)
# bot.stop()
# bot.move_for("backward", duration=1.0, block=False)   # returns immediately


# bot.set_servo(0, 90)
# bot.set_servo(7, 60)              # head horizontal to 100 degrees


# bot.set_servos({0: 90, 1: 100, 7: 90})   # sent together in one request
# bot.move_group(channels=[1, 3], angle=110)  # both arm-rotate servos together


# bot.open_gripper()
# time.sleep(0.5)
# bot.close_gripper()
# time.sleep(0.5)



# print("Available poses:", bot.list_poses())
# bot.set_pose("arms_up")
# time.sleep(1)
# bot.set_pose("home")

#     # Save your own pose from current state, export/import to a JSON file
# bot.save_pose("my_custom_pose", {0: 60, 7: 100})
# bot.export_poses("my_poses.json")
#     # bot.import_poses("my_poses.json")


# bot.wave(channel=1)
# bot.look_around()
# bot.sweep_servo(channel=0, start=45, end=135, step=10)

# bot.square_patrol(chasiss_duration=0.8)

# bot.move_group(channels=[0, 7], angle=90)  # move head to center
