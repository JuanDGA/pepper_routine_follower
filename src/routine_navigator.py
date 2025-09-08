#!/usr/bin/env python
import rospy
from robot_toolkit_msgs.srv import navigation_tools_srv, navigation_tools_srvRequest
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import time
import numpy as np

def get_inverse(q: Quaternion):
    norm_squared = q.w**2 + q.x**2 + q.y**2 + q.z**2

    # Check for a zero norm to prevent division by zero
    if norm_squared == 0:
        # A zero quaternion has no inverse; return a zero quaternion
        return Quaternion(0, 0, 0, 0)

    # Calculate the inverse components
    inv_x = -q.x / norm_squared
    inv_y = -q.y / norm_squared
    inv_z = -q.z / norm_squared
    inv_w = q.w / norm_squared

    return Quaternion(inv_x, inv_y, inv_z, inv_w)


def multiply_quaternions(a: Quaternion, b: Quaternion):
        """
        Multiplies this quaternion by another quaternion.
        """
        w1, x1, y1, z1 = a.w, a.x, a.y, a.z
        w2, x2, y2, z2 = b.w, b.x, b.y, b.z

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return Quaternion(x, y, z, w)

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0
        self.previous_error = 0

    def calculate(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class RoutineNavigator:
    odometry_info: Odometry
    calibrated = False
        
    def __init__(self):
        rospy.init_node("routine_navigator")

        self.odometry_info = None
        self._init_navigation()
        
        input("Press enter to set the origin...")
        
        origin_odom = self.odometry_info
        
        self.origin_ref = np.array([origin_odom.pose.pose.position.x, origin_odom.pose.pose.position.y])
        self.rotation_bias = origin_odom.pose.pose.orientation
        
        y_change = float(input("Mueve el robot en el eje Y de referencia, indica cuantos metros en el eje Y cambiaste:").strip())
        y_odom = self.odometry_info
        y_ref = np.array([y_odom.pose.pose.position.x, y_odom.pose.pose.position.y])
        
        x_change = float(input("Devuelve el robot a donde estaba lo más exacto posible, luego mueve el robot en el eje X de referencia e indica cuantos metros en el eje X cambiaste:").strip())
        x_odom = self.odometry_info
        x_ref = np.array([x_odom.pose.pose.position.x, x_odom.pose.pose.position.y])
        
        print("Calculando transformación lineal...")
        
        V = np.column_stack((x_ref - self.origin_ref, y_ref - self.origin_ref))
        V_prime = np.column_stack((np.array([x_change, 0]), np.array([0, y_change])))
        
        self.A = V_prime @ np.linalg.inv(V)
        
        input("Calibrated, press enter to continue...")
        self.calibrated = True
        
        self._init_topic()
        
    def _init_navigation(self):
        print("Initilazing Navigation...")
        rospy.wait_for_service("/robot_toolkit/navigation_tools_srv")
        self.navigation_tools_srv = rospy.ServiceProxy("/robot_toolkit/navigation_tools_srv", navigation_tools_srv)
        
        init_msg = navigation_tools_srvRequest()
        init_msg.data.command = "enable_all"

        init_msg.data.depth_to_laser_parameters.resolution = 0
        init_msg.data.depth_to_laser_parameters.scan_time = 0.0
        init_msg.data.depth_to_laser_parameters.range_min = 0.0
        init_msg.data.depth_to_laser_parameters.range_max = 0.0
        init_msg.data.depth_to_laser_parameters.scan_height = 0.0

        init_msg.data.tf_enable = False
        init_msg.data.tf_frequency = 0.0

        init_msg.data.odom_enable = False
        init_msg.data.odom_frequency = 0.0

        init_msg.data.laser_enable = False
        init_msg.data.laser_frequency = 0.0

        init_msg.data.cmd_vel_enable = False
        init_msg.data.security_timer = 0.0

        init_msg.data.move_base_enable = False
        init_msg.data.goal_enable = False
        init_msg.data.robot_pose_suscriber_enable = False

        init_msg.data.path_enable = False
        init_msg.data.path_frequency = 0.0

        init_msg.data.robot_pose_publisher_enable = False
        init_msg.data.robot_pose_publisher_frequency = 0.0

        init_msg.data.result_enable = False
        init_msg.data.depth_to_laser_enable = False
        init_msg.data.free_zone_enable = False
        
        self.navigation_tools_srv(init_msg)
        print("Navigation has been initialized correctly")
        
        rospy.Subscriber("/odom", Odometry, self.read_odom) 
        
        while self.odometry_info is None:
            time.sleep(0.1)
        
    def read_odom(self, value: Odometry):
        if self.calibrated:
            current = np.array([value.pose.pose.position.x, value.pose.pose.position.y])
            transformed = self.A @ (current - self.origin_ref)
            value.pose.pose.position.x = float(transformed[0])
            value.pose.pose.position.y = float(transformed[1])
            value.pose.pose.orientation = multiply_quaternions(get_inverse(self.rotation_bias), value.pose.pose.orientation)
        self.odometry_info = value
            
        
    def _init_topic(self):  
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        print("Moving in ten seconds...")
        
        self.current_position = self.odometry_info
        
        self.line = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)]
        self.target_index = 1
        
        self.linear_pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
        self.angular_pid = PIDController(kp=1.0, ki=0.01, kd=0.1)
        
        rospy.sleep(10)
        
        while not rospy.is_shutdown() and self.target_index < len(self.line):
            current = self.odometry_info
            self.move_to_point(current)
            rospy.sleep(0.5)

        print("Routine complete.")

    def move_to_point(self, odometry_info: Odometry):
        current_x = odometry_info.pose.pose.position.x
        current_y = odometry_info.pose.pose.position.y
        quaternion = (
            odometry_info.pose.pose.orientation.x,
            odometry_info.pose.pose.orientation.y,
            odometry_info.pose.pose.orientation.z,
            odometry_info.pose.pose.orientation.w,
        )
        _, _, current_yaw = euler_from_quaternion(quaternion)
        print(f"Currently at ({current_x}, {current_y})")
        target_point = self.line[self.target_index]
        print(f"Going to ({target_point[0]}, {target_point[1]})")
        dx = target_point[0] - current_x
        dy = target_point[1] - current_y
        distance_to_target = math.sqrt(dx**2 + dy**2)

        if distance_to_target < 0.1:
            self.target_index += 1
            if self.target_index < len(self.line):
                print(f"Reached point. Moving to {self.line[self.target_index]}")
            return

        target_angle = math.atan2(dy, dx)
        angular_error = target_angle - current_yaw

        if angular_error > math.pi:
            angular_error -= 2 * math.pi
        elif angular_error < -math.pi:
            angular_error += 2 * math.pi

        message = Twist()
        
        # Prioritize rotation until aligned
        if abs(angular_error) > 0.1:  # A small tolerance for alignment
            angular_speed = self.angular_pid.calculate(angular_error, 0.5)
            message.angular.z = min(max(angular_speed, -0.5), 0.5)
            message.linear.x = 0.0  # Stop linear movement while rotating
        else:
            # Once aligned, move forward
            linear_speed = self.linear_pid.calculate(distance_to_target, 0.5)
            message.linear.x = min(max(linear_speed, 0.0), 0.5) # Ensure it only moves forward
            message.angular.z = 0.0
        
        self.pub.publish(message)

if __name__ == "__main__":
    RoutineNavigator()