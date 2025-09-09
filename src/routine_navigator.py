#!/usr/bin/env python
import rospy
from robot_toolkit_msgs.srv import navigation_tools_srv, navigation_tools_srvRequest
from geometry_msgs.msg import Twist, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation


MAX_PEPPER_SPEED = 0.5
HALF_AROUND = 0.95
STEEP = 20

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


def norm_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def is_closer_left(target_angle: float, pepper_angle: float) -> bool:
    return norm_angle(target_angle - pepper_angle) > 0



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
    
    
def get_matrix_transformation(q: Quaternion, b_xyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.array([q.x, q.y, q.z, q.w])
    b_xyz = np.asarray(b_xyz)
    
    # 1. Crear el objeto de rotación a partir del cuaternión.
    # SciPy usa el formato [x, y, z, w] por defecto.
    rotacion_q = Rotation.from_quat(q_xyzw)

    # 2. Calcular la rotación inversa. Para un cuaternión unitario, el inverso
    # es igual a su conjugado. El método inv() lo maneja correctamente.
    rotacion_inversa = rotacion_q.inv()

    # 3. Obtener la matriz de rotación de 3x3 de la rotación inversa.
    matriz_rotacion_3x3 = rotacion_inversa.as_matrix()

    # 4. Calcular el vector de traslación.
    # Es -R_inv * B, lo que significa que rotamos el vector -B.
    vector_traslacion_3d = -matriz_rotacion_3x3 @ b_xyz

    # 5. Ensamblar la matriz de transformación homogénea de 4x4.
    matriz_transformacion = np.eye(4)
    matriz_transformacion[0:3, 0:3] = matriz_rotacion_3x3
    matriz_transformacion[0:3, 3] = vector_traslacion_3d

    return matriz_transformacion


def transform_strength(strength: float) -> float:
    return 1 - 1 / (1 + math.exp(-STEEP * (1 - strength - HALF_AROUND)))


class RoutineNavigator:
    odometry_info: Odometry
    calibrated = False
    angle: float = 0.0
    
    def __init__(self):
        rospy.init_node("routine_navigator")

        self.odometry_info = None
        self._init_navigation()
        
        print("Calibrando sistema de coordenadas del ROBOT...")
        print("El punto de origen es el más importante. Este asume que el robot está 'derecho',")
        print("es decir que el ángulo de rotación del robot será tomado como *0*.\n")
        time.sleep(0.5)
        print("Se asume que al momento de settear el origen, el robot está mirando en la dirección del eje Y.")
        input("Presiona enter para settear el punto de origen.\nRECUERDA QUE EL ROBOT DEBE MIRAR EN LA DIRECCION DEL EJE Y DE TU SISTEMA REFERENCIADO\n> ")
        
        origin_odom = self.odometry_info
        
        self.origin_ref = np.array([origin_odom.pose.pose.position.x, origin_odom.pose.pose.position.y, origin_odom.pose.pose.position.z])
        self.rotation_bias = origin_odom.pose.pose.orientation
        
        self.A = get_matrix_transformation(self.rotation_bias, self.origin_ref)
        
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
            current = np.array([value.pose.pose.position.x, value.pose.pose.position.y, value.pose.pose.position.z, 1])
            transformed = self.A @ current
            value.pose.pose.position.x = float(transformed[0])
            value.pose.pose.position.y = float(transformed[1])
            value.pose.pose.position.z = float(transformed[2])
            value.pose.pose.orientation = multiply_quaternions(get_inverse(self.rotation_bias), value.pose.pose.orientation)
        
        self.odometry_info = value
            
    def _init_topic(self):  
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        self.previous_position = np.array([self.odometry_info.pose.pose.position.x, self.odometry_info.pose.pose.position.y])
        
        self.line = [[0.0925,0.08,0],[0.0975,0.085,0],[0.1175,0.1,0],[0.1375,0.115,0],[0.1525,0.125,0],[0.1625,0.135,0],[0.1725,0.145,0],[0.1825,0.155,0],[0.1925,0.165,0],[0.2025,0.175,0],[0.2125,0.185,0],[0.2225,0.2,0],[0.2275,0.21,0],[0.2375,0.225,0],[0.2525,0.24,0],[0.2625,0.255,0],[0.2775,0.275,0],[0.2825,0.285,0],[0.2875,0.295,0],[0.2975,0.305,0],[0.3025,0.31,0],[0.3125,0.325,0],[0.3225,0.335,0],[0.3275,0.34,0],[0.3425,0.355,0],[0.3475,0.36,0],[0.3525,0.365,0],[0.3625,0.38,0],[0.3675,0.385,0],[0.3725,0.39,0],[0.3825,0.4,0],[0.3875,0.405,0],[0.3975,0.415,0],[0.4025,0.42,0],[0.4075,0.425,0],[0.4125,0.435,0],[0.4175,0.44,0],[0.4275,0.455,0],[0.4325,0.46,0],[0.4375,0.465,0],[0.4425,0.475,0],[0.4525,0.48,0],[0.4575,0.485,0],[0.4625,0.49,0],[0.4775,0.5,0],[0.4825,0.505,0],[0.4875,0.51,0],[0.4975,0.515,0],[0.5025,0.525,0],[0.5075,0.53,0],[0.5125,0.54,0],[0.5225,0.555,0],[0.5275,0.56,0],[0.5325,0.565,0],[0.5375,0.575,0],[0.5425,0.585,0],[0.5525,0.59,0],[0.5575,0.6,0],[0.5625,0.605,0],[0.5725,0.61,0],[0.5775,0.615,0],[0.5825,0.615,0],[0.5925,0.62,0],[0.5975,0.625,0],[0.6025,0.63,0],[0.6075,0.635,0],[0.6125,0.64,0],[0.6175,0.645,0],[0.6175,0.65,0],[0.6225,0.655,0],[0.6275,0.66,0],[0.6425,0.665,0],[0.6475,0.67,0],[0.6525,0.675,0],[0.6625,0.68,0],[0.6675,0.685,0],[0.6725,0.685,0],[0.6775,0.685,0],[0.6775,0.69,0],[0.6825,0.69,0],[0.6875,0.69,0],[0.6925,0.695,0],[0.6975,0.695,0],[0.7025,0.7,0],[0.7075,0.705,0],[0.7075,0.71,0],[0.7125,0.71,0],[0.7175,0.715,0],[0.7225,0.72,0],[0.7275,0.725,0],[0.7325,0.73,0],[0.7375,0.735,0],[0.7425,0.735,0],[0.7475,0.74,0],[0.7575,0.75,0],[0.7575,0.755,0],[0.7625,0.76,0],[0.7675,0.765,0],[0.7725,0.765,0],[0.7725,0.77,0],[0.7775,0.775,0],[0.7775,0.785,0],[0.7775,0.795,0],[0.7825,0.8,0],[0.7875,0.805,0],[0.7875,0.81,0],[0.7875,0.82,0],[0.7825,0.83,0],[0.7775,0.84,0],[0.7775,0.845,0],[0.7775,0.85,0],[0.7725,0.865,0],[0.7675,0.88,0],[0.7575,0.89,0],[0.7525,0.9,0],[0.7475,0.91,0],[0.7475,0.92,0],[0.7475,0.925,0],[0.7425,0.93,0],[0.7375,0.935,0],[0.7375,0.94,0],[0.7375,0.955,0],[0.7325,0.965,0],[0.7325,0.97,0],[0.7325,0.975,0],[0.7275,0.98,0],[0.7225,0.995,0],[0.7225,1,0],[0.7225,1.015,0],[0.7225,1.02,0],[0.7225,1.03,0],[0.7225,1.045,0],[0.7125,1.055,0],[0.7125,1.07,0],[0.7075,1.085,0],[0.7075,1.095,0],[0.7075,1.105,0],[0.7075,1.125,0],[0.7025,1.14,0],[0.7025,1.155,0],[0.7025,1.18,0],[0.7025,1.2,0],[0.7025,1.225,0],[0.7025,1.25,0],[0.7025,1.29,0],[0.7025,1.32,0],[0.7175,1.355,0],[0.7325,1.385,0],[0.7475,1.41,0],[0.7675,1.445,0],[0.7825,1.465,0],[0.8075,1.5,0],[0.8225,1.515,0],[0.8375,1.53,0],[0.8475,1.54,0],[0.8575,1.55,0],[0.8925,1.57,0],[0.9025,1.575,0],[0.9225,1.585,0],[0.9425,1.59,0],[0.9525,1.59,0],[0.9775,1.6,0],[0.9875,1.6,0],[1.0075,1.6,0],[1.0325,1.6,0],[1.0525,1.61,0],[1.0725,1.61,0],[1.0975,1.61,0],[1.1275,1.61,0],[1.1575,1.61,0],[1.1775,1.61,0],[1.2075,1.61,0],[1.2275,1.61,0],[1.2475,1.61,0],[1.2675,1.61,0],[1.2825,1.605,0],[1.2925,1.605,0],[1.3075,1.6,0],[1.3175,1.595,0],[1.3275,1.595,0],[1.3425,1.585,0],[1.3575,1.575,0],[1.3775,1.57,0],[1.3925,1.56,0],[1.4025,1.555,0],[1.4225,1.55,0],[1.4325,1.545,0],[1.4575,1.535,0],[1.4675,1.525,0],[1.4875,1.51,0],[1.5025,1.5,0],[1.5175,1.485,0],[1.5325,1.48,0],[1.5475,1.47,0],[1.5575,1.455,0],[1.5625,1.45,0],[1.5775,1.43,0],[1.5825,1.42,0],[1.5875,1.405,0],[1.5975,1.385,0],[1.6025,1.375,0],[1.6075,1.36,0],[1.6275,1.33,0],[1.6275,1.315,0],[1.6375,1.285,0],[1.6425,1.265,0],[1.6475,1.25,0],[1.6475,1.23,0],[1.6475,1.22,0],[1.6475,1.205,0],[1.6475,1.195,0],[1.6475,1.18,0],[1.6475,1.16,0],[1.6475,1.15,0],[1.6475,1.13,0],[1.6475,1.12,0],[1.6475,1.095,0],[1.6475,1.08,0],[1.6475,1.07,0],[1.6425,1.06,0],[1.6425,1.04,0],[1.6425,1.03,0],[1.6375,1.015,0],[1.6375,1.01,0],[1.6325,1,0],[1.6325,0.985,0],[1.6225,0.975,0],[1.6175,0.96,0],[1.6075,0.935,0],[1.5975,0.925,0],[1.5925,0.91,0],[1.5875,0.9,0],[1.5825,0.89,0],[1.5825,0.885,0],[1.5775,0.87,0],[1.5725,0.865,0],[1.5675,0.855,0],[1.5625,0.85,0],[1.5575,0.84,0],[1.5525,0.835,0],[1.5425,0.825,0],[1.5325,0.81,0],[1.5225,0.8,0],[1.5025,0.78,0],[1.4875,0.775,0],[1.4775,0.77,0],[1.4675,0.76,0],[1.4625,0.755,0],[1.4525,0.755,0],[1.4475,0.755,0],[1.4425,0.75,0],[1.4375,0.75,0],[1.4275,0.75,0],[1.4175,0.75,0],[1.4025,0.75,0],[1.3975,0.75,0],[1.3875,0.75,0],[1.3775,0.75,0],[1.3725,0.75,0],[1.3675,0.75,0],[1.3575,0.75,0],[1.3525,0.75,0],[1.3475,0.75,0],[1.3425,0.75,0],[1.3325,0.75,0],[1.3275,0.75,0],[1.3225,0.75,0],[1.3175,0.75,0],[1.3075,0.75,0],[1.3025,0.75,0],[1.2975,0.755,0],[1.2925,0.76,0],[1.2875,0.76,0],[1.2825,0.76,0],[1.2725,0.76,0],[1.2575,0.76,0],[1.2525,0.765,0],[1.2475,0.765,0],[1.2325,0.77,0],[1.2175,0.775,0],[1.2125,0.775,0],[1.2075,0.775,0],[1.2025,0.775,0],[1.1975,0.775,0],[1.1925,0.775,0],[1.1875,0.775,0],[1.1825,0.775,0],[1.1725,0.775,0],[1.1675,0.775,0],[1.1625,0.775,0],[1.1475,0.775,0],[1.1425,0.775,0],[1.1425,0.77,0],[1.1425,0.765,0],[1.1425,0.76,0],[1.1425,0.755,0],[1.1425,0.745,0],[1.1425,0.74,0],[1.1425,0.735,0],[1.1425,0.73,0],[1.1425,0.72,0],[1.1425,0.705,0],[1.1425,0.685,0],[1.1425,0.67,0],[1.1425,0.66,0],[1.1425,0.645,0],[1.1525,0.635,0],[1.1525,0.625,0],[1.1525,0.62,0],[1.1525,0.61,0],[1.1525,0.595,0],[1.1525,0.575,0],[1.1525,0.55,0],[1.1525,0.535,0],[1.1525,0.52,0],[1.1525,0.51,0],[1.1525,0.49,0],[1.1525,0.475,0],[1.1525,0.465,0],[1.1525,0.45,0],[1.1525,0.44,0],[1.1525,0.42,0],[1.1525,0.4,0],[1.1525,0.39,0],[1.1525,0.385,0],[1.1525,0.375,0],[1.1525,0.37,0],[1.1525,0.365,0],[1.1525,0.36,0],[1.1525,0.35,0],[1.1525,0.345,0],[1.1525,0.34,0],[1.1525,0.335,0],[1.1525,0.32,0],[1.1525,0.315,0],[1.1525,0.31,0],[1.1525,0.3,0],[1.1525,0.295,0],[1.1525,0.29,0],[1.1525,0.28,0],[1.1525,0.275,0],[1.1525,0.27,0],[1.1525,0.265,0],[1.1525,0.255,0],[1.1525,0.25,0],[1.1575,0.25,0],[1.1675,0.25,0],[1.1775,0.25,0],[1.1825,0.25,0],[1.1925,0.25,0],[1.1975,0.25,0],[1.2075,0.25,0],[1.2175,0.25,0],[1.2275,0.25,0],[1.2375,0.255,0],[1.2425,0.255,0],[1.2525,0.26,0],[1.2575,0.26,0],[1.2675,0.26,0],[1.2725,0.265,0],[1.2775,0.265,0],[1.2825,0.265,0],[1.2925,0.265,0],[1.2975,0.265,0],[1.3025,0.265,0],[1.3075,0.265,0],[1.3175,0.265,0],[1.3225,0.27,0],[1.3275,0.275,0],[1.3425,0.28,0],[1.3475,0.28,0],[1.3625,0.28,0],[1.3675,0.285,0],[1.3725,0.285,0],[1.3775,0.285,0],[1.3975,0.285,0],[1.4125,0.285,0],[1.4275,0.285,0],[1.4425,0.285,0],[1.4625,0.285,0],[1.4775,0.285,0],[1.4975,0.285,0],[1.5275,0.285,0],[1.5475,0.285,0],[1.5725,0.285,0],[1.5925,0.285,0],[1.6225,0.285,0],[1.6475,0.285,0],[1.6775,0.285,0],[1.7175,0.285,0],[1.7575,0.285,0],[1.8025,0.285,0],[1.8325,0.285,0],[1.8625,0.285,0],[1.8875,0.285,0],[1.9275,0.285,0],[1.9525,0.285,0],[1.9775,0.285,0],[1.9975,0.285,0],[2.0225,0.285,0],[2.0475,0.285,0],[2.0775,0.285,0],[2.1025,0.285,0],[2.1275,0.285,0],[2.1525,0.285,0],[2.1775,0.285,0],[2.2025,0.285,0],[2.2475,0.285,0],[2.2675,0.285,0],[2.2875,0.285,0],[2.3075,0.285,0],[2.3125,0.285,0],[2.3175,0.285,0],[2.3275,0.285,0]]
        self.target_index = 1
        
        time.sleep(3)
        
        rotation = None
        
        while not rospy.is_shutdown() and self.target_index < len(self.line):
            current = self.odometry_info
            rotation = self.follow_line(current, rotation)
            self.previous_position = np.array([current.pose.pose.position.x, current.pose.pose.position.y])
            rospy.sleep(0.1)

        print("Routine complete.")
        
    
    def _rotate_left(self, movement_msg: Twist, strength: float = 1.0) -> Twist:   
        movement_msg.angular.x = 0.0
        movement_msg.angular.y = 0.0
        movement_msg.angular.z = MAX_PEPPER_SPEED * min(1.0, strength)
        
        return movement_msg
        
    
    def _rotate_right(self, movement_msg: Twist, strength: float = 1.0) -> Twist:
        movement_msg = self._rotate_left(movement_msg, strength)
        movement_msg.angular.z *= -1
        
        return movement_msg
    
    
    def _move_forward_relative_to_orientation(self, movement_msg: Twist, strength: float = 1.0) -> Twist:        
        # The robot will move in the direction he is looking at. By following its own coordinates system. Not the referenced on this script
        
        movement_msg.linear.x = MAX_PEPPER_SPEED * strength
        return movement_msg
    
        
    def follow_line(self, odometry_info: Odometry, rotation: str):
        msg = Twist()
        
        position = np.array([odometry_info.pose.pose.position.x, odometry_info.pose.pose.position.y])
        target = np.array([self.line[self.target_index][0], self.line[self.target_index][1]])
        movement_vector = target - position
        
        target_angle = norm_angle(math.atan2(movement_vector[1], movement_vector[0]))
        robot_angle = norm_angle(2 * math.atan2(odometry_info.pose.pose.orientation.z, odometry_info.pose.pose.orientation.w))
        
        rotation_diff = norm_angle(target_angle - robot_angle)
        rotation_strength = transform_strength(abs(rotation_diff) / math.pi)
        print("=========================================")
        print("Target:", target, "| Position:", position)
        print("Target angle:", target_angle, "| Robot angle:", robot_angle, "| Diff:", rotation_diff, "| Rotation Strength:", rotation_strength)
        print("Target distance:", np.linalg.norm(target - position))
        
        if abs(rotation_diff) > 0.1 and np.linalg.norm(movement_vector) > 0.1:
            if is_closer_left(target_angle, robot_angle):
                self._rotate_left(msg, rotation_strength)
            else:
                self._rotate_right(msg, rotation_strength)
        
        if np.linalg.norm(movement_vector) > 0.1:
            print("Aligned, walking")
            self._move_forward_relative_to_orientation(msg, 1 - rotation_strength)
        else:
            self.target_index += 1
            print("Arrived to point")
            
        
        self.pub.publish(msg)
        return rotation

if __name__ == "__main__":
    RoutineNavigator()