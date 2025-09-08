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

class RoutineNavigator:
    odometry_info: Odometry
    calibrated = False
    
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
        
        self.origin_ref = np.array([origin_odom.pose.pose.position.x, origin_odom.pose.pose.position.y, origin_odom.pose.pose.position.z, 1])
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
        print("Moving in ten seconds...")
        
        self.previous_position = np.array([self.odometry_info.pose.pose.position.x, self.odometry_info.pose.pose.position.y])
        
        self.line = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0)]
        self.target_index = 1
        
        rospy.sleep(10)
        
        while not rospy.is_shutdown() and self.target_index < len(self.line):
            current = self.odometry_info
            self.follow_line(current)
            self.previous_position = np.array([current.pose.pose.position.x, current.pose.pose.position.y])
            rospy.sleep(0.5)

        print("Routine complete.")
        
    
    def _rotate_left(self, movement_msg: Twist, strength: float = 1.0) -> Twist:
        if strength > 1.0 or strength < 0.0:
            raise ValueError("Strength must be a percentage")
        
        movement_msg.angular.x = 0.0
        movement_msg.angular.y = 0.0
        movement_msg.angular.z = MAX_PEPPER_SPEED * strength
        
        return movement_msg
        
    
    def _rotate_right(self, movement_msg: Twist, strength: float = 1.0) -> Twist:
        movement_msg = self._rotate_left(movement_msg, strength)
        movement_msg.angular.z *= -1
        
        return movement_msg
    
    
    def _move_forward_relative_to_orientation(self, movement_msg: Twist, strength: float = 1.0) -> Twist:
        if strength > 1.0 or strength < -1.0:
            raise ValueError("Strength must be a percentage (positive for forward negative for backward)")
        
        # The robot will move in the direction he is looking at. By following its own coordinates system. Not the referenced on this script
        
        movement_msg.linear.x = MAX_PEPPER_SPEED * strength
        return movement_msg
    
        
    def follow_line(self, odometry_info: Odometry):
        message = Twist()
        
        # TODO() Move to follow the path described by the line
        
        self.pub.publish(message)

if __name__ == "__main__":
    RoutineNavigator()