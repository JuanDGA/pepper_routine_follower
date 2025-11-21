#!/usr/bin/env python
import rospy
from robot_toolkit_msgs.srv import navigation_tools_srv, navigation_tools_srvRequest
from robot_toolkit_msgs.msg import animation_msg, speech_msg
from geometry_msgs.msg import Twist, Quaternion
from navigation_msgs.srv import follow_routine as follow_routine_srv, follow_routineResponse
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
import math
import time
import numpy as np
from scipy.spatial.transform import Rotation


MAX_PEPPER_SPEED = 0.25
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
        
        rospy.Service("/routine_follower/follow", follow_routine_srv, self._navigation_service)
        self.speech_pub = rospy.Publisher("/speech", speech_msg)
        self.animation_pub = rospy.Publisher("/animations", animation_msg)

        self.odometry_info = None
        self._init_navigation()
        
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
            
        rospy.spin()
        
    def read_odom(self, value: Odometry):
        if self.calibrated:
            current = np.array([value.pose.pose.position.x, value.pose.pose.position.y, value.pose.pose.position.z, 1])
            transformed = self.A @ current
            value.pose.pose.position.x = float(transformed[0])
            value.pose.pose.position.y = float(transformed[1])
            value.pose.pose.position.z = float(transformed[2])
            value.pose.pose.orientation = multiply_quaternions(get_inverse(self.rotation_bias), value.pose.pose.orientation)
        
        self.odometry_info = value
            
    def _navigation_service(self, request):
        print("Calibrando sistema de coordenadas del ROBOT...")
        print("El punto de origen es el más importante. Este asume que el robot está 'derecho',")
        print("es decir que el ángulo de rotación del robot será tomado como *0*.\n")
        time.sleep(0.5)
        print("Se asume que al momento de settear el origen, el robot está mirando en la dirección del eje Y.")
        
        origin_odom = self.odometry_info
        
        self.origin_ref = np.array([origin_odom.pose.pose.position.x, origin_odom.pose.pose.position.y, origin_odom.pose.pose.position.z])
        self.rotation_bias = origin_odom.pose.pose.orientation
        
        self.A = get_matrix_transformation(self.rotation_bias, self.origin_ref)
        
        self.calibrated = True
        
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        
        self.previous_position = np.array([self.odometry_info.pose.pose.position.x, self.odometry_info.pose.pose.position.y])
        
        self.line = [[it.z, it.x, 0] for it in request.line]
        nodes = request.nodes
        
        next_node = len(nodes)
    
        self.target_index = 0
        
        time.sleep(3)
        
        rotation = None
        
        while not rospy.is_shutdown() and self.target_index < len(self.line):
            missing_node = next_node < len(nodes)
            if missing_node and nodes[next_node].index == self.target_index:
                print(request.nodes[nodes[next_node]])
                speech_m = speech_msg()
                speech_msg.animated = bool(request.nodes[nodes[next_node]].animation)
                speech_msg.language = "Spanish"
                speech_msg.text = request.nodes[nodes[next_node]].text
                if request.nodes[nodes[next_node]].text:
                    self.speech_pub.publish(speech_m)
                # TODO: Talk
                # TODO: Play animation
            else:
                current = self.odometry_info
                rotation = self.follow_line(current, rotation)
                self.previous_position = np.array([current.pose.pose.position.x, current.pose.pose.position.y])
                rospy.sleep(0.1)

        print("Routine complete.")
        response = follow_routineResponse()
        response.received = True
        return response
        
    
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
        
        # Cambios
        # Look Ahead: Por deafult se realizara con los proximos 2-3 puntos
        look_ahead_count = min(3, len(self.line) - self.target_index)
        
        # Curvatura
        path_curvature = self.calculate_path_curvature(self.target_index, look_ahead=3)
        
        # Calcular vector promedio
        weight_vector = np.array([0.0,0.0])
        total_weight = 0.0
        for i in range(look_ahead_count):
            target_idx = self.target_index + i
            target = np.array([self.line[target_idx][0], self.line[target_idx][1]])
            vector = target - position
            
            # Peso decreciente
            weight = 1.0/ (i + 1)
            weight_vector  += vector * weight
            total_weight += weight
        
        # Normalizar
        if total_weight>0:
            weight_vector/= total_weight
            
        movement_vector = weight_vector
        
        print("In line:", self.line[self.target_index])
        current_target = np.array([self.line[self.target_index][0], self.line[self.target_index][1]])
        distance_to_current = np.linalg.norm(current_target - position)
        
        # TODO: smooth movement, revisar el cambio
        speed_factor = 1.0
        speed_factor *= (1.0-0.6*path_curvature) # Limite de 60% de reduccion de velocidad
        
        # No esta alineado
        
        # Cambios
        
        #target = np.array([self.line[self.target_index][0], self.line[self.target_index][2]])
        #movement_vector = target - position
        
        target_angle = norm_angle(math.atan2(movement_vector[1], movement_vector[0])) # Codigo original
        robot_angle = norm_angle(2 * math.atan2(odometry_info.pose.pose.orientation.z, odometry_info.pose.pose.orientation.w)) # Codigo original
        
        rotation_diff = norm_angle(target_angle - robot_angle)
        rotation_strength = transform_strength(abs(rotation_diff) / math.pi)
        speed_factor *= (1.0 - rotation_strength * 0.5)
        # print("=========================================")
        # print("Target:", target, "| Position:", position)
        # print("Target angle:", target_angle, "| Robot angle:", robot_angle, "| Diff:", rotation_diff, "| Rotation Strength:", rotation_strength)
        # print("Target distance:", np.linalg.norm(target - position))
        
        if abs(rotation_diff) > 0.1 and np.linalg.norm(movement_vector) > 0.1:
            if is_closer_left(target_angle, robot_angle):
                self._rotate_left(msg, rotation_strength)
            else:
                self._rotate_right(msg, rotation_strength)
        
        if distance_to_current > 0.1:
            print(f"Moving -  curvature: {path_curvature: .2f}, Speed Factor: {speed_factor: .2f}", current_target, position)
            self._move_forward_relative_to_orientation(msg, speed_factor)
        else:
            self.target_index += 1
            print( "Arrived to point", position, current_target)
        
        #if np.linalg.norm(movement_vector) > 0.1:
        #    print("Aligned, walking")
        #    self._move_forward_relative_to_orientation(msg, 1 - rotation_strength)
        #else:
        #    self.target_index += 1
        #    print("Arrived to point", position, target)
            
        
        self.pub.publish(msg)
        return rotation
    
    # Look Ahead requirements: Recalculate path curvature, rotating before target
    # TODO: Check for a better aproximation
    
    def calculate_path_curvature(self, start_idx: int, look_ahead: int=3) -> float:
        """
        Calcula la curvatura del camino basandonos en el cambio de direccion
        entre puntos consecutivos.
        Retorna un valor entre 0 (recto) o 1 (curva).
        """
        # Puede retornar la variacion angular? 
        
        if start_idx + look_ahead >= len(self.line):
            look_ahead = len(self.line) - start_idx - 1
        if look_ahead < 2:
            return 0.0
        
        angles = []
        
        # TODO: Revisar variacion en los ejes.
        for i in range(look_ahead):
            p1 = np.array([self.line[start_idx + i][0], self.line[start_idx + i][1]])
            p2 = np.array([self.line[start_idx + i + 1][0], self.line[start_idx + i + 1][1]])
            
            vector = p2 - p1
            angle = math.atan2(vector[1], vector[0])
            angles.append(angle)
        
        # Calcula la variacion angular total
        total_angle_change = 0.0
        for i in range(len(angles)-1):
            angle_diff = abs(norm_angle(angles[i+1]-angles[i]))
            total_angle_change += angle_diff
        
        # Normalizar
        curvature = min(1.0, total_angle_change/(math.pi/2))
        return curvature
    
    # TODO: validar el uso de predecir la rotacion con aticipacion
    def should_start_rotating( self, position: np.ndarray, current_target: np.ndarray, next_target: np.ndarray):
        distance_to_current = np.linalg.norm(current_target - position)
        
        if distance_to_current < 0.5 and self.target_index + 1 < len(self.line):
            vector_to_next = next_target - current_target
            angle_to_next = math.atan2(vector_to_next[1], vector_to_next[0])
            return True, angle_to_next
        return False, 0.0
            
            

if __name__ == "__main__":
    RoutineNavigator()