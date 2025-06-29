#!/home/sungjae/.conda/envs/mediapipe_env/bin/python

import rospy
import numpy as np
import os
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe.framework.formats import landmark_pb2

import threading

fingers = {
    "Thumb":  [1, 2, 3, 4],
    "Index":  [5, 6, 7, 8],
    "Middle": [9, 10, 11, 12],
    "Ring":   [13, 14, 15, 16],
    "Pinky":  [17, 18, 19, 20]
}

class HandLandmarkerNode:
    def __init__(self):
        rospy.init_node('hand_landmarker_node')
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback, queue_size=1)
        self.joint_pub = rospy.Publisher('/hand_joint_states', JointState, queue_size=10)
        self.image_publisher = rospy.Publisher('/hand_landmarks/image', Image, queue_size=10)

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_result = None
        self.latest_image = None

        model_path = '/home/sungjae/son_ws/conda/hand_landmarker.task'
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,
            result_callback=self.print_result,
            num_hands=1,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # 결과 퍼블리시용 타이머 (30Hz)
        rospy.Timer(rospy.Duration(0.03), self.publish_result)

        self.joint_names = []
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        joints = ["mcp_flex", "mcp_abd", "pip", "dip"]
        for finger in fingers:
            for joint in joints:
                self.joint_names.append(f"{finger}_{joint}")

        rospy.loginfo("MediaPipe Hand Landmarker initialized.")
        rospy.spin()

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge error: {e}")
            return

        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        timestamp_ms = rospy.Time.now().to_nsec() // 1_000_000
        self.detector.detect_async(mp_image, timestamp_ms)

    def print_result(self, result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        with self.lock:
            self.latest_result = result
            self.latest_image = output_image

    def publish_result(self, event):
        with self.lock:
            if self.latest_result is None or self.latest_image is None:
                return

            annotated_image = self.latest_image.numpy_view()
            annotated_image_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

            if self.latest_result.hand_landmarks:
                for idx in range(len(self.latest_result.hand_landmarks)):
                    hand_landmarks = self.latest_result.hand_landmarks[idx]

                    wrist = self._vec(hand_landmarks[0])
                    index_mcp = self._vec(hand_landmarks[5])
                    middle_mcp = self._vec(hand_landmarks[9])
                    ring_mcp = self._vec(hand_landmarks[13])
                    pinky_mcp = self._vec(hand_landmarks[17])

                    # 1. 손바닥 평면 법선 벡터
                    # palm_normal = np.cross(ring_mcp - middle_mcp, pinky_mcp - middle_mcp)
                    # palm_normal /= np.linalg.norm(palm_normal)

                    palm_normal = self.get_stable_palm_normal(wrist, index_mcp, ring_mcp)



                    # 2. 기준 축: 중지 기준 MCP → PIP 벡터
                    middle_pip = self._vec(hand_landmarks[10])
                    middle_dir = middle_pip - middle_mcp
                    middle_proj = middle_dir - np.dot(middle_dir, palm_normal) * palm_normal
                    middle_proj /= np.linalg.norm(middle_proj)

                    # 3. 각 손가락에 대해
                    os.system("clear")
                    for name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():
                        mcp = self._vec(hand_landmarks[mcp_i])
                        pip = self._vec(hand_landmarks[pip_i])
                        dip = self._vec(hand_landmarks[dip_i])
                        tip = self._vec(hand_landmarks[tip_i])

                        # Flexion
                        mcp_flex = self.calculate_angle(wrist, mcp, pip)
                        pip_flex = self.calculate_angle(mcp, pip, dip)
                        dip_flex = self.calculate_angle(pip, dip, tip)

                        # 4. Abduction
                        middle_dir = middle_pip - middle_mcp
                        middle_proj = middle_dir - np.dot(middle_dir, palm_normal) * palm_normal
                        middle_proj /= np.linalg.norm(middle_proj)

                        # 손가락 처리 루프 내부
                        reference_dir = middle_mcp - wrist
                        reference_dir /= np.linalg.norm(reference_dir)

                        # abduction 계산 (새 방식)
                        abd_angle = self.compute_abduction_from_hand_center(wrist, mcp, reference_dir)

                        # if(name == 'Index'):
                        print(f"[{name}]")
                        print(f"  MCP Flexion: {mcp_flex:.1f}°")
                        print(f"  MCP Abduction: {abd_angle:.1f}°")
                        print(f"  PIP Flexion: {pip_flex:.1f}°")
                        print(f"  DIP Flexion: {dip_flex:.1f}°")

                        joint_msg = JointState()
                        joint_msg.header.stamp = rospy.Time.now()
                        joint_msg.name = self.joint_names
                        joint_msg.position = []

                        reference_dir = middle_mcp - wrist
                        reference_dir /= np.linalg.norm(reference_dir)

                        # 루프 내부
                        for name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():
                            mcp = self._vec(hand_landmarks[mcp_i])
                            pip = self._vec(hand_landmarks[pip_i])
                            dip = self._vec(hand_landmarks[dip_i])
                            tip = self._vec(hand_landmarks[tip_i])

                            mcp_flex = self.calculate_angle(wrist, mcp, pip, in_degrees=False)
                            pip_flex = self.calculate_angle(mcp, pip, dip, in_degrees=False)
                            dip_flex = self.calculate_angle(pip, dip, tip, in_degrees=False)

                            # ✅ 안정화된 abduction 계산 사용
                            abd_angle_rad = np.radians(self.compute_abduction_from_hand_center(wrist, mcp, reference_dir))

                            joint_msg.position += [mcp_flex, abd_angle_rad, pip_flex, dip_flex]
                            self.joint_pub.publish(joint_msg)


                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
                    ])

                    mp_drawing = mp.solutions.drawing_utils
                    mp_styles = mp.solutions.drawing_styles

                    mp_drawing.draw_landmarks(
                        image=annotated_image_bgr,
                        landmark_list=hand_landmarks_proto,
                        connections=mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_styles.get_default_hand_connections_style()
                    )

            try:
                ros_image = self.bridge.cv2_to_imgmsg(annotated_image_bgr, encoding="bgr8")
                self.image_publisher.publish(ros_image)
            except Exception as e:
                rospy.logerr(f"Image publish error: {e}")

    def get_stable_palm_normal(self, wrist, index_mcp, ring_mcp):
        # 두 벡터: wrist → index_mcp, wrist → ring_mcp
        v1 = index_mcp - wrist
        v2 = ring_mcp - wrist

        # 법선 벡터 계산
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # palm_forward: wrist → middle_mcp
        middle_mcp = (index_mcp + ring_mcp) / 2  # 중간 손가락 위치 근사
        palm_forward = middle_mcp - wrist
        palm_forward /= np.linalg.norm(palm_forward)

        # normal 방향 안정화
        if np.dot(normal, palm_forward) < 0:
            normal = -normal

        return normal

    def _vec(self, landmark):
        return np.array([landmark.x, landmark.y, landmark.z])

    def angle_between(self, v1, v2, normal=None):
        v1 = np.array(v1)
        v2 = np.array(v2)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        angle = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
        if normal is not None:
            cross = np.cross(v1, v2)
            if np.dot(cross, normal) < 0:
                angle *= -1
        return np.degrees(angle)

    def calculate_angle(self, a, b, c, in_degrees=True):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return np.degrees(angle) if in_degrees else angle

    def compute_abduction_angle(self, base_vec, target_vec, palm_normal):
        base_vec = base_vec / np.linalg.norm(base_vec)
        target_vec = target_vec / np.linalg.norm(target_vec)

        angle = np.arccos(np.clip(np.dot(base_vec, target_vec), -1.0, 1.0))
        cross = np.cross(base_vec, target_vec)
        dot_val = np.dot(cross, palm_normal)

        if abs(dot_val) < 1e-3:
            sign = 1.0  # 또는 이전 프레임 sign 유지
        else:
            sign = np.sign(dot_val)

        signed_angle = angle * sign
        return np.degrees(signed_angle)

    def compute_abduction_from_hand_center(self, wrist, mcp, reference_dir):
        """
        wrist → MCP 벡터를 reference_dir에 투영하여 abduction 크기 및 부호 계산
        """
        vec = mcp - wrist
        vec_proj = vec - np.dot(vec, reference_dir) * reference_dir

        if np.linalg.norm(vec_proj) < 1e-6:
            return 0.0

        # normalize projected vector
        vec_proj /= np.linalg.norm(vec_proj)

        # 기준 축: 손바닥 기준 왼쪽 방향 (ex: index 기준으로 middle - wrist)
        hand_x_axis = np.cross(reference_dir, np.array([0, 0, 1]))  # 혹은 고정된 z축에 대한 수직

        angle = np.arccos(np.clip(np.dot(vec_proj, hand_x_axis), -1.0, 1.0))
        sign = np.sign(np.dot(np.cross(hand_x_axis, vec_proj), reference_dir))
        return np.degrees(angle * sign)

if __name__ == '__main__':
    try:
        HandLandmarkerNode()
    except rospy.ROSInterruptException:
        pass