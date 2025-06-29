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

        rospy.Timer(rospy.Duration(0.03), self.publish_result)

        self.joint_names = []
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            for joint in ["mcp_flex", "mcp_abd", "pip", "dip"]:
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
        timestamp_ms = rospy.Time.now().to_nsec()//1_000_000
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
                for idx, landmarks in enumerate(self.latest_result.hand_landmarks):
                    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
                    wrist = pts[0]
                    middle_mcp = pts[9]
                    # reference_dir = middle_mcp - wrist
                    # reference_dir /= np.linalg.norm(reference_dir)

                    x_axis = pts[17] - pts[5]
                    x_axis /= np.linalg.norm(x_axis)

                    # 손바닥 평면 normal
                    palm_normal = np.cross(pts[5] - pts[0], pts[17] - pts[0])
                    palm_normal /= np.linalg.norm(palm_normal)

                    joint_msg = JointState()
                    joint_msg.header.stamp = rospy.Time.now()
                    joint_msg.name = self.joint_names
                    joint_msg.position = []
                    
                    os.system("clear")
                    for finger_name, (mcp_i, pip_i, dip_i, tip_i) in fingers.items():
                        mcp = pts[mcp_i]
                        pip = pts[pip_i]
                        dip = pts[dip_i]
                        tip = pts[tip_i]

                        mcp_flex = self.calculate_angle(wrist, mcp, pip, in_degrees=False)
                        pip_flex = self.calculate_angle(mcp, pip, dip, in_degrees=False)
                        dip_flex = self.calculate_angle(pip, dip, tip, in_degrees=False)
                        mcp_abd = self.compute_abduction_from_hand_center(mcp, pip, x_axis, palm_normal)

                        print(f"[{finger_name}]")
                        print(f"  MCP Flexion: {180.0/3.141592*mcp_flex:.1f}°")
                        print(f"  MCP Abduction: {180.0/3.141592*mcp_abd:.1f}°")
                        print(f"  PIP Flexion: {180.0/3.141592*pip_flex:.1f}°")
                        print(f"  DIP Flexion: {180.0/3.141592*dip_flex:.1f}°")

                        joint_msg.position += [mcp_flex, mcp_abd, pip_flex, dip_flex]

                    self.joint_pub.publish(joint_msg)

                    # Draw landmarks
                    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                    hand_landmarks_proto.landmark.extend([
                        landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks
                    ])
                    mp.solutions.drawing_utils.draw_landmarks(
                        image=annotated_image_bgr,
                        landmark_list=hand_landmarks_proto,
                        connections=mp.solutions.hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )

            try:
                ros_image = self.bridge.cv2_to_imgmsg(annotated_image_bgr, encoding="bgr8")
                self.image_publisher.publish(ros_image)
            except Exception as e:
                rospy.logerr(f"Image publish error: {e}")

    def calculate_angle(self, a, b, c, in_degrees=True):
        ba = a - b
        bc = c - b
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle) if in_degrees else angle

    def compute_abduction_from_hand_center(self, mcp, pip, x_axis, palm_normal):
        vec = pip - mcp
        proj = vec - np.dot(vec, palm_normal) * palm_normal
        if np.linalg.norm(proj) < 1e-6:
            return 0.0
        proj /= np.linalg.norm(proj)

        angle = np.arccos(np.clip(np.dot(proj, x_axis), -1.0, 1.0))
        sign = np.sign(np.dot(np.cross(x_axis, proj), palm_normal))
        return angle * sign

if __name__ == '__main__':
    try:
        HandLandmarkerNode()
    except rospy.ROSInterruptException:
        pass