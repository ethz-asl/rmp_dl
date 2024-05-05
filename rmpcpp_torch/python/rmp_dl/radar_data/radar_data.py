import os
from typing import List

import cv2
import rosbag
import numpy as np

import sensor_msgs.point_cloud2 as pc2

from scipy.spatial.transform import Rotation as SciPyRot
from scipy.spatial.transform import Slerp

from cv_bridge import CvBridge


class RadarData():
    rio_topic = '/rio/odometry_optimizer'
    radar_topic = '/radar/cfar_detections'
    camera_topic = '/cam/image_raw'
    radar_field_names = ("x", "y", "z", "doppler", "snr", "noise")
    body_to_radar_pos = np.array([0.146, 0.012, -0.04,])
    body_to_radar_quat = np.array([0.6830127,  0.6830127,  -0.1830127, -0.1830127])

    body_to_radar_homogeneous = np.eye(4)
    body_to_radar_homogeneous[:3, :3] = SciPyRot.from_quat(body_to_radar_quat).as_matrix()
    body_to_radar_homogeneous[:3, 3] = body_to_radar_pos

    def __init__(self, path, relative_path=True):
        # Current file dir
        if relative_path:
            file = os.path.dirname(os.path.abspath(__file__))
            path = os.path.join(file, 'data', 'experiment7', path)
        
        self.path = path

        self.bag = rosbag.Bag(self.path)

    @staticmethod
    def Converted(path):
        rd = RadarData(path)
        rd._read_bag()
        rd._convert_into_inertial_frame()

        return rd

    def get_video(self, output_path):
        fs = (820, 616)
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps=20, frameSize=fs)
        bridge = CvBridge()

        i = 0
        for topic, msg, t in self.bag.read_messages(topics=[RadarData.camera_topic]):
            img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            assert img.shape[:2][::-1] == fs, "Image size does not match expected size. Expected: (820, 616), found: {}".format(img.shape[:2][::-1])
            video.write(img)
            i += 1
            print(i)

        video.release()

    def _read_bag(self):
        topics = [RadarData.rio_topic, RadarData.radar_topic]

        self._rio_data = np.zeros((0, 14))
        self._radar_timestamp_nr_points = np.zeros((0, 2))
        self._radar_points: List[np.ndarray] = []

        topic_set = set()

        for topic, msg, t in self.bag.read_messages(topics=topics):
            if topic not in topic_set:
                topic_set.add(topic)
                print(f"Found topic {topic}")

            if topic == RadarData.rio_topic:
                self._rio_data = np.vstack((self._rio_data, [msg.header.stamp.to_sec(), 
                                                msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z,
                                                msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w, 
                                                msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z,
                                                msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]))

            elif topic == RadarData.radar_topic:
                self._radar_timestamp_nr_points = np.vstack((self._radar_timestamp_nr_points, 
                                                             [msg.header.stamp.to_sec(), msg.width]))
                
                self._radar_points.append(list(pc2.read_points(msg, field_names=RadarData.radar_field_names, skip_nans=True)))
    
    def _convert_into_inertial_frame(self):
        # Inertial radar points are the radar points in the inertial frame. 
        # Every entry k in the list contains the radar points at time step k.
        self._inertial_radar_points: List[np.ndarray] = [] 

        # We also keep track of the estimated position at that time step
        self._inertial_position_estimates: List[np.ndarray] = []
        self._inertial_velocity_estimates: List[np.ndarray] = []
        self._inertial_forward_estimates: List[np.ndarray] = []
        
        if len(self._rio_data) <= 2 or len(self._radar_points) == 0:
            raise ValueError("Not enough data to convert radar points into inertial frame")
        
        rio_index = 0
        for i in range(0, len(self._radar_points)):
            radar_time = self._radar_timestamp_nr_points[i, 0]
            
            # We are before any useful state estimate
            if radar_time < self._rio_data[rio_index, 0]:
                continue
            
            # We only consider radar points that lie inbetween two rio timestamps
            # So we use timestamp rio_index and rio_index + 1, and linearly interpolate between those 2
            # In this loop we increment the rio index until we find the next rio timestamp that is larger than the radar timestamp
            while rio_index < len(self._rio_data) - 1 and radar_time > self._rio_data[rio_index + 1, 0]:
                rio_index += 1

            # We are after the last useful state estimate
            if rio_index == len(self._rio_data) - 1:
                break

            rio_left = self._rio_data[rio_index]
            rio_right = self._rio_data[rio_index + 1]
            t_left = rio_left[0]
            t_right = rio_right[0]

            # Linearly interpolate rotations
            left_rotation = SciPyRot.from_quat(rio_left[4:8])
            right_rotation = SciPyRot.from_quat(rio_right[4:8])
            rotations = SciPyRot.concatenate((left_rotation, right_rotation))
            slerp = Slerp([t_left, t_right], rotations)
            rotation = slerp([radar_time])

            left_pos = rio_left[1:4]
            right_pos = rio_right[1:4]
            pos = left_pos + (right_pos - left_pos) * (radar_time - t_left) / (t_right - t_left)
            left_vel = rio_left[8:11]
            right_vel = rio_right[8:11]
            body_vel = left_vel + (right_vel - left_vel) * (radar_time - t_left) / (t_right - t_left)

            # We need to rotate the radar points into the inertial frame
            radar_points = np.array(self._radar_points[i])
            inertial_to_body_homogeneous = np.eye(4)
            inertial_to_body_homogeneous[:3, :3] = rotation.as_matrix()
            inertial_to_body_homogeneous[:3, 3] = pos

            self._inertial_position_estimates.append(pos)

            # We also keep track of the estimated world velocity at that time step
            inertial_vel = inertial_to_body_homogeneous[:3, :3] @ body_vel # We ignore the translation
            self._inertial_velocity_estimates.append(inertial_vel)
            
            # We also keep track of the estimated world forward direction at that time step
            inertial_forward = inertial_to_body_homogeneous[:3, :3] @ np.array([1, 0, 0]) 
            self._inertial_forward_estimates.append(inertial_forward)

            if len(radar_points) == 0:
                self._inertial_radar_points.append(np.zeros((0, 6)))
                continue
            # Radar points is (n, 6) where n is the number of radar points. [0:3] is the position and [3:6] is other info
            # We extract the position and convert it into homogeneous coordinates
            radar_points_homogeneous = np.concatenate((radar_points[:, :3], np.ones((radar_points.shape[0], 1))), axis=1).T

            radar_points_converted = inertial_to_body_homogeneous @ RadarData.body_to_radar_homogeneous @ radar_points_homogeneous
            # Rescale by the last component
            radar_points_converted /= radar_points_converted[3]
            radar_points[:, :3] = radar_points_converted[:3].T
            self._inertial_radar_points.append(radar_points)

    def get_pointcloud(self) -> np.ndarray:
        return np.vstack(self._inertial_radar_points)[:, :3]

    def get_radar_points_at(self, idx) -> np.ndarray:
        return self._inertial_radar_points[idx][:, :3]

    def get_position(self, idx) -> np.ndarray:
        return self.get_positions()[idx]

    def get_positions(self) -> np.ndarray:
        return self._inertial_position_estimates

    def get_velocity(self, idx) -> np.ndarray:
        return self.get_velocities()[idx]
    
    def get_velocities(self) -> np.ndarray:
        return self._inertial_velocity_estimates

    def get_forward_direction(self, idx) -> np.ndarray:
        return self.get_forward_directions()[idx]
    
    def get_forward_directions(self) -> np.ndarray:
        return self._inertial_forward_estimates

    def get_doppler(self) -> np.ndarray:
        return np.vstack(self._inertial_radar_points)[:, 3]

    def __len__(self):
        return len(self._inertial_radar_points)

if __name__ == "__main__":
    # path = os.path.join('experiment7_urban', '01_urban_night_H_processed.bag')
    path = os.path.join('/media/isar/46E2F75E00965998/radar_data', '05_urban_night_F.bag')

    radar_data = RadarData(path, relative_path=False)
    output_path = os.path.join('/media/isar/46E2F75E00965998/radar_data', '05_urban_night_F.mp4')
    radar_data.get_video(output_path)
    