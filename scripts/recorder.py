#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import json
import tf
import cv2
from sys import argv
from os import system
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image
from threading import RLock
from copy import deepcopy
from cv_bridge.core import CvBridge
# from thr_infrastructure_msgs.msg import ActionHistoryEvent


class Recorder:
    def __init__(self, path, rate=20, timeout=1):
        self.tfl = tf.TransformListener(True, rospy.Duration(timeout))
        self.transforms = []
        self.start_time = None
        self.path = path
        self.frames = rospy.get_param('/recorder/frames', []) if rospy.get_param('/recorder/enabled/frames') else []
        self.actions = []
        self.world = "base"
        self.rate_hz = rate
        self.rate = rospy.Rate(rate)
        self.microrate = rospy.Rate(1000)
        self.bridge = CvBridge()
        self.ready = False  # True when all components are ready
        self.recording = False # True when all components are ready and the user asked to start

        # Enabled cameras
        self.cameras_enabled = {'kinect': rospy.get_param('/recorder/enabled/kinect', False),
                                'depth': rospy.get_param('/recorder/enabled/depth', False)}
                                
        # Non-cameras sources to record as well
        self.components_enabled = {'frames': rospy.get_param('/recorder/enabled/frames', True),
                                   'actions': rospy.get_param('/recorder/enabled/actions', False)}

        # Recording is triggered when all components are ready
        self.readiness_lock = RLock()
        self.components_ready = {'kinect': False, 'depth': False, 'clock': not rospy.get_param('use_sim_time', default=False)}

        self.frames = rospy.get_param('/recorder/frames') if self.components_enabled['frames'] else []
        self.image = {'kinect': None, 'depth': None}
        self.locks = {'kinect': RLock(), 'depth': RLock()}
        # self.four_cc = cv.CV_FOURCC('F' ,'M','P', '4')
        self.four_cc = cv2.VideoWriter_fourcc('F' ,'M','P', '4')
        self.extension = rospy.get_param('/recorder/extension/', '.avi')
        self.writers = {'kinect': None, 'depth': None}
        self.formats = {'kinect': 'bgr8', 'depth': '8UC1'} #'depth': 32FC1?}

        self.clock_sub = rospy.Subscriber('/clock', Clock, self.cb_clock, queue_size=1)
        self.kinect_name = rospy.get_param('/recorder/kinect_name', 'kinectv2')
        self.quality = rospy.get_param('quality', 'qhd')
        self.kinect_rgb_topic = [self.kinect_name, self.quality, 'image_color']
        self.kinect_depth_topic = [self.kinect_name, self.quality, 'image_depth_rect']
        self.kinect_rgb_sub = rospy.Subscriber('/'.join(self.kinect_rgb_topic), Image, self.cb_kinect_rgb, queue_size=1)
        self.kinect_depth_sub = rospy.Subscriber('/'.join(self.kinect_depth_topic), Image, self.cb_kinect_depth, queue_size=1)
        # self.actions_sub = rospy.Subscriber('/thr/action_history', ActionHistoryEvent, self.cb_action_history, queue_size=100)

    def get_unready_components(self):
        return [component for component in self.cameras_enabled if self.cameras_enabled[component] and not self.components_ready[component]]

    def _update_readiness(self, component):
        self.components_ready[component] = True
        with self.readiness_lock:
            not_ready = self.get_unready_components()
            self.ready = len(not_ready) == 0

    def cb_clock(self, clock):
        self.clock_sub.unregister()
        self._update_readiness('clock')

    def cb_kinect_rgb(self, image):
        with self.locks['kinect']:
            self.image['kinect'] = image
        self.open_writer('kinect')
        self._update_readiness('kinect')

    def cb_kinect_depth(self, image):
        with self.locks['depth']:
            self.image['depth'] = image
        self.open_writer('depth')
        self._update_readiness('depth')

    def cb_action_history(self, action):
        events = ['start', 'success', 'failure']
        if self.components_enabled['actions'] and self.recording:
            self.actions.append({'time': (action.header.stamp - self.start_time).to_sec(),
                                 'action': action.action.type,
                                 'parameters': action.action.parameters,
                                 'event': events[action.type],
                                 'arm': action.side})

    def open_writer(self, camera):
        if self.cameras_enabled[camera] and not self.writers[camera]:
            self.writers[camera] = cv2.VideoWriter(path + '/' + camera + self.extension,
                                               self.four_cc, self.rate_hz,
                                               (self.image[camera].width, self.image[camera].height),
                                               isColor=camera != 'depth')
    def save_transforms(self):
        transformations = {}
        for frame in self.frames:
            try:
                lct = self.tfl.getLatestCommonTime(self.world, frame)
                transform = self.tfl.lookupTransform(self.world, frame, lct)
            except tf.Exception, e:
                transformations[frame] = {"visible": False}
            else:
                transformations[frame] = {"visible": True, "pose": transform}
        sample = {"time": (rospy.Time.now() - self.start_time).to_sec(), "objects": transformations}
        self.transforms.append(sample)

    def save_image(self, camera):
        with self.locks[camera]:
            ros_image = deepcopy(self.image[camera])
        cv_image = self.bridge.imgmsg_to_cv2(ros_image, desired_encoding=self.formats[camera])
        self.writers[camera].write(cv_image)

    def dump(self):
        # Videos streams
        for camera, enabled in self.cameras_enabled.iteritems():
            if enabled:
                if self.writers[camera]:
                    rospy.loginfo("Generating video file, side {}...".format(camera))
                    self.writers[camera].release()
                    rospy.loginfo("Video file side {} generated".format(camera))
                else:
                    rospy.logwarn("Cannot generate file {} or no data to save".format(camera))

        # Actions
        if self.components_enabled['actions']:
            rospy.loginfo("Generating JSON file of actions...")
            file = self.path + '/actions.json'
            with open(file, 'w') as f:
                json.dump(self.actions, f)
            rospy.loginfo("JSON generated at {}".format(file))

        # Transformations (frames)
        if self.components_enabled['frames']:
            rospy.loginfo("Generating JSON file of frames...")
            file = self.path + '/frames.json'
            data = {"metadata": {"world": self.world, "objects": self.frames, "timestamp": self.start_time.to_sec()}, "transforms": self.transforms}
            with open(file, 'w') as f:
                json.dump(data, f)
            rospy.loginfo("JSON generated at {}".format(file))

    def wait_subscribers(self):
        rospy.loginfo('Waiting all subscribers to be ready...')
        counter = 1
        while not self.ready and not rospy.is_shutdown():
            self.microrate.sleep()
            if counter == 0:
                with self.readiness_lock:
                    not_ready = self.get_unready_components()
                rospy.loginfo("Waiting components {}...".format(str(not_ready)))
            counter = (counter + 1) % 1000

    def run(self):
        self.wait_subscribers()
        if not rospy.is_shutdown():
            raw_input("Subscribers ready, press <Enter> to start recording...")
            self.recording = True
            self.start_time = rospy.Time.now()
            rospy.loginfo("Starting recording ... press <Ctrl-C> to stop")
            try:
                while not rospy.is_shutdown():
                    if self.components_enabled['frames']:
                        self.save_transforms()
                    for component, enabled in self.cameras_enabled.iteritems():
                        if enabled:
                            self.save_image(component)
                    self.rate.sleep()
            except rospy.exceptions.ROSInterruptException:
                pass
            finally:
                self.dump()

if __name__=='__main__':
    rospy.init_node('recorder')
    path = rospy.get_param("/recorder/path", "/tmp")

    rospy.loginfo("Creating new empty dataset under {}".format(path))
    mkdir = system("mkdir -p {}".format(path))

    if mkdir == 0:
        Recorder(path).run()
    else:
        rospy.logerr("Unable to create dataset path {}, mkdir returned {}".format(path, mkdir))
