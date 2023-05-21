""" This module provides a wrapper for the DepthAI API and OAKD Camera. """
__author__ = "Trent Shumay"
__date__ = "2023-05-18"
__version__ = "0.1.0"
__license__ = "MIT"
__status__ = "Prototype"

# Package dependencies
import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets

class DepthAICam:
    def __init__(self, width=1920, height=1080, fps=30):
        """ Initialize the DepthAI camera manager. """
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.frame = None
        self.frame_count = 0
        self.frame_time = 0
        self.frame_rate = 0 
        
    # Create depthai pipeline
    def _create_pipeline(self):
        # Set up RGB camera pipeline
        pipeline = depthai.Pipeline()
        cam_rgb = pipeline.create(depthai.node.ColorCamera)

        cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
        cam_rgb.setInterleaved(False) # Planar format camera data
        
        if (self.width == 1920 and self.height == 1080):
            cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam_rgb.setVideoSize(1920, 1080)

        elif (self.width == 3840 and self.height == 2160): 
            cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)
            cam_rgb.setVideoSize(3840, 2160)
        else:
            raise ValueError("Invalid resolution: " + str(self.width) + ":" + str(self.height))

        # Set link to transfer RGB from camera to host
        xout_rgb = pipeline.create(depthai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")

        xout_rgb.input.setBlocking(False)
        xout_rgb.input.setQueueSize(1)

        cam_rgb.video.link(xout_rgb.input)


        return pipeline

    # Start the pipeline
    def start(self):
        """ Open the camera. """

        # Is there a DepthAI device available?
        if not self.is_depthai_device_available():
            return False

        # Create the pipeline
        self.pipeline = self._create_pipeline()
        self.device = depthai.Device(self.pipeline)
        self.video = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        return True

    def stop(self):
        """ Close the camera. """
        self.device.close()
        self.device = None

    def read_frame(self):
        """ Read a frame from the camera. """
        self.frame_count += 1
        self.frame_time = cv2.getTickCount()
        self.frame = self.video.get().getCvFrame()
        return True,self.frame
    
    def is_opened(self):
        """ Check if the camera is open. """
        return not self.device.isClosed()
    
    def is_depthai_device_available(self):
        """ Check if a DepthAI device is available. """
        if not depthai.Device.getAllAvailableDevices():
            return False
        else:
            return True



# Stand alone test
if __name__ == "__main__":
    cam = DepthAICam()
    if (cam.start()):
        while cam.is_opened():
            success, frame = cam.read_frame()
            if success:
                cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cam.stop()
        cv2.destroyAllWindows()
    else:
        print("No DepthAI device connected.")
