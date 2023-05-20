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

# Create depthai pipeline
def create_pipeline():
    # Set up RGB camera pipeline
    pipeline = depthai.Pipeline()
    cam_rgb = pipeline.create(depthai.node.ColorCamera)

    cam_rgb.setBoardSocket(depthai.CameraBoardSocket.RGB)
    cam_rgb.setResolution(depthai.ColorCameraProperties.SensorResolution.THE_4_K)
    cam_rgb.setInterleaved(False) # Planar format camera data
    cam_rgb.setVideoSize(1920*2, 1080*2)

    # Set link to transfer RGB from camera to host
    xout_rgb = pipeline.create(depthai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")

    xout_rgb.input.setBlocking(False)
    xout_rgb.input.setQueueSize(1)

    cam_rgb.video.link(xout_rgb.input)


    return pipeline


# Start the pipeline
pipeline = create_pipeline()
with depthai.Device(pipeline) as device:
    
    video = device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
    
    # Main loop
    while True:
        videoIn = video.get()

        cv2.imshow("rgb", videoIn.getCvFrame())


        # Exit if escape is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break