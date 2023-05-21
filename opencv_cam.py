"""OpenCV Camera Manager"""
__author__ = "Trent Shumay"
__date__ = "2023-05-20"
__version__ = "0.1.0"
__license__ = "MIT"
__status__ = "Prototype"


import cv2

class OpenCVCam:

    def __init__(self, cam_id=0, width=1920, height=1080, fps=30):
        """ Initialize the OpenCV camera manager. """
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cam = None
        self.frame = None
        self.frame_count = 0
        self.frame_time = 0
        self.frame_rate = 0

    def start(self):
        """ Open the camera. """
        self.cam = cv2.VideoCapture(self.cam_id)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cam.set(cv2.CAP_PROP_FPS, self.fps)

        return self.is_opened()

    def stop(self):
        """ Close the camera. """
        self.cam.release()
        self.cam = None

    def read_frame(self):
        """ Read a frame from the camera. """
        self.frame_count += 1
        self.frame_time = cv2.getTickCount()
        ret, self.frame = self.cam.read()
        return ret,self.frame
    
    def is_opened(self):
        """ Check if the camera is open. """
        return self.cam.isOpened()