# mediapipe-robot-arm-controller
A demonstration of using MediaPipe Holistic to create a controller for a robot arm from the shoulder to fingers. 

# Installation

I would recommend creating a Python virtual environment first:

### Mac and Linux
```
$ python3 -m venv mp_env && source mp_env/bin/activate
```
Then, install the requirements.txt as follows:
```
$ pip install -r requirements.txt
```

### Windows
```
python -m venv mp_env
.\mp_env\Scripts\activate
```
Then, install the requirements.txt as follows:
```
$ pip install -r requirements.txt
```



# Running the Demo

To launch the demo script:
```
$ python controller.py
```

It will use the first web camera available on your machine

## What am I seeing?
The demo will open with a video view (note this is mirrored so that you can move more organically) which displays computed values for the skeletal pose (right shoulder down to right hand currently). 

It will also open 3 planar projection views for X=0 (Side View), Y=0 (Top View), and Z=0 (Front View).

In those views, the joints being used for the calculations will be highlighted:

- Cyan (Elbow Calculation)
- Magenata (Shoulder Yaw Calculation)
- Yellow (Shoulder Pitch Calculation)

