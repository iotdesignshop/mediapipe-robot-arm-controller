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

To see command line options:
```
$ python controller.py help
```

## Camera Selection and Preview Window
The default behaviour is to look at your machine to see if there is a DepthAI powered device (OAK-D Camera) first and to try initialize capture at 4K resolution. If an OAK-D is not detected, the script will fall back to using the first webcam on the system detected by OpenCV at a default 1080P resolution 

The preview window defaults to a 1280x720 resolution regardless of the resolution of the input camera, so don't use this as a reference of the capture size. This is to make it a bit easier to manage on your desktop.

These values can all be overriden via command-line parameters as follows:

- **oakd-capture-width** (3840, 1920) - Sets the horizontal resolution of the capture (note this must match the vertical, either 4K or 1080P)
- **oakd-capture-height** (2160, 1080) - Sets the vertical resolution of the capture (note this must match the horizontal, either 4K or 1080P)
- **webcam-capture-width** (Any valid value for your system)
- **webcam-capture-height** (Any valid value for your system)
- **preview-width** (Width of the preview window on the desktop)
- **preview-height** (Height of the preview window on the desktop)
- **force-webcam** (Ignore any DepthAI devices present and use the system webcam)

Examples:

To launch DepthAI in 1080P mode instead of 4K
```
$ python controller.py --oakd-capture-width 1920 --oakd-capture-height 1080
```

To force webcam use at 720P
```
$ python controller.py --force-webcam --webcam-capture-width 1280 --webcam-capture-height 720
```

## Serial Communication
The demo contains the code from the original demo which packs the 23 DOF angles into an integer array and transmits over a serial port of choice. Because the demo won't run without a proper serial port, this function is disabled by default, and has to be enabled on the command -line.

- **enable-serial** - Turns on the serial comms
- **serial-port** - Sets the serial port (defaults to COM15 as per original demo)

Example:

To launch with serial enabled:
```
$ python controller.py --enable-serial --serial-port COM5
```

### Serial enumeration utility
There is a python script provided for enumerating serial devices on the system. To see available devices:
```
$ python list_serial_ports.py
```

This will display a result similar to this
```
COM4 - Standard Serial over Bluetooth link (COM4)
COM3 - Standard Serial over Bluetooth link (COM3)
```


## What am I seeing?
The demo will open with a video view (note this is mirrored so that you can move more organically) which displays computed values for the skeletal pose (right shoulder down to right hand currently). 

It will also open 3 planar projection views for X=0 (Side View), Y=0 (Top View), and Z=0 (Front View).

In those views, the joints being used for the calculations will be highlighted:

- Cyan (Elbow Calculation)
- Magenata (Shoulder Yaw Calculation)
- Yellow (Shoulder Pitch Calculation)

### Runtime hotkeys
With the demo active, you can press the following keys to perform actions:

- **ESC** - Quit the demo
- **d** - Toggle debug views on and off


