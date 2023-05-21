import cv2
import mediapipe as mp
import numpy as np
from copy import deepcopy
import argparse
import opencv_cam
import depthai_cam

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
mp_hand = mp.solutions.hands
mp_holistic = mp.solutions.holistic

def visibilityToColour(vis):
  if (vis < 0.5):
    return (0, 0, 255)  # Red - low visibility
  elif (vis < 0.75):
    return (0,255,255)  # Yellow - medium visibility
  else:
    return (0, 255, 0)  # Green - high visibility
  
def angle(a, b, c):
    # https://stackoverflow.com/questions/35176451/python-code-to-calculate-angle-between-three-point-using-their-3d-coordinates
    # a, b and c : points as np.array([x, y, z]) 
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def landmark_to_np(landmark):
  # Create a numpy array of the landmark positions
  landmark_np = np.array([landmark.x, landmark.y, landmark.z])
  return landmark_np

# Calculate a rotation matrix that will take a vector and rotate it so that Y points up
def calculate_y_up_matrix(v):
   # Normalize the vector
    v = v / np.linalg.norm(v)

    # Compute the rotation axis
    axis = np.cross(v, np.array([0.0, 1.0, 0.0]))
    axis = axis / np.linalg.norm(axis)

    # Compute the rotation angle
    angle = -np.arccos(np.dot(v, np.array([0.0, 1.0, 0.0])))
    
    # Compute the rotation matrix using the axis-angle representation
    axis_crossproduct_matrix = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # Compute final rotation matrix
    rotation_matrix = (
        np.eye(3) +
        np.sin(angle) * axis_crossproduct_matrix +
        (1 - np.cos(angle)) * np.dot(axis_crossproduct_matrix, axis_crossproduct_matrix)
    )

    return rotation_matrix


  
def calculate_pose_angles(pose_world_landmarks):
  # Grab our points of interest for easy access
  right_elbow = pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
  right_wrist = pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
  right_shoulder = pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
  left_shoulder = pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
  right_hip = pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

  # Calculate the angle of the right elbow joint in the Z=0 plane (Front View)
  right_elbow_angle = angle(np.array([right_shoulder.x, right_shoulder.y]), np.array([right_elbow.x, right_elbow.y]), np.array([right_wrist.x, right_wrist.y]))
  
  # Invert angle to match robot arm
  right_elbow_angle = 180.0 - right_elbow_angle

  # Calculate shoulder Yaw in the Z=0 plane
  right_shoulder_yaw = angle(np.array([right_hip.x, right_hip.y]), np.array([right_shoulder.x, right_shoulder.y]), np.array([right_elbow.x, right_elbow.y]))
  
  # Calculate shoulder Pitch
  yaw_cutoff = 30.0
  if (right_shoulder_yaw < yaw_cutoff or right_shoulder_yaw > 180.0-yaw_cutoff):
      # Use the X=0 plane (side view) to calculate the pitch
      right_shoulder_pitch = angle(np.array([right_hip.z, right_hip.y]), np.array([right_shoulder.z, right_shoulder.y]), np.array([right_elbow.z, right_elbow.y]))
      pitchmode = "Side View"
  else:
      # Use the Y=0 plane (top view) to calculate the pitch
      right_shoulder_pitch = 180.0-angle(np.array([right_elbow.x, right_elbow.z]), np.array([right_shoulder.x, right_shoulder.z]), np.array([left_shoulder.x, left_shoulder.z]))
      pitchmode = "Top View"
  
  return right_elbow_angle,right_shoulder_yaw,right_shoulder_pitch,pitchmode

def calculate_finger_angles(joint_angles, joint_xyz):
  
  # First finger, fore or index
  # Angles calculated correspond to knuckle flex, knuckle yaw and long tendon length for all fingers,
  # note difference in knuckle yaw for little
  joint_angles[0] = angle(joint_xyz[0], joint_xyz[5], joint_xyz[8])
  joint_angles[1] = angle(joint_xyz[9], joint_xyz[5], joint_xyz[6])
  joint_angles[2] = angle(joint_xyz[5], joint_xyz[6], joint_xyz[7])
  #print(int(joint_angles[0]), int(joint_angles[1]), int(joint_angles[2]))
  
  # Second finger, middle
  joint_angles[3] = angle(joint_xyz[0], joint_xyz[9], joint_xyz[12])
  joint_angles[4] = angle(joint_xyz[13], joint_xyz[9], joint_xyz[10])
  joint_angles[5] = angle(joint_xyz[9], joint_xyz[10], joint_xyz[11])
  #print(joint_angles[3], joint_angles[4], joint_angles[5])

  # Third finger, ring
  joint_angles[6] = angle(joint_xyz[0], joint_xyz[13], joint_xyz[16])
  joint_angles[7] = angle(joint_xyz[9], joint_xyz[13], joint_xyz[14])
  joint_angles[8] = angle(joint_xyz[13], joint_xyz[14], joint_xyz[15])
  #print(joint_angles[6], joint_angles[7], joint_angles[8])

  # Fourth finger, pinky
  joint_angles[9] = angle(joint_xyz[0], joint_xyz[17], joint_xyz[20])
  joint_angles[10] = angle(joint_xyz[13], joint_xyz[17], joint_xyz[18])
  joint_angles[11] = angle(joint_xyz[17], joint_xyz[18], joint_xyz[19])
  #print(joint_angles[9], joint_angles[10], joint_angles[11])

  # Thumb, bit of a guess for basal rotation might be better automatic
  joint_angles[12] = angle(joint_xyz[1], joint_xyz[2], joint_xyz[3])
  joint_angles[13] = angle(joint_xyz[2], joint_xyz[1], joint_xyz[5])
  joint_angles[14] = angle(joint_xyz[2], joint_xyz[3], joint_xyz[4])
  joint_angles[15] = angle(joint_xyz[9], joint_xyz[5], joint_xyz[2])
  #print(joint_angles[12], joint_angles[13], joint_angles[14], joint_angles[15])

  return joint_angles


def drawDebugViews(results, hand_points, hcp, hncp, hand_points_norm, pitchmode):
  # Create images for the 3 planar projection views
  window_size = 256
  xaxis = np.zeros((window_size, window_size, 3), np.uint8)
  xaxis[:] = (0, 0, 64)
  yaxis = np.zeros((window_size, window_size, 3), np.uint8)
  yaxis[:] = (0, 64, 0)
  zaxis = np.zeros((window_size, window_size, 3), np.uint8)
  zaxis[:] = (64, 0, 0)
  
  # Draw planar projection views for debugging
  if results.pose_world_landmarks is not None:
      last = None
      names = ['Wrist','Elbow','RSho','RHip', 'LHip', 'LSho']
      joints = [mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_SHOULDER]
      
      # Put all the world landmark positions for the joints into numpi array
      world_landmarks = np.array([[results.pose_world_landmarks.landmark[i].x, results.pose_world_landmarks.landmark[i].y, results.pose_world_landmarks.landmark[i].z] for i in joints])
      
      # Center the landmarks in the window
      world_landmarks += 0.5

      # Scale the landmarks to fit in the window
      world_landmarks *= window_size

      # Estimate center of torso
      cp = (world_landmarks[2]+world_landmarks[4])/2.0

      # Compute the normal to the center of the torso
      normal = np.cross(world_landmarks[3]-world_landmarks[2],world_landmarks[4]-world_landmarks[2])
      normal /= np.linalg.norm(normal)
      ncp = cp+(normal*20.0)

      # To bump the rendering down a bit to use the window better
      yoffset = int(window_size*.25)

      # To integers
      world_landmarks = world_landmarks.astype(int)
      cp = cp.astype(int)
      ncp = ncp.astype(int)
      
      for idx in range(len(world_landmarks)):
          landmark = world_landmarks[idx]
          #print(i, landmark)
          cv2.circle(zaxis, (landmark[0], landmark[1]+yoffset), 2, (255, 255, 255), -1)
          cv2.circle(yaxis, (landmark[0], landmark[2]+yoffset), 2, (255, 255, 255), -1)
          cv2.circle(xaxis, (landmark[2], landmark[1]+yoffset), 2, (255, 255, 255), -1)
          cv2.putText(zaxis, names[idx], (landmark[0], landmark[1]+yoffset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
          cv2.putText(yaxis, names[idx], (landmark[0], landmark[2]+yoffset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
          cv2.putText(xaxis, names[idx], (landmark[2], landmark[1]+yoffset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
          if last is not None:
              cv2.line(zaxis, (landmark[0], landmark[1]+yoffset), (last[0], last[1]+yoffset), (255, 255, 255), 1)
              cv2.line(yaxis, (landmark[0], landmark[2]+yoffset), (last[0], last[2]+yoffset), (255, 255, 255), 1)
              cv2.line(xaxis, (landmark[2], landmark[1]+yoffset), (last[2], last[1]+yoffset), (255, 255, 255), 1)
          last = landmark

          # Draw debug info for the shoulder pitch
          if pitchmode == "Top View":
              # Draw a yellow line between the shoulder and the elbow
              cv2.line(yaxis, (world_landmarks[2][0], world_landmarks[2][2]+yoffset), (world_landmarks[1][0], world_landmarks[1][2]+yoffset), (0, 255, 255), 2)

              # Draw a yellow line between the right shoulder and the left shoulder
              cv2.line(yaxis, (world_landmarks[2][0], world_landmarks[2][2]+yoffset), (world_landmarks[5][0], world_landmarks[5][2]+yoffset), (0, 255, 255), 2)

          else:
              # Draw a yellow line between the shoulder and the elbow
              cv2.line(xaxis, (world_landmarks[2][2], world_landmarks[2][1]+yoffset), (world_landmarks[1][2], world_landmarks[1][1]+yoffset), (0, 255, 255), 2)
              
              # Draw a yellow line between the sholder and the hip
              cv2.line(xaxis, (world_landmarks[2][2], world_landmarks[2][1]+yoffset), (world_landmarks[3][2], world_landmarks[3][1]+yoffset), (0, 255, 255), 2)
              
          # Draw debug info for the elbow angle
          # Draw a cyan line between the shoulder and the elbow
          cv2.line(zaxis, (world_landmarks[2][0]+2, world_landmarks[2][1]+yoffset+2), (world_landmarks[1][0]+2, world_landmarks[1][1]+yoffset+2), (255, 255, 0), 2)
          # Draw a cyan line between the elbow and the wrist
          cv2.line(zaxis, (world_landmarks[1][0], world_landmarks[1][1]+yoffset), (world_landmarks[0][0], world_landmarks[0][1]+yoffset), (255, 255, 0), 2)
          
          # Draw debug info for the shoulder yaw
          # Draw a magenta line between the shoulder and the hip
          cv2.line(zaxis, (world_landmarks[2][0], world_landmarks[2][1]+yoffset), (world_landmarks[3][0], world_landmarks[3][1]+yoffset), (255, 0, 255), 2)
          # Draw a magenta line between the shoulder and the elbow
          cv2.line(zaxis, (world_landmarks[2][0], world_landmarks[2][1]+yoffset), (world_landmarks[1][0], world_landmarks[1][1]+yoffset), (255, 0, 255), 2)

          # Draw torso center in each view
          cv2.circle(zaxis, (cp[0], cp[1]+yoffset), 2, (255, 255, 0), -1)
          cv2.circle(yaxis, (cp[0], cp[2]+yoffset), 2, (255, 255, 0), -1)
          cv2.circle(xaxis, (cp[2], cp[1]+yoffset), 2, (255, 255, 0), -1)
          
          # Draw normal line
          cv2.line(zaxis, (cp[0], cp[1]+yoffset), (ncp[0], ncp[1]+yoffset), (255, 255, 0), 2)
          cv2.line(yaxis, (cp[0], cp[2]+yoffset), (ncp[0], ncp[2]+yoffset), (255, 255, 0), 2)
          cv2.line(xaxis, (cp[2], cp[1]+yoffset), (ncp[2], ncp[1]+yoffset), (255, 255, 0), 2)


      if results.right_hand_landmarks is not None:  
        
        # Translate the points for rendering in center of screen
        hand_points += 0.5
        hcp += 0.5
        hncp += 0.5
        hand_points_norm *= 0.5 # Scale down the normalized points
        hand_points_norm += 0.5

        # Scale the landmarks to fit in the window
        hand_points *= window_size
        hcp *= window_size
        hncp *= window_size
        hand_points_norm *= window_size

        # To integers for OpenCV drawing
        hand_points = hand_points.astype(int)
        hncp = hncp.astype(int)
        hcp = hcp.astype(int)
        hand_points_norm = hand_points_norm.astype(int)
        
        # Draw hand points in each view, with unrotated hand in lower right of window
        for i in range(21):
          cv2.circle(zaxis, (hand_points[i][0], hand_points[i][1]+yoffset), 2, (255, 255, 255), -1)
          cv2.circle(yaxis, (hand_points[i][0], hand_points[i][2]+yoffset), 2, (255, 255, 255), -1)
          cv2.circle(xaxis, (hand_points[i][2], hand_points[i][1]+yoffset), 2, (255, 255, 255), -1)

          cv2.circle(zaxis, (hand_points_norm[i][0]+100, hand_points_norm[i][1]+100), 2, (0, 255, 255), -1)
          cv2.circle(yaxis, (hand_points_norm[i][0]+100, hand_points_norm[i][2]+100), 2, (0, 255, 255), -1)
          cv2.circle(xaxis, (hand_points_norm[i][2]+100, hand_points_norm[i][1]+100), 2, (0, 255, 255), -1)

        # Draw hand center in each view
        cv2.circle(zaxis, (hcp[0], hcp[1]+yoffset), 2, (255, 255, 0), -1)
        cv2.circle(yaxis, (hcp[0], hcp[2]+yoffset), 2, (255, 255, 0), -1)
        cv2.circle(xaxis, (hcp[2], hcp[1]+yoffset), 2, (255, 255, 0), -1)

        # Draw coordinate system for hand center
        cols = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i,pt in enumerate(hncp):
            cv2.line(zaxis, (hcp[0], hcp[1]+yoffset), (pt[0], pt[1]+yoffset), cols[i], 2)
            cv2.line(yaxis, (hcp[0], hcp[2]+yoffset), (pt[0], pt[2]+yoffset), cols[i], 2)
            cv2.line(xaxis, (hcp[2], hcp[1]+yoffset), (pt[2], pt[1]+yoffset), cols[i], 2)
                

  # Show the debug views
  cv2.imshow('YZ Plane (Side View)',xaxis)
  cv2.imshow('XZ Plane (Top View)',yaxis)
  cv2.imshow('XY Plane (Front View)',zaxis)

  
  # Move the windows over to the left side of the main window and stack them vertically
  # Only on the first apparence of the debug views, in case the user overrides the positions
  if not hasattr(drawDebugViews, "views_moved"):
    cv2.moveWindow('YZ Plane (Side View)', 0, 512)
    cv2.moveWindow('XZ Plane (Top View)', 0, 256)
    cv2.moveWindow('XY Plane (Front View)', 0, 0)
    drawDebugViews.views_moved = True

  


# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nodebug', action='store_true', help='Disable debug views')
parser.add_argument('--force-webcam', action='store_true', help='Force webcam input even if OAKD device is present')
parser.add_argument('--oakd-capture-width', type=int, default=3840, help='Set OAKD capture width (default=3840)')
parser.add_argument('--oakd-capture-height', type=int, default=2160, help='Set OAKD capture height (default=2160)')
parser.add_argument('--webcam-capture-width', type=int, default=1920, help='Set webcam capture width (default=1920)')
parser.add_argument('--webcam-capture-height', type=int, default=1080, help='Set webcam capture height (default=1080)')
parser.add_argument('--preview-width', type=int, default=1280, help='Set preview width (default=1280)')
parser.add_argument('--preview-height', type=int, default=720, help='Set preview height (default=720)')
args = parser.parse_args()
show_debug_views = not args.nodebug

# For the camera, we look to see if there is a DepthAI device connected (OAK-D camera) and prefer that by default
# If not, we fall through to webcam

# Start with DepthAI camera
cvcam = depthai_cam.DepthAICam(width=args.oakd_capture_width,height=args.oakd_capture_height) # Default to 4K OAK-D camera
if (parser.parse_args().force_webcam or cvcam.is_depthai_device_available() is False):
   # Fall back to default webcam
   print("No DepthAI device available, falling back to webcam.")
   cvcam = opencv_cam.OpenCVCam(width=args.webcam_capture_width,height=args.webcam_capture_height)

# Start the video strema
if cvcam.start() is False:
  print("Failed to start video capture - exiting.")
  exit()

# Process the video stream
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
  while cvcam.is_opened():

    # Start a frame time counter
    frame_time = cv2.getTickCount()

    success, image = cvcam.read_frame()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_pose_landmarks_style())
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles
        .get_default_hand_landmarks_style())   
    
    # Once mediapipe has processed the frame, we can scale it down for display
    image = cv2.resize(image, (args.preview_width,args.preview_height))
    
    # Create array with enough space for all calculated angles
    joint_angles = np.zeros(23)

    # Calculate hand angles
    hand_points = None
    wrist_rotation = 0.0
    is_valid_frame = True
    
    if results.right_hand_landmarks is not None:
      hand_landmarks = results.right_hand_landmarks
          
      # Create a numpy array of the hand landmarks
      hand_points = np.array([[hand_landmarks.landmark[i].x, hand_landmarks.landmark[i].y, hand_landmarks.landmark[i].z] for i in range(21)]) 

      # The idea here is to rotate the hand so that the middle finger points up to make it more consistent to pick off
      # angles including the rotation angle around the wrist which isn't easily obtained otherwise.

      # Make a copy of the array for the normalized positions
      hand_points_norm = deepcopy(hand_points)
      hand_points_norm -= hand_points_norm[0] # Move all points relative to the wrist

      # Compute up vector for the hand
      normalized_up = hand_points_norm[mp_hand.HandLandmark.WRIST] - hand_points_norm[mp_hand.HandLandmark.MIDDLE_FINGER_MCP]
      normalized_up /= np.linalg.norm(normalized_up)

      # Compute matrix to rotate the hand so that the middle finger points up
      hand_rotation_matrix = calculate_y_up_matrix(normalized_up)

      # Transform the hand points to the new coordinate system
      hand_points_norm = np.matmul(hand_points_norm, hand_rotation_matrix)

      # Use normalized points to calculate hand rotation in the Y=0 plane
      index = hand_points_norm[mp_hand.HandLandmark.INDEX_FINGER_MCP]
      pinky = hand_points_norm[mp_hand.HandLandmark.PINKY_MCP]
      zaxis = hand_points_norm[mp_hand.HandLandmark.PINKY_MCP] + np.array([0.0,0.0,1.0]) 
      rel = index - pinky

      # Depending on which side of the hand the thumb is on, the angle will be positive or negative
      # These angles are set up to mimic the results from the previous demo, but can be adjusted if needed
      wrist_rotation = 180-angle(
        np.array([index[0], index[2]]),
        np.array([pinky[0], pinky[2]]),
        np.array([zaxis[0], zaxis[2]]))
      
      if (rel[0] < 0): # Look at X axis direction beteen index finger mcp and pinky to determine direction of hand
        wrist_rotation = 360-wrist_rotation
    
      
      # Calculate finger joint angles
      hand_angles = calculate_finger_angles(joint_angles, hand_points_norm)

      # Calculate wrist angles
      
      # Model does not seem to be in the same origin as the pose, so we need to translate
      # the hand points to the pose frame of reference if we want to compare them
      pose_wrist = landmark_to_np(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST])
      delta = pose_wrist - hand_points[0]
      hand_points += delta

      # Estimate the center of the palm
      hcp = (hand_points[0]+hand_points[5]+hand_points[17])/3.0

      # Compute the normal to the center of the palm
      hup = hand_points[9]-hand_points[0]
      hup /= np.linalg.norm(hup)

      hright = hand_points[5]-hand_points[17]
      hright /= np.linalg.norm(hright)

      hand_normal = np.cross(hright, hup)
      hand_normal /= np.linalg.norm(hand_normal)

      hncp = np.array([hcp+hright*0.2, hcp+hup*0.2, hcp+hand_normal*0.2])

      # Wrist pitch
      right_elbow = landmark_to_np(results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW])
      fk = hand_points[0]+hand_normal
      joint_angles[16] = angle(fk, hand_points[0], right_elbow)-30.0  # The 30.0 is an empirical fudge factor - I don't know why this angle is offset
      
      # Use Middle finger calculate wrist yaw
      wrist_yaw = angle(hand_points[mp_hand.HandLandmark.MIDDLE_FINGER_MCP], hand_points[mp_hand.HandLandmark.WRIST], np.array([1.0,0,0]))
      joint_angles[17] = wrist_yaw

      # Wrist roll
      joint_angles[18] = wrist_rotation
      #print(int(joint_angles[16]), int(joint_angles[17]), int(joint_angles[18]))
    else: # No hand detected
      is_valid_frame = False
    
    # Calculate pose angles
    if results.pose_world_landmarks is not None:
        # Grab our points of interest for easy access
        right_elbow_angle,right_shoulder_yaw,right_shoulder_pitch,pitchmode = calculate_pose_angles(results.pose_world_landmarks)

        joint_angles[19] = right_shoulder_pitch
        joint_angles[20] = right_shoulder_yaw
        joint_angles[21] = 0.0 # Right shoulder roll TBD
        joint_angles[22] = right_elbow_angle
    else: # No arm detected
        is_valid_frame = False

    # Valid data frame?
    if (is_valid_frame):
      joint_angles = joint_angles.astype(int)
      print(joint_angles)

    # Calculate a point to approximate the center of the torso at the midpoint between the left shoulder and right hip
    if results.pose_landmarks is not None:
        right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        torso_center = np.array([right_hip.x+left_shoulder.x, right_hip.y+left_shoulder.y, right_hip.z+left_shoulder.z])/2.0
        
        torso_render = torso_center*np.array([image.shape[1], image.shape[0], 0.0])
        torso_render = torso_render.astype(int)

        cv2.circle(image, (torso_render[0],torso_render[1]), 4, (255, 255, 0), -1)
    

    # Flip the image horizontally for a selfie-view display.
    flipped_image = cv2.flip(image, 1)

    # Add annotations after flipping the image
    if is_valid_frame and show_debug_views:
        
        # Screen points for drawing text
        screen_right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        screen_right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        screen_right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        # Elbow
        cv2.rectangle(flipped_image, (int(image.shape[1] - screen_right_elbow.x * image.shape[1]) + 5, int(screen_right_elbow.y * image.shape[0]) - 15), (int(image.shape[1] - screen_right_elbow.x * image.shape[1]) + 100, int(screen_right_elbow.y * image.shape[0]) + 5), (0, 0, 0), -1)
        cv2.putText(flipped_image, "Elb: {:.2f}".format(right_elbow_angle), (5+int(image.shape[1] - screen_right_elbow.x * image.shape[1]), int(screen_right_elbow.y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(screen_right_elbow.visibility), 1, cv2.LINE_AA)

        # Wrist
        cv2.rectangle(flipped_image, (int(image.shape[1] - screen_right_wrist.x * image.shape[1]) + 5, int(screen_right_wrist.y * image.shape[0]) - 15), (int(image.shape[1] - screen_right_wrist.x * image.shape[1]) + 100, int(screen_right_wrist.y * image.shape[0]) + 5), (0, 0, 0), -1)
        cv2.putText(flipped_image, "Wri: {:.2f}".format(wrist_rotation), (5+int(image.shape[1] - screen_right_wrist.x * image.shape[1]), int(screen_right_wrist.y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(screen_right_wrist.visibility), 1, cv2.LINE_AA)
        
        # Shoulder
        cv2.rectangle(flipped_image, (int(image.shape[1] - screen_right_shoulder.x * image.shape[1]) + 5, int(screen_right_shoulder.y * image.shape[0]) - 15), (int(image.shape[1] - screen_right_shoulder.x * image.shape[1]) + 200, int(screen_right_shoulder.y * image.shape[0]) + 5), (0, 0, 0), -1)
        cv2.putText(flipped_image, "Sh Yaw:{:.2f} Pit:{:.2f}".format(5+right_shoulder_yaw, right_shoulder_pitch), (int(image.shape[1] - screen_right_shoulder.x * image.shape[1]), int(screen_right_shoulder.y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(screen_right_shoulder.visibility), 1, cv2.LINE_AA)

    
    
    
    # Debug output
    if (is_valid_frame and show_debug_views):
      drawDebugViews(results, hand_points, hcp, hncp, hand_points_norm, pitchmode)

    # Calculate the frame rate
    frame_rate = cv2.getTickFrequency() / (cv2.getTickCount() - frame_time)

    # Display frame rate on frame
    cv2.rectangle(flipped_image, (0, 0), (200, 40), (0, 0, 0), -1)
    cv2.putText(flipped_image, "FPS: {:.2f}".format(frame_rate), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    # Render view    
    cv2.imshow('MediaPipe Pose', flipped_image)

        
    # Keyboard input
    key = cv2.waitKey(1) & 0xFF    
    if key == 27:
      break
    elif key == ord('d'):
      show_debug_views = not show_debug_views
      if (not show_debug_views):
        cv2.destroyWindow('YZ Plane (Side View)')
        cv2.destroyWindow('XZ Plane (Top View)')
        cv2.destroyWindow('XY Plane (Front View)')
    
      
# Clean up camera and windows
cvcam.stop()
cv2.destroyAllWindows()