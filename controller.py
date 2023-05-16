import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

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
  

# For webcam input:
cap = cv2.VideoCapture(0)
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    # Calculate the angle of the right elbow joint
    if results.pose_landmarks is not None:
        right_elbow = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        screen_right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_wrist = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        right_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        right_elbow_angle = angle(np.array([right_shoulder.x, right_shoulder.y]), np.array([right_elbow.x, right_elbow.y]), np.array([right_wrist.x, right_wrist.y]))
        
        # Invert angle to match robot arm
        right_elbow_angle = 180.0 - right_elbow_angle


    # Calculate the pitch/yaw of the right shoulder joint
    if results.pose_landmarks is not None:
        right_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_shoulder = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        screen_right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        right_hip = results.pose_world_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
        
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
        
        
    # Flip the image horizontally for a selfie-view display.
    flipped_image = cv2.flip(image, 1)

    # Add annotations after flipping the image
    if results.pose_landmarks is not None:
        
        # Draw dark box behind the text
        cv2.rectangle(flipped_image, (int(image.shape[1] - screen_right_elbow.x * image.shape[1]) + 5, int(screen_right_elbow.y * image.shape[0]) - 15), (int(image.shape[1] - screen_right_elbow.x * image.shape[1]) + 100, int(screen_right_elbow.y * image.shape[0]) + 5), (0, 0, 0), -1)
        cv2.putText(flipped_image, "Elb: {:.2f}".format(right_elbow_angle), (5+int(image.shape[1] - screen_right_elbow.x * image.shape[1]), int(screen_right_elbow.y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(right_elbow.visibility), 1, cv2.LINE_AA)
        
        cv2.rectangle(flipped_image, (int(image.shape[1] - screen_right_shoulder.x * image.shape[1]) + 5, int(screen_right_shoulder.y * image.shape[0]) - 15), (int(image.shape[1] - screen_right_shoulder.x * image.shape[1]) + 200, int(screen_right_shoulder.y * image.shape[0]) + 5), (0, 0, 0), -1)
        cv2.putText(flipped_image, "Sh Yaw:{:.2f} Pit:{:.2f}".format(5+right_shoulder_yaw, right_shoulder_pitch), (int(image.shape[1] - screen_right_shoulder.x * image.shape[1]), int(screen_right_shoulder.y * image.shape[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, visibilityToColour(right_shoulder.visibility), 1, cv2.LINE_AA)

        
        
    
    cv2.imshow('MediaPipe Pose', flipped_image)


    # Render orthogonal views of the landmarks

    # Create an empty image with a red background
    window_size = 256
    xaxis = np.zeros((window_size, window_size, 3), np.uint8)
    xaxis[:] = (0, 0, 127)
    yaxis = np.zeros((window_size, window_size, 3), np.uint8)
    yaxis[:] = (0, 127, 0)
    zaxis = np.zeros((window_size, window_size, 3), np.uint8)
    zaxis[:] = (127, 0, 0)
    
    # Print out all of the world landmarks and their names
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

        yoffset = int(window_size*.25)

        # To integers'
        world_landmarks = world_landmarks.astype(int)

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

    cv2.imshow('YZ Plane (Side View)',xaxis)
    cv2.imshow('XZ Plane (Top View)',yaxis)
    cv2.imshow('XY Plane (Front View)',zaxis)

    
        
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()