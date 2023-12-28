import cv2
import pyautogui

# Load the pre-trained face and eye cascade classifiers
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Open the webcam
cap = cv2.VideoCapture(0)

# Move the mouse cursor out of the screen
pyautogui.moveTo(-100, -100)

# Define constants
EYES_OFF_SCREEN_MESSAGE = "Keep your eyes on the screen"
EYE_DISTANCE_THRESHOLD = 300  # Adjust as needed
CENTER_TOLERANCE = 20

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    eyes_off_screen = True
    eye_centroid = None

    for (x, y, w, h) in faces:
        # Get the region of interest (ROI) for the face
        roi_gray = gray[y:y + h, x:x + w]

        # Detect eyes in the face region
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            # Draw rectangles around the face and eyes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

            # Get the centroid of the detected eye
            eye_centroid = (x + ex + ew // 2, y + ey + eh // 2)

            # Check if the eye is within the distance threshold from the screen edges
            if (
                eye_centroid[0] >= EYE_DISTANCE_THRESHOLD
                and eye_centroid[0] <= frame.shape[1] - EYE_DISTANCE_THRESHOLD
                and eye_centroid[1] >= EYE_DISTANCE_THRESHOLD
                and eye_centroid[1] <= frame.shape[0] - EYE_DISTANCE_THRESHOLD
            ):
                eyes_off_screen = False

    # Display the appropriate message based on the eyes' position
    if eyes_off_screen:
        cv2.putText(frame, EYES_OFF_SCREEN_MESSAGE, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # If eyes are on the screen, determine the direction of gaze
    elif eye_centroid is not None:
        screen_center = frame.shape[1] // 2  # Get the x-coordinate of the screen center

        # Adjust the conditions for left and right gaze using the tolerance
        if eye_centroid[0] < screen_center - CENTER_TOLERANCE:
            cv2.putText(frame, "Right", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        elif eye_centroid[0] > screen_center + CENTER_TOLERANCE:
            cv2.putText(frame, "Left", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Center", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Eye Tracker', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
