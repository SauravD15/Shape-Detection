import cv2
import numpy as np

def detect_circle(frame):
  # Convert the live video frame to grayscale
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

  # Apply edge detection
  edges = cv2.Canny(gray, 100, 200)

  # Apply Hough transform to detect circles
  circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=100, param2=30, minRadius=0, maxRadius=0)

  # Filter and postprocess the circle detection results
  circles = np.uint16(np.around(circles))
  for circle in circles[0, :]:
    # Draw the circle on the frame
    cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)

  # Output the detected circle shape, along with its center coordinates and radius
  return circles

# Read the live video feed
cap = cv2.VideoCapture(0)

# Loop over the video frames
while True:
  # Read the next frame
  ret, frame = cap.read()

  # Detect circles in the frame
  circles = detect_circle(frame)

  # Display the frame with the detected circles
  cv2.imshow('frame', frame)

  # Check if the user pressed the ESC key
  if cv2.waitKey(1) & 0xFF == 27:
    break

# Release the video capture object
cap.release()

# Destroy all windows
cv2.destroyAllWindows()