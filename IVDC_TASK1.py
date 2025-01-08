import cv2
import numpy as np

# Initialize video capture with video file path
cap = cv2.VideoCapture("line.mp4")

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply binary thresholding to segment the black line
    _, thresholded = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (to remove noise)
    min_area = 500
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            # Draw the contour on the original frame
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

            # Calculate the center of the contour
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Mark the center
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                # Display the coordinates
                cv2.putText(frame, f"Center: ({cx}, {cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Display the result
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Thresholded", thresholded)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
