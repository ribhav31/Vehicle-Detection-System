# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# cap = cv2.VideoCapture("pexels-kelly-lacy-5669768 (2160p).mp4")

# # Create background subtractor
# algo = cv2.createBackgroundSubtractorMOG2()

# # Define the line parameters
# line_y = 1500
# line_start_x = 1400
# line_end_x = 2400
# line_thickness = 2
# car_count = 0

# while True:
#     ret, frame1 = cap.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian Blur
#     blur = cv2.GaussianBlur(gray, (3, 3), 5)

#     # Apply background subtraction
#     img_sub = algo.apply(blur)

#     # Dilate the resulting image
#     dilat = cv2.dilate(img_sub, np.ones((5, 5)))

#     # Define a kernel for morphological operations
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#     # Perform morphological closing
#     dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

#     # Find contours
#     contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours on the original frame
#     cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

#     # Draw the line on the frame
#     cv2.line(frame1, (line_start_x, line_y), (line_end_x, line_y), (0, 0, 255), line_thickness)

#     # Check if contours intersect with the line
#     for contour in contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         if y < line_y < (y + h) and line_start_x < x < line_end_x:
#             car_count += 1

#     # Display the result using Matplotlib
#     plt.imshow(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
#     plt.title(f'Car Count: {car_count}')
#     plt.show()

#     if cv2.waitKey(1) == 13:
#         break

# cv2.destroyAllWindows()
# cap.release()

# import cv2
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# cap = cv2.VideoCapture("pexels-kelly-lacy-5669768 (2160p).mp4")

# fig, ax = plt.subplots()
# image = ax.imshow(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))

# def update(frame):
#     ret, frame = cap.read()
#     if ret:
#         image.set_array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     return image,

# ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# plt.show()

# cap.release()

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# cap = cv2.VideoCapture("pexels-kelly-lacy-5669768 (2160p).mp4")

# # Create background subtractor
# algo = cv2.createBackgroundSubtractorMOG2()

# # Define the line parameters
# line_y = 1500
# line_start_x = 1400
# line_end_x = 2400
# line_thickness = 2
# car_count = 0

# fig, ax = plt.subplots()
# image = ax.imshow(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))

# def update(frame):
#     ret, frame1 = cap.read()
#     if not ret:
#         return

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian Blur
#     blur = cv2.GaussianBlur(gray, (3, 3), 5)

#     # Apply background subtraction
#     img_sub = algo.apply(blur)

#     # Dilate the resulting image
#     dilat = cv2.dilate(img_sub, np.ones((5, 5)))

#     # Define a kernel for morphological operations
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#     # Perform morphological closing
#     dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

#     # Find contours
#     contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours on the original frame
#     cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

#     # Draw the line on the frame
#     cv2.line(frame1, (line_start_x, line_y), (line_end_x, line_y), (0, 0, 255), line_thickness)

#     # Check if contours intersect with the line
#     for contour in contours:
#         (x, y, w, h) = cv2.boundingRect(contour)
#         if y < line_y < (y + h) and line_start_x < x < line_end_x:
#             car_count += 1

#     # Update the Matplotlib plot
#     image.set_array(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
#     ax.set_title(f'Car Count: {car_count}')

#     return image,

# ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)

# plt.show()

# cap.release()

# import cv2
# import numpy as np

# cap = cv2.VideoCapture("pexels-kelly-lacy-5669768 (2160p).mp4")

# # Create background subtractor
# algo = cv2.createBackgroundSubtractorMOG2()

# while True:
#     ret, frame1 = cap.read()
#     if not ret:
#         break

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#     # Apply Gaussian Blur
#     blur = cv2.GaussianBlur(gray, (3, 3), 5)

#     # Apply background subtraction
#     img_sub = algo.apply(blur)

#     # Dilate the resulting image
#     dilat = cv2.dilate(img_sub, np.ones((5, 5)))

#     # Define a kernel for morphological operations
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

#     # Perform morphological closing
#     dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)

#     # Find contours
#     contours, _ = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     # Draw contours on the original frame
#     cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

#     # Display the result
#     cv2.imshow('Detector', frame1)

#     if cv2.waitKey(1) == 13:
#         break

# cv2.destroyAllWindows()
# cap.release()

# import cv2

# # Load the video file
# filename = "pexels-kelly-lacy-5669768 (2160p).mp4"
# video_reader = cv2.VideoCapture(filename)

# # Define line parameters
# line_y = 1500
# line_start_x = 1400
# line_end_x = 2400
# line_thickness = 2
# car_count = 0

# # Process each frame and create an animation
# processed_frames = []
# while True:
#     ret, frame = video_reader.read()
#     if not ret:
#         break
    
#     # Convert frame to grayscale
#     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Find contours and draw the line
#     # (Note: This part needs to be implemented based on your specific requirements)
#     # Here, we assume that you have already implemented the contour detection and line drawing logic
    
#     # Update the car count and display it on the frame
#     car_count += 1  # Placeholder for car counting logic
#     cv2.putText(frame, f'Car Count: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

#     processed_frames.append(frame)

# # Release the video reader
# video_reader.release()

# # Save the processed frames to a new video file
# output_filename = "processed_video.mp4"
# video_writer = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame.shape[1], frame.shape[0]))
# for frame in processed_frames:
#     video_writer.write(frame)
# video_writer.release()

# # Display the processed video
# from IPython.display import Video
# Video(output_filename)

import cv2
from google.colab.patches import cv2_imshow

# Load the video file
filename = "production_id_4261469 (2160p).mp4"
cap = cv2.VideoCapture(filename)

# Define line parameters
line_y = 1500
line_start_x = 600
line_end_x = 3000

# Create background subtractor
algo = cv2.createBackgroundSubtractorMOG2()

# Initialize variables
car_count = 0

min_width_rect = 80
min_height_rect = 80

def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+ x1 
    cy = y + y1
    return cx , cy

detect = []


# Process each frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blur = cv2.GaussianBlur(gray, (3, 3), 5)

    # Apply background subtraction
    img_sub = algo.apply(blur)

    # Dilate the resulting image
    dilat = cv2.dilate(img_sub, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Perform morphological closing
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find contours
    contours, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Draw the line on the frame
    cv2.line(frame, (line_start_x, line_y), (line_end_x, line_y), (0, 0, 255), 2)

    # Check if contours intersect with the line
    # for contour in contours:
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     if y < line_y < (y + h) and line_start_x < x < line_end_x:
    #         car_count += 1

    for (i,c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(contours)
        validate_counter = (w>=min_width_rect) and (h>= min_height_rect)
        if not validate_counter:
            continue
        
        cv2.rectangle(frame,(x,y), (x+w, y+h), (0,0,255),2)

        center = center_handle(x,y,w,h)
        detect.append(center)

        cv2.circle(frame, center, 4, (0,0,255), -1)


    # Display the frame with car count
    cv2.putText(frame, f'Car Count: {car_count}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2_imshow(frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()



