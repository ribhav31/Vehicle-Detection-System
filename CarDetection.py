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



