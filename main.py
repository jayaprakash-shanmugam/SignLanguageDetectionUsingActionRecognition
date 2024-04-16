import cv2
import streamlit as st
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
import numpy as np

# Set page title and favicon
st.set_page_config(page_title="YOLO Object Detection", page_icon=":mag:")

# Load YOLO model
model = YOLO("best.pt")

# Function to perform object detection on an image
def detect_objects(image):
    # Save the image
    image.save("uploaded_image.jpg")

    # Perform object detection
    model.predict(source="uploaded_image.jpg", show=False, save=True)

    # Read the detected image
    detected_image = Image.open("runs/detect/predict/uploaded_image.jpg")
    return detected_image

# Page title and description
st.title("YOLO Object Detection")
st.write("Choose an option below to perform object detection.")

# Create dropdown for options with custom CSS class
option = st.selectbox("", ("Upload Image", "Capture Photo", "Live Video"), key="dropdown", 
                      help="Select an option")

if option == "Upload Image":
    st.subheader("Upload Image")
    st.write("Choose an image file to upload.")
    uploaded_file = st.file_uploader("", type=["jpg", "png"])

    if uploaded_file is not None:
        image = Image.open(BytesIO(uploaded_file.read()))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Detect"):
            detected_image = detect_objects(image)
            st.image(detected_image, caption="Predicted Image", use_column_width=True)

elif option == "Capture Photo":
    st.subheader("Capture Photo")
    st.write("Click the button below to capture a photo.")
    capture_button = st.button("Capture Photo")

    if capture_button:
        # Capture video from the webcam
        cap = cv2.VideoCapture(0)

        # Check if the webcam is opened successfully
        if not cap.isOpened():
            st.error("Failed to open camera.")
        else:
            # Capture a single frame
            ret, frame = cap.read()

            # Check if the frame is captured successfully
            if not ret:
                st.error("Failed to capture frame from camera.")
            else:
                # Convert the frame to PIL image format
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption="Captured Image", use_column_width=True)
                if st.button("Detect"):
                    detected_image = detect_objects(image)
                    st.image(detected_image, caption="Predicted Image", use_column_width=True)

        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

elif option == "Live Video":
    st.subheader("Live Video Detection")
    st.write("Press the 'Start' button to begin object detection on the live video stream.")
    
    # Placeholder for displaying the live video feed
    video_placeholder = st.empty()
    
    # Start button to initiate live video detection
    start_button = st.button("Start")
    
    if start_button:
        # Initialize video capture
        cap = cv2.VideoCapture(0)
        
        # Check if the webcam is opened successfully
        if not cap.isOpened():
            st.error("Failed to open camera.")
        else:
            # Display the live video feed
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera.")
                    break
                
                # Perform object detection on the frame
                detected_frame = model.predict(source=frame, show=False, save=False)
                
                # Display the detected frame
                video_placeholder.image(detected_frame, channels="BGR", use_column_width=True)
                
                # Check if the user wants to stop
                if st.button("Stop"):
                    break
            
            # Release the webcam and close the window
            cap.release()
            cv2.destroyAllWindows()
