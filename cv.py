import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion recognition model (Update this with the correct path)
emotion_model_path = "C:/Users/yashm/Downloads/archive (1)/face_model.h5"  # Replace with your model path
emotion_classifier = load_model(emotion_model_path, compile=False)

# Define the list of emotion labels (based on FER-2013)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam video capture (0 for the default webcam)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess the face ROI (Region of Interest) for emotion prediction
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))  # Resize to 48x48 as required by the model
        face_roi = face_roi.astype("float") / 255.0  # Normalize the image
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)  # Add batch dimension

        # Predict emotion
        emotion_prediction = emotion_classifier.predict(face_roi)
        max_index = np.argmax(emotion_prediction[0])  # Get the emotion with the highest probability
        emotion_label = emotion_labels[max_index]

        # Display the emotion label on the frame
        label_position = (x, y - 10)
        cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame with emotion labels
    cv2.imshow('Live Emotion and Face Detection', frame)

    # Press 'q' to quit the video capture loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()
