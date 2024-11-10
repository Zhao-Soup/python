import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion recognition model (Make sure you have the .h5 model file)
emotion_model_path = "C:/Users/yashm/Downloads/archive (1)/face_model.h5"
emotion_classifier = load_model(emotion_model_path, compile=False)

# Define the list of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Initialize webcam video capture
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Preprocess the face ROI for emotion detection
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.astype("float") / 255.0
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)

        # Predict emotion
        emotion_prediction = emotion_classifier.predict(face_roi)
        max_index = np.argmax(emotion_prediction[0])
        emotion_label = emotion_labels[max_index]

        # Display the label
        label_position = (x, y - 10)
        
        # Ensure we're only using ASCII-compatible characters (fallback to English)
        emotion_label = str(emotion_label).encode('ascii', 'ignore').decode()

        cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Emotion and Face Detection', frame)

    # Press 'q' to break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()


# # Load the Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Load the emotion recognition model (Make sure you have the .h5 model file)
# emotion_model_path = "C:/Users/yashm/Downloads/archive (1)/face_model.h5"
# emotion_classifier = load_model(emotion_model_path, compile=False)

# # Define the list of emotion labels
# emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# # Initialize webcam video capture
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert frame to grayscale for face detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         # Draw rectangle around detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#         # Preprocess the face ROI for emotion detection
#         face_roi = gray[y:y + h, x:x + w]
#         face_roi = cv2.resize(face_roi, (48, 48))
#         face_roi = face_roi.astype("float") / 255.0
#         face_roi = img_to_array(face_roi)
#         face_roi = np.expand_dims(face_roi, axis=0)

#         # Predict emotion
#         emotion_prediction = emotion_classifier.predict(face_roi)
#         max_index = np.argmax(emotion_prediction[0])
#         emotion_label = emotion_labels[max_index]

#         # Display the label
#         label_position = (x, y - 10)
#         cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Emotion and Face Detection', frame)

#     # Press 'q' to break the loop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()