import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
def detect_emotion_from_image(image_path):
    # Use DeepFace to analyze the emotion of the image
    result = DeepFace.analyze(image_path, actions=['emotion'])
    
    # Extract the dominant emotion
    dominant_emotion = result[0]['dominant_emotion']
    print(f"Predicted Emotion: {dominant_emotion}")
    
    # Visualize the image with emotion label
    img = cv2.imread(image_path)
    cv2.putText(img, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Convert from BGR to RGB for displaying with Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.imshow(img_rgb)
    plt.axis('off')  # Hide axes
    plt.show()

    
def detect_emotion_in_video():
    # Initialize video capture (0 for default webcam)
    cap = cv2.VideoCapture(0)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Save the frame temporarily to use with DeepFace
        cv2.imwrite("temp_frame.jpg", frame)
        
        # Use DeepFace to analyze the emotion in the frame
        result = DeepFace.analyze("temp_frame.jpg", actions=['emotion'])
        
        # Extract the dominant emotion
        dominant_emotion = result[0]['dominant_emotion']
        
        # Display the dominant emotion on the video feed
        cv2.putText(frame, dominant_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the frame
        cv2.imshow('Emotion Detection', frame)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
