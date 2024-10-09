import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small, preprocess_input, decode_predictions

# Load pre-trained MobileNetV3Small model
model = MobileNetV3Small(weights='imagenet')

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Preprocess the frame for MobileNetV3
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(np.expand_dims(img, axis=0))
    
    # Make prediction
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]
    
    # Draw results on frame
    for i, (imagenet_id, label, score) in enumerate(decoded_preds):
        text = f"{label}: {score:.2f}"
        cv2.putText(frame, text, (10, 30 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Road Signal Detection', frame)
    
    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
