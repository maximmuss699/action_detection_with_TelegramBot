import cv2
from ultralytics import YOLO
import numpy as np
import torch
from telegramBOT import send_notification
import mediapipe as mp
from datetime import datetime

class ActionRecognition:
    def __init__(self, model_path, confidence_person=0.5, confidence_phone=0.6):
        self.model = YOLO(model_path)
        self.confidence_person = confidence_person
        self.confidence_phone = confidence_phone

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         min_detection_confidence=0.2,
                                         max_num_hands=2)
        self.mp_drawing = mp.solutions.drawing_utils
        self.notified = False

    def detect_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks and draw:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                )
        return image

    def predict(self, img):
        # Combine person and phone detection
        results_person = self.model.predict(img, classes=[0], conf=self.confidence_person)
        results_phone = self.model.predict(img, classes=[67], conf=self.confidence_phone)
        return results_person + results_phone

    def predict_and_notify(self, img):
        results = self.predict(img)
        phone_detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id in [0, 67]:  # Check if person or phone is detected
                    # Drwaw bounding box
                    cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                                  (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (0, 255, 0), 2)
                    # Add class name and confidence
                    confidence = box.conf.item() if isinstance(box.conf, torch.Tensor) else box.conf
                    cv2.putText(img, f"{result.names[class_id]} {confidence:.2f}",
                                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

                    if class_id == 67:  # Check if phone is detected
                        phone_detected = True

        if phone_detected and not self.notified:
            cv2.imwrite('detected_phone.jpg', img)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            caption = f"Phone detected at {current_time}"
            send_notification('detected_phone.jpg', caption)
            self.notified = True

        return img, results

    def process_video(self, video_source=2):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error: Could not open video device")
            return

        try:
            while True:
                success, img = cap.read()
                if not success:
                    print("Failed to capture image from camera")
                    break

                # Hand detection
                img = self.detect_hands(img)

                # Prediction and notification
                result_img, _ = self.predict_and_notify(img)
                cv2.imshow("MediaPipe Hands + YOLO Detection", result_img)

                if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    action_recognition = ActionRecognition('yolov8n.pt', confidence_person=0.5, confidence_phone=0.3)
    action_recognition.process_video()
