import cv2
from ultralytics import YOLO
import numpy as np
from telegramBOT import send_notification
import mediapipe as mp

model = YOLO('yolov8n.pt')

# Инициализация MediaPipe для детекции поз
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       min_detection_confidence=0.2,
                       max_num_hands=2)

# Инициализация утилиты для визуализации
mp_drawing = mp.solutions.drawing_utils



def detect_hands(image, draw=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    if results.multi_hand_landmarks and draw:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )
    return image


def predict(chosen_model, img, classes=[0], conf=0.5):  # class 0 is for 'person'
    results = chosen_model.predict(img, classes=classes, conf=conf)
    return results

def predict_and_notify(chosen_model, img, classes=[0], conf=0.5, notified=False):
    results = predict(chosen_model, img, classes, conf)
    detected = False

    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:  # Проверка, что обнаружен человек (класс 0)
                detected = True
                # Рисование прямоугольника вокруг объекта
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
                # Добавление текста с именем класса
                cv2.putText(img, f"{result.names[int(box.cls[0])]}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)

    if detected and not notified:
        # Сохранение изображения и отправка уведомления в Telegram
        cv2.imwrite('detected_person.jpg', img)
        send_notification('detected_person.jpg')
        notified = True

    return img, results, notified

# Функция для обработки видео
def process_video():
    # Использование первого подключенного видеоустройства (обычно это веб-камера ноутбука)
    cap = cv2.VideoCapture(2)  # Изменение здесь: '0' указывает на веб-камеру

    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    notified = False

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image from camera")
                break

            # Детекция рук
            img = detect_hands(img)

            result_img, _, notified = predict_and_notify(model, img, classes=[0], conf=0.5, notified=notified)
            cv2.imshow("MediaPipe Hands + YOLO Detection", result_img)

            if cv2.waitKey(1) & 0xFF == 27:  # Выход по нажатию ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Запуск функции для обработки видео с веб-камеры
process_video()


