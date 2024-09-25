import cv2
import numpy as np
from ultralytics import YOLO
import datetime
import time
import platform

if platform.system() == "Windows":
    import winsound

model = YOLO("yolov8s.pt")
intrusion_log = []
intrusion_detected = False

cap = cv2.VideoCapture(0)

def detect_people():
    global intrusion_log, intrusion_detected
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("웹캠에서 프레임을 가져오지 못했습니다.")
                break

            results = model.predict(source=frame, show=True)
            intrusion_detected = False

            for result in results:
                for r in result.boxes.data:
                    class_id = int(r[-1])
                    if class_id == 0:
                        intrusion_detected = True
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_entry = f"침입 감지: {timestamp}"
                        intrusion_log.append(log_entry)

                        if platform.system() == "Windows":
                            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)

            time.sleep(1)
        except Exception as e:
            print(f"YOLO 감지 중 오류: {e}")


