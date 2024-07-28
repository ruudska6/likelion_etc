from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import os
import time
import threading
import numpy as np
import absl.logging

# 경고 메시지 줄이기
absl.logging.set_verbosity(absl.logging.ERROR)

app = Flask(__name__)

camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
captured_landmarks = None

bad_posture_start_time = None
POSTURE_THRESHOLD_TIME = 5  # 5 seconds

def capture_landmarks():
    global captured_landmarks
    success, frame = camera.read()
    if success:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        if results.pose_landmarks:
            captured_landmarks = results.pose_landmarks.landmark
            filename = os.path.join('static', 'capture.jpg')
            cv2.imwrite(filename, frame)

def is_good_posture(landmarks, captured_landmarks):
    if not captured_landmarks:
        return False

    neck_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    chin_idx = mp_pose.PoseLandmark.NOSE.value  # Using NOSE as a proxy for chin

    neck_distance = abs(landmarks[neck_idx].x - captured_landmarks[neck_idx].x) + abs(landmarks[neck_idx].y - captured_landmarks[neck_idx].y)
    chin_distance = abs(landmarks[chin_idx].x - captured_landmarks[chin_idx].x) + abs(landmarks[chin_idx].y - captured_landmarks[chin_idx].y)

    threshold = 0.1
    return neck_distance < threshold and chin_distance < threshold

def draw_points(image, landmarks):
    left_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
    right_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
    chin_idx = mp_pose.PoseLandmark.NOSE.value  # Using NOSE as a proxy for chin

    # Draw points on the landmarks
    for idx in [left_shoulder_idx, right_shoulder_idx, chin_idx]:
        landmark = landmarks[idx]
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

    # Add additional points along the shoulder line
    left_shoulder = landmarks[left_shoulder_idx]
    right_shoulder = landmarks[right_shoulder_idx]

    # Adding more points along the shoulder line
    for i in range(1, 6):
        x = left_shoulder.x + (right_shoulder.x - left_shoulder.x) * (i / 6.0)
        y = left_shoulder.y + (right_shoulder.y - left_shoulder.y) * (i / 6.0)
        x = int(x * image.shape[1])
        y = int(y * image.shape[0])
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

def generate_frames():
    global bad_posture_start_time
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            good_posture = False
            if results.pose_landmarks:
                draw_points(image, results.pose_landmarks.landmark)

                good_posture = is_good_posture(results.pose_landmarks.landmark, captured_landmarks)

                if good_posture:
                    bad_posture_start_time = None
                else:
                    if bad_posture_start_time is None:
                        bad_posture_start_time = time.time()
                    elif time.time() - bad_posture_start_time >= POSTURE_THRESHOLD_TIME:
                        # 이미 알람이 울렸음을 확인할 수 있도록 상태를 유지
                        bad_posture_start_time = time.time()

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    capture_landmarks()
    return '', 204

@app.route('/posture_status')
def posture_status():
    global bad_posture_start_time
    if bad_posture_start_time is None:
        return jsonify(status="바른 자세입니다", alert=False)
    elif time.time() - bad_posture_start_time >= POSTURE_THRESHOLD_TIME:
        return jsonify(status="나쁜 자세입니다", alert=True)
    else:
        return jsonify(status="나쁜 자세입니다", alert=False)

if __name__ == "__main__":
    app.run(debug=True)
