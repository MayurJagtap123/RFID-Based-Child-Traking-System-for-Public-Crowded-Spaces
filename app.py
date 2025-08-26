from flask import Flask, render_template, Response, jsonify
import cv2
import face_recognition
import numpy as np
import os

app = Flask(__name__, static_folder='static')

video_capture = None  # Initialize camera variable

# Load known faces
def load_known_faces(folder_path):
    known_face_encodings = []
    known_face_names = []
    
    if not os.path.exists(folder_path):
        print("Error: Folder not found!")
        return [], []

    for filename in os.listdir(folder_path):
        if filename.endswith(('jpg', 'jpeg', 'png')):
            filepath = os.path.join(folder_path, filename)
            image = face_recognition.load_image_file(filepath)
            encodings = face_recognition.face_encodings(image)
            if encodings:  # Avoid adding empty encodings
                known_face_encodings.append(encodings[0])
                known_face_names.append(os.path.splitext(filename)[0])
    
    return known_face_encodings, known_face_names

# Load faces from the "images" folder
known_faces_folder = "images"
known_face_encodings, known_face_names = load_known_faces(known_faces_folder)

# Function to generate frames from the camera
def generate_frames():
    global video_capture
    process_every_nth_frame = 3  # Process every 3rd frame to optimize performance
    frame_count = 0

    while video_capture is not None:
        success, frame = video_capture.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % process_every_nth_frame != 0:
            continue  # Skip processing for better efficiency

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            face_names.append(name)

        # Draw rectangles and names on the frame
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def admin_login():
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')

@app.route('/trackuser')
def trackuser():
    return render_template('trackuser.html')

@app.route('/start_camera')
def start_camera():
    global video_capture
    if video_capture is None:
        video_capture = cv2.VideoCapture(0)
        if not video_capture.isOpened():
            return jsonify({"status": "failed", "message": "Cannot access the camera"})
    return jsonify({"status": "started"})

@app.route('/stop_camera')
def stop_camera():
    global video_capture
    if video_capture:
        video_capture.release()
        video_capture = None
    return jsonify({"status": "stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)