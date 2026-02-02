import os
import cv2
import numpy as np
import datetime
import csv
import joblib
from flask import Flask, render_template, request, redirect, session, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'your_secret_key'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_path = 'model/cnn_face_model.h5'
label_path = 'model/labels.pkl'

def load_face_model():
    return load_model(model_path), joblib.load(label_path)

def datetoday():
    return datetime.date.today().strftime('%Y-%m-%d')

def mark_attendance(username):
    date = datetoday()
    os.makedirs('Attendance', exist_ok=True)
    file_path = f'Attendance/Attendance-{date}.csv'
    is_new = not os.path.exists(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['Username', 'Date', 'Time'])
        now = datetime.datetime.now().strftime('%H:%M:%S')
        writer.writerow([username, date, now])

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    uname = request.form['username']
    passwd = request.form['password']
    role = request.form['role']
    if role == 'admin' and uname == 'admin' and passwd == 'admin':
        session['admin'] = uname
        return redirect('/admin')
    elif validate_user(uname, passwd):
        session['user'] = uname
        return redirect('/user')
    return 'Invalid credentials'

@app.route('/admin')
def admin_dashboard():
    if 'admin' in session:
        users = get_users()
        return render_template('admin_dashboard.html', users=users)
    return redirect('/')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        uid = request.form['userid']
        pwd = request.form['password']
        role = request.form['role']
        save_user(uid, uname, pwd,role)
        capture_faces(uid, uname,role)
        return redirect('/admin')
    return render_template('register.html')

@app.route('/train_model')
def train_model_route():
    train_model()
    return redirect(url_for('admin_dashboard', trained='yes'))

@app.route('/user')
def user_dashboard():
    if 'user' in session:
        username = session['user']
        model, labels = load_face_model()
        result, detected_name = recognize_and_mark(username, model, labels)
        if result:
            return render_template('user_dashboard.html', name=detected_name)
        return 'Face not recognized.'
    return redirect('/')

@app.route('/user_attendance')
def user_attendance():
    if 'user' in session:
        uname = session['user']
        attendance = []
        for file in os.listdir('Attendance'):
            if file.endswith('.csv'):
                with open(f'Attendance/{file}') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if row[0] == uname:
                            attendance.append(row)
        attendance.sort(key=lambda x: x[1], reverse=True)
        return render_template('attendance_history.html', records=attendance)
    return redirect('/')@app.route('/view_attendance')

def capture_faces(uid, uname,role):
    folder = f'static/faces/{uid}_{uname}_{role}'
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while count < 200:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            cv2.imwrite(f"{folder}/{count}.jpg", face)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Captured: {count}/20', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.imshow('Capturing', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def recognize_and_mark(username, model, labels):
    cap = cv2.VideoCapture(0)
    start_time = datetime.datetime.now()
    detected_user = None

    while (datetime.datetime.now() - start_time).seconds < 10000:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            face = face.reshape(1, 100, 100, 1) / 255.0
            pred = model.predict(face)
            pred_class = np.argmax(pred)
            for user_key, label in labels.items():
                if label == pred_class and username in user_key:
                    mark_attendance(username)
                    detected_user = user_key
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f'{user_key}', (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    break
            if detected_user:
                break
        cv2.imshow('Face Recognition - User', frame)
        if cv2.waitKey(1) == 27 or detected_user:
            break

    cap.release()
    cv2.destroyAllWindows()
    return bool(detected_user), detected_user

def train_model():
    data_dir = "static/faces"
    X, y, label_dict = [], [], {}
    label_id = 0
    for user in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user)
        if os.path.isdir(user_path):
            label_dict[user] = label_id
            for img_name in os.listdir(user_path):
                img = cv2.imread(os.path.join(user_path, img_name), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (100, 100))
                X.append(img)
                y.append(label_id)
            label_id += 1

    X = np.array(X).reshape(-1, 100, 100, 1) / 255.0
    y = to_categorical(np.array(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(y.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
    os.makedirs("model", exist_ok=True)
    model.save(model_path)
    joblib.dump(label_dict, label_path)

def save_user(uid, uname, pwd, role):
    os.makedirs('Database', exist_ok=True)
    with open('Database/users.txt', 'a') as f:
        f.write(f'{uid},{uname},{pwd},{role}\n')


def validate_user(uname, pwd):
    try:
        with open('Database/users.txt') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 4:
                    _, u, p, role = parts  # uid, username, password, role
                    if u == uname and p == pwd:
                        return True, role
    except:
        return False, None
    return False, None



def get_users():
    users = []
    try:
        with open('Database/users.txt') as f:
            for line in f:
                users.append(line.strip().split(','))
    except:
        pass
    return users

@app.route('/view_attendance')
def view_attendance():
    from collections import defaultdict
    counts = defaultdict(int)
    date = datetoday()
    file_path = f'Attendance/Attendance-{date}.csv'
    records = []
    labels = []
    data = []

    if os.path.exists(file_path):
        with open(file_path) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                records.append(row)
                counts[row[0]] += 1

    for name, count in counts.items():
        labels.append(name)
        data.append(count)

    return render_template('attendance.html', records=records, labels=labels, data=data)


if __name__ == '__main__':
    app.run(debug=False)