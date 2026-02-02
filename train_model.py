import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import joblib

# Step 1: Capture face images from webcam
def capture_images(user_id, username, save_path="static/faces", samples=100):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    user_folder = f"{user_id}_{username}"
    user_path = os.path.join(save_path, user_folder)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print(f"[INFO] Capturing images for '{username}' (ID: {user_id})")

    while count < samples:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle on original frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (100, 100))
            cv2.imwrite(f"{user_path}/{count}.jpg", face)
            count += 1

            # Show count on frame
            cv2.putText(frame, f"Captured: {count}/{samples}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow('Capturing Images...', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Saved {count} face images in '{user_path}'")

# Step 2: Load dataset
def load_data(data_dir='static/faces', img_size=100):
    X, y, label_dict = [], [], {}
    label_id = 0
    for user_folder in os.listdir(data_dir):
        user_path = os.path.join(data_dir, user_folder)
        if os.path.isdir(user_path):
            label_dict[user_folder] = label_id
            for img_name in os.listdir(user_path):
                img_path = os.path.join(user_path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, (img_size, img_size))
                    X.append(img_resized)
                    y.append(label_id)
            label_id += 1
    X = np.array(X).reshape(-1, img_size, img_size, 1) / 255.0
    y = to_categorical(np.array(y))
    return X, y, label_dict

# Step 3: Train CNN
def train_cnn(X, y, label_dict, model_dir='model'):
    os.makedirs(model_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

    # Save model and labels
    model.save(os.path.join(model_dir, 'cnn_face_model.h5'))
    joblib.dump(label_dict, os.path.join(model_dir, 'labels.pkl'))

    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Confusion Matrix
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)
    conf = confusion_matrix(y_true, y_pred_classes)

    plt.subplot(1, 2, 2)
    plt.imshow(conf, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    for (i, j), val in np.ndenumerate(conf):
        plt.text(j, i, f'{val}', ha='center', va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'cnn_training_metrics.png'))
    plt.show()

    print(f"[INFO] Training complete. Model and metrics saved in '{model_dir}/'")

# MAIN
if __name__ == '__main__':
    user_id = input("Enter User ID: ").strip()
    username = input("Enter Username: ").strip()
    capture_images(user_id, username)
    X, y, label_dict = load_data()
    if len(X) == 0:
        print("[ERROR] No training data found!")
    else:
        train_cnn(X, y, label_dict)