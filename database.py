import os
import csv

def init_db():
    if not os.path.exists('Attendance'):
        os.makedirs('Attendance')

def add_attendance(user, date, time):
    filename = f'Attendance/Attendance-{date}.csv'
    is_new = not os.path.exists(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['Username', 'Date', 'Time'])
        writer.writerow([user, date, time])

def verify_user_login(username, password):
    if not os.path.exists("Database/users.txt"):
        return False
    with open("Database/users.txt") as f:
        for line in f:
            user, pwd, _ = line.strip().split(',')
            if username == user and password == pwd:
                return True
    return False

def get_user_data():
    users = []
    if os.path.exists("Database/users.txt"):
        with open("Database/users.txt") as f:
            for line in f:
                users.append(line.strip().split(','))
    return users
