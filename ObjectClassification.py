from tkinter import *
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import model_from_json

main = tkinter.Tk()
main.title("Object Classification Using CNN-Based Fusion of Vision and LIDAR in Autonomous Vehicle Environment")
main.geometry("1200x1200")

global filename
global lidar_model

COLORS = np.random.uniform(0, 255, size=(21, 3))

def get_class(idx):
    classes = {2: "Cyclist", 7: "Car", 15: "Pedestrian", 6: "Truck"}
    return classes.get(idx, "Others")

def upload_dataset():
    global filename
    try:
        filename = filedialog.askdirectory(initialdir=".")
        if filename:
            text.delete('1.0', END)
            text.insert(END, f"{filename} loaded\n")
            pathlabel.config(text=f"{filename} dataset loaded")
    except Exception as e:
        text.insert(END, f"Error loading dataset: {e}\n")

def load_lidar_model():
    global lidar_model
    try:
        proto_path = os.path.join("model", "alexnet.txt")
        model_path = os.path.join("model", "alexnet.caffemodel")
        lidar_model = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        pathlabel.config(text="LIDAR CNN Model loaded")
    except Exception as e:
        text.insert(END, f"Error loading LIDAR model: {e}\n")

def process_image(image_path):
    try:
        image_np = cv2.imread(image_path)
        image_np = cv2.resize(image_np, (800, 500))
        return image_np
    except Exception as e:
        text.insert(END, f"Error processing image: {e}\n")
        return None

def classify_lidar():
    global filename, lidar_model
    try:
        file_path = filedialog.askopenfilename(initialdir="testImages")
        if not file_path:
            return

        text.delete('1.0', END)
        text.insert(END, f"{file_path} loaded\n")
        pathlabel.config(text=f"{file_path} loaded")

        image_np = process_image(file_path)
        if image_np is None:
            return

        (h, w) = image_np.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 0.007843, (300, 300), 127.5)
        lidar_model.setInput(blob)
        detections = lidar_model.forward()

        row = 50
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                label = f"{get_class(idx)}: {confidence * 100:.2f}%"
                if confidence * 100 > 70:
                    cv2.putText(image_np, label, (10, row), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2, cv2.LINE_AA)
                    row += 30
                    text.insert(END, f"Detected & Classified Object: {get_class(idx)}\n")
                if confidence * 100 > 50:
                    cv2.rectangle(image_np, (startX, startY), (endX, endY), COLORS[idx], 2)
        cv2.imshow("LIDAR Object Classification Output", image_np)
        cv2.waitKey(0)
    except Exception as e:
        text.insert(END, f"Error in classification: {e}\n")

def display_graph():
    try:
        with open(os.path.join("model", "history.pckl"), "rb") as f:
            data = pickle.load(f)
        accuracy = data['accuracy']
        loss = data['loss']

        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.plot(loss, 'ro-', color='red')
        plt.plot(accuracy, 'ro-', color='green')
        plt.legend(['Loss', 'Accuracy'], loc='upper left')
        plt.title('RGB-LIDAR Accuracy & Loss Graph')
        plt.show()
    except Exception as e:
        text.insert(END, f"Error loading graph: {e}\n")

font = ('times', 14, 'bold')
title = Label(main, text="Object Classification Using CNN-Based Fusion of Vision and LIDAR in Autonomous Vehicle Environment")
title.config(bg='DarkGoldenrod1', fg='black', font=font, height=3, width=120)
title.place(x=5, y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Kitti Dataset", command=upload_dataset)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

lidarButton = Button(main, text="Load Alexnet LIDAR CNN Model", command=load_lidar_model)
lidarButton.place(x=50, y=150)
lidarButton.config(font=font1)

clsButton = Button(main, text="Run LIDAR Object Detection & Classification", command=classify_lidar)
clsButton.place(x=50, y=200)
clsButton.config(font=font1)

graphButton = Button(main, text="LIDAR Accuracy & Loss Graph", command=display_graph)
graphButton.place(x=480, y=200)
graphButton.config(font=font1)

pathlabel = Label(main, bg='brown', fg='white', font=font1)
pathlabel.place(x=480, y=100)

text = Text(main, height=18, width=150, font=('times', 12, 'bold'))
text.place(x=10, y=250)

main.config(bg='LightSteelBlue1')
main.mainloop()
