import cv2
import numpy as np
import face_recognition
import os
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Smart Vision")
root.attributes('-fullscreen', True)  

bg_color = "#282828"       
btn_color = "#4CAF50"     
exit_color = "#FF6347"     
text_color = "#FFFFFF"     
highlight_color = "#1E1E1E"
video_frame = tk.Frame(root, bg="white", width=int(root.winfo_screenwidth() * 0.8), height=int(root.winfo_screenheight() * 0.8))
video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

control_frame = tk.Frame(root, bg=bg_color, width=int(root.winfo_screenwidth() * 0.2), height=root.winfo_screenheight())
control_frame.pack(side=tk.RIGHT, fill=tk.BOTH)

title = tk.Label(control_frame, text="SMART VISION", bg=highlight_color, fg=text_color, font=("Arial", 24, 'bold'))
title.pack(pady=30)

cam = cv2.VideoCapture(0)

persons_path = 'D:/abdo1/VS Code/Smart Vision/persons'
images = []
classNames_faces = [] 
personsList = os.listdir(persons_path)

for cl in personsList:
    person_folder_path = os.path.join(persons_path, cl)
    if os.path.isdir(person_folder_path): 
        for image_name in os.listdir(person_folder_path):
            curPerson = cv2.imread(f'{person_folder_path}/{image_name}')
            if curPerson is not None:
                images.append(curPerson)
                classNames_faces.append(cl)

def findEncodeings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodeings(images)

classfile = 'coco.names'
with open(classfile, 'rt') as f:
    classNames_objects = f.read().rstrip('\n').split('\n')

configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
wightpath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(wightpath, configpath)
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

confThreshold = 0.6  
nmsThreshold = 0.4  

exit_button = Button(control_frame, text="Exit", bg=exit_color, fg=text_color, font=("Arial", 16), command=root.destroy, relief="flat")
exit_button.pack(pady=40, ipadx=10, ipady=10, fill="x")

def update_frame():
    success, img = cam.read()
    if success:
        img = cv2.flip(img, 1)
        
        classIds, confs, bbox = net.detect(img, confThreshold=confThreshold)
        if len(classIds) != 0:
            boxes = []
            confidences = []
            classIds_list = []
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                boxes.append(box)
                confidences.append(float(confidence))
                classIds_list.append(classId)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
            if len(indices) > 0:
                for i in indices.flatten():
                    box = boxes[i]
                    classId = classIds_list[i]
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames_objects[classId - 1],
                                (box[0] + 10, box[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
        
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        faceCurentFrame = face_recognition.face_locations(imgS)
        encodeCurentFrame = face_recognition.face_encodings(imgS, faceCurentFrame)
        
        for encodeface, faceLoc in zip(encodeCurentFrame, faceCurentFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeface)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeface)
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                name = classNames_faces[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(img, name, (x1 + 6, y1 - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), 3)
                
                match_confidence = (1 - faceDis[matchIndex])  
                bar_width = int(match_confidence * (x2 - x1))
                cv2.rectangle(img, (x1, y2 + 10), (x1 + bar_width, y2 + 20), (0, 255, 0), -1)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    video_label.after(10, update_frame)

video_label = tk.Label(video_frame)
video_label.pack(fill=tk.BOTH, expand=True)
update_frame()


logo = Image.open("logo.png")  
logo = logo.resize((200,200))  
logo = ImageTk.PhotoImage(logo)

logo2 = Image.open("logo2.png") 
logo2 = logo2.resize((200, 200))  
logo2 = ImageTk.PhotoImage(logo2)

root.update()

logo_label = tk.Label(video_frame, image=logo, bg="white")
logo_label.place(x=video_frame.winfo_width() - 210, y=10)

logo2_label = tk.Label(video_frame, image=logo2, bg="white")
logo2_label.place(x=video_frame.winfo_width() - 1310, y=10)

root.mainloop()


cam.release()
cv2.destroyAllWindows()
