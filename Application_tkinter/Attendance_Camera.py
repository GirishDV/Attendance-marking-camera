# --------------------tkinter imports
from tkinter import *
import tkinter
#---------------------cv2 and image pillow image processing library
import cv2
import PIL.Image, PIL.ImageTk
#----------------------pandas imports
import pandas as pd
from pandastable import Table
#----------------------Face Recognition lib----------------
import os
import math
import pickle
import os.path
import numpy as np
import face_recognition
from sklearn import neighbors
from PIL import Image, ImageDraw
from time import localtime, strftime
from face_recognition.face_recognition_cli import image_files_in_folder

  
class App:
    def __init__(self,window,video_source=0):
        self.window = window
        self.window.title("Attendance Camera")
        self.video_source = video_source

         # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

         # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

         # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Capture", width=10, command=self.snapshot, fg='blue')
        self.btn_snapshot.pack(side = LEFT)
        
        # Button that lets the user to view logS
        self.btn_logs =tkinter.Button(window, text="View Logs", width=10,command = self.logs, fg='blue')
        self.btn_logs.pack(side = LEFT)
        
        # shutdown button
        self.btn_off = tkinter.Button(window, text = ("Shutdown"), width = 10, command = self.shutdown, fg = 'blue').pack(side = RIGHT)
        

         # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()
        
    def snapshot(self):
        
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
             cv2.imwrite('/home/pi/Desktop/Attendance-marking-camera/Application_tkinter/test/frame.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
         ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

         def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.6):
            
            
            if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
                raise Exception("Invalid image path: {}".format(X_img_path))

            if knn_clf is None and model_path is None:
                raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

            # Load a trained KNN model (if one was passed in)
            if knn_clf is None:
                with open(model_path, 'rb') as f:
                    knn_clf = pickle.load(f)

            # Load image file and find face locations
            X_img = face_recognition.load_image_file(X_img_path)
            X_face_locations = face_recognition.face_locations(X_img)

            # If no faces are found in the image, return an empty result.
            if len(X_face_locations) == 0:
                return []

            # Find encodings for faces in the test iamge
            faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

            # Use the KNN model to find the best matches for the test face
            closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

            # Predict classes and remove classifications that aren't within the threshold
            return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]
         
         for image_file in os.listdir("test"):
                full_file_path = os.path.join("test", image_file)
            
                presents=[]

                messagebox.showinfo("Attendance Camera","Recognizing faces")

                # Find all people in the image using a trained classifier model
                # Note: You can pass in either a classifier file name or a classifier model instance
                predictions = predict(full_file_path, model_path="trained_knn_model.clf")

                # Print results on the console
                for name, (top, right, bottom, left) in predictions:
                    presents.append(name)
                # Display results overlaid on an image
              
                #show_prediction_labels_on_image(os.path.join("test", image_file), predictions)
                
         # Data cleaning
         presents = list(set(presents))
         OF = pd.read_csv('Attendance.csv',usecols=["Name"])
         OF=OF['Name'].to_list()
         # Generating attendance list
         att=[]
         for i in range(len(OF)):
             for j in range(len(presents)):
                 if OF[i]==presents[j]:
                     a=1
                     break
                 else:
                     a=0
                     
             if a==1:
                 att.append('1')
             else:
                 att.append('0')
        #Creating dataframe
         present_time = strftime("%d %b %Y %H:%M:%S", localtime())
         df= pd.read_csv('Attendance.csv',index_col=0)
         df[present_time]=att
         df.to_csv('Attendance.csv', mode='w' )
         messagebox.showinfo("Attendance Camera","Done")
        

    def update(self):
        
         # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)
        
    def shutdown(self):
        os.system('sudo shutdown now')
        
    def logs(self):
        
        df= pd.read_csv('Attendance.csv',index_col=0)
        
        root = tkinter.Tk()
        root.title('LOGS')

        frame = tkinter.Frame(root)
        frame.pack(fill='both', expand=True)

        pt = Table(frame, dataframe=df)
        pt.show()
 
 
class MyVideoCapture:
    
    def __init__(self, video_source=0):
         # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
 
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
 
     # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
 
 # Create a window and pass it to the Application object
App(tkinter.Tk())

