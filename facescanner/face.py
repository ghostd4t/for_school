import cv2
import time
import os
import datetime
from threading import Thread
from queue import Queue
import sys
import face_recognition
import numpy
import os.path
import glob


def thread_stream():
    def face_detection():
        count1 = 0
        count2 = 0
        image_queue = Queue()
        dirname, filename = os.path.split(os.path.abspath(sys.argv[0]))
        cascPath = os.path.join(dirname, "haarcascade_frontalface_default.xml")
        faceCascade = cv2.CascadeClassifier(cascPath)

        video_capture = cv2.VideoCapture(1)
        # could be changed to 1 if usb is not integrated, or if video isn't streaming use check if camera port is 3.0
        text = 'Face detected!'
        font = cv2.FONT_HERSHEY_SIMPLEX
        while True:

            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=40, minSize=(50, 50))
            try:
                for (x, y, w, h) in faces:
                    cv2.putText(frame, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0, 0), 2)
                    rectangle = True
            except Exception:
                continue
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            elif cv2.waitKey(1) & 0xFF == ord('s'):
                if ret:
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0, 0), 2)
                        rectangle = True
                    crop_frame = frame[y:y + h, x:x + w]
                    filename = f'./known_faces/{count1}.png'
                    cv2.imwrite(filename, crop_frame)
                    image_queue.put_nowait(filename)
                    print('Face_saved')
                    count1 += 1


            elif cv2.waitKey(1) & 0xFF == ord('d'):
                detect = False
                while not detect:
                    if ret:
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0, 0), 2)
                            rectangle = True

                        crop_frame = frame[y:y + h, x:x + w]
                        filename = f'./images/example.png'
                        cv2.imwrite(filename, crop_frame)
                        image_queue.put_nowait(filename)
                        detect = True
                picture = face_recognition.load_image_file(f'./images/example.png')
                picture = face_recognition.face_encodings(picture)
                if len(picture) > 0:
                   picture = picture[0]
                else:
                    print("No faces found in the image!")
                # make if here
                if len(glob.glob('./known_faces/*')) > 0:
                    know = False
                    for number in range(len(glob.glob('./known_faces/*'))):
                        known_faces = face_recognition.load_image_file(f'./known_faces/{number}.png')
                        known_faces = face_recognition.face_encodings(known_faces)[0]
                        res = face_recognition.compare_faces([known_faces], picture)
                        if res != [0]:
                            know = True
                    if know == True:
                        print('Face recognized')
                    else:
                        print('Unknown face')
                else:
                    print('database empty')


        video_capture.release()
        cv2.destroyAllWindows()
    Thread(target=face_detection()).start()

thread_stream()
