
import json
import cv2 as cv
import face_recognition
import time
from imutils.video import WebcamVideoStream
import requests
#from sort import *
print("..............")
Conf_threshold = 0.8
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]
image = face_recognition.load_image_file("C:/Users/kesha/OneDrive/Pictures/Camera Roll/Srivatsan.jpg")
encoding = face_recognition.face_encodings(image)[0]
image2 = face_recognition.load_image_file("C:/Users/kesha/OneDrive/Pictures/Camera Roll/Srivatsan.jpg")
encoding2 = face_recognition.face_encodings(image2)[0]
image3 = face_recognition.load_image_file("C:/Users/kesha/OneDrive/Pictures/Camera Roll/Keshav.jpg")
encoding3 = face_recognition.face_encodings(image3)[0]
known_face_encodings = [
    encoding,encoding2,encoding3
]
known_face_names = [
    "Srivatsan","Srivatsan","keshav"
]
face_locations = []
face_encodings = []
face_names = []
def get_faces(frame):
    try:
        small_frame = frame
        cv.imshow("frame3",frame)
        rgb_small_frame = small_frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        name = "Unknown"
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Person"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            if name!="Unknown":
                face_names.append(name)
            if name=="Unknown":
                print(".//.//.///.")
                serverToken = 'AAAABD-Z7oI:APA91bF4_D-3E9BbHgg8t_GWxxAhPlEEuceAP9tgh5rMCOhAtZY1kz7sY9pPPfi1YVtVpmH-2lj9I7ITMOU2H8W5SopGrIyT4XWkIaod-0GhUxok48BUZZTDFD-_WEC5ZjBEIn4d8p0Z'
                deviceToken = 'device token here'

                headers = {
                        'Content-Type': 'application/json',
                        'Authorization': 'key=' + serverToken,
                    }

                body = {
                        'notification': {'title': 'Sending push form python script',
                                            'body': 'New Message'
                                            },
                        'to':
                            "d85-XdArTxufWQSLAimkVp:APA91bHJVfR6stB6xalUJ4MAWOmPjaiL15X-gkyqbHQo6K_1ZaSmHL2cpiVpb--qFv1yRZqiBLUnN17QsBa2XcgExDG1ttWMS_JKoSxFWDRCAaNdx56U9jdkITm921aM-ZavOLHxdfZC",
                        'priority': 'high',
                        #   'data': dataPayLoad,
                        }
                response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
            print(response.status_code)
        print(face_locations,name)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 10
            right *= 10
            bottom *= 10
            left *= 10
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            try:
                cv.imshow(name,frame)
                pass
            except:
                pass
    except:
        pass
class_name = []
with open('C:/Studies and Applications/Mini Project/classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
net = cv.dnn.readNet('C:/Studies and Applications/Mini Project/yolov4-tiny.weights', 'C:/Studies and Applications/Mini Project/yolov4-tiny.cfg')
model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)


cap = cv.VideoCapture("http://192.168.245.94:81/stream")
#cap = cv.VideoCapture(0)
print("video started")
starting_time = time.time()
frame_counter = 0
while True:
    _,frame = cap.read()
    frame_counter += 1
    
    classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_name[classid], score)
        cv.rectangle(frame, box, color, 1)
        
        if (class_name[classid]=="person"):
            get_faces(frame=frame[box[1]:box[3]+box[1],box[0]:box[2]+box[0]])
     
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()