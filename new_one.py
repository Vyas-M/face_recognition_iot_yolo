import cv2 as cv
import face_recognition
import time
from imutils.video import WebcamVideoStream
from tensorflow.keras.models import load_model
import tensorflow as tf
print("..............")
Conf_threshold = 0.4
NMS_threshold = 0.4
COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0),
          (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# Load a sample picture and learn how to recognize it.
image = face_recognition.load_image_file("C://Studies and Applications//Mini Project//webcam_face_recognition-master//faces//WIN_20230212_10_46_36_Pro.jpg")
encoding = face_recognition.face_encodings(image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    encoding
]
known_face_names = [
    "elon",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

def get_faces(frame):
    print(frame.shape)
    facemodel = load_model('model2.h5')
    small_frame=cv.resize(frame,(224,224),interpolation = cv.INTER_AREA)
    print("reached .........")
    print(small_frame.shape)
    small_frame=tf.expand_dims(small_frame,axis=0)
    print(facemodel.predict([small_frame]))
    print("[[ Keshav    Srivatsan    Vyas]]")

    # Resize frame of video to 1/10 size for faster face recognition processing
    # small_frame = frame

    # # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    # rgb_small_frame = small_frame[:, :, ::-1]

    # # Find all the faces and face encodings in the current frame of video
    # face_locations = face_recognition.face_locations(rgb_small_frame)
    # face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    # name = "Person"
    # face_names = []
    # for face_encoding in face_encodings:
    #     # See if the face is a match for the known face(s)
    #     matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    #     name = "Person"

    #     # If a match was found in known_face_encodings, just use the first one.
    #     if True in matches:
    #         first_match_index = matches.index(True)
    #         name = known_face_names[first_match_index]

    #     face_names.append(name)
    # print(face_locations,name)
    # # Display the results
    # for (top, right, bottom, left), name in zip(face_locations, face_names):
    #     # Scale back up face locations since the frame we detected in was scaled to 1/10 size
    #     top *= 10
    #     right *= 10
    #     bottom *= 10
    #     left *= 10

    #     # Draw a box around the face
    #     cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    #     # Draw a label with a name below the face
    #     cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
    #     font = cv.FONT_HERSHEY_DUPLEX
    #     cv.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    try:
        cv.imshow("frame2",frame)
    except:
        pass
class_name = []
with open('C://Studies and Applications//Mini Project//classes.txt', 'r') as f:
    class_name = [cname.strip() for cname in f.readlines()]
net = cv.dnn.readNet('C:\Studies and Applications\Mini Project\yolov4-tiny.weights', 'C:\Studies and Applications\Mini Project\yolov4-tiny.cfg')
# net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)

model = cv.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/511, swapRB=True)


# cap = WebcamVideoStream(0).start()
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
        get_faces(frame=frame[box[1]:box[3]+box[1],box[0]:box[2]+box[0]])
        # cv.putText(frame, label, (box[0], box[1]-10),
        #            cv.FONT_HERSHEY_COMPLEX, 0.3, color, 3)
    endingTime = time.time() - starting_time
    fps = frame_counter/endingTime
    # print(fps)
    cv.putText(frame, f'FPS: {fps}', (20, 50),
               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
    cv.imshow('frame', frame)
    key = cv.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
