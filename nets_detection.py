import numpy
import cv2
import os

players = [
    "Kyrie Irving",
    "Kevin Durant",
    "James Harden",
    "Blake Griffin",
    "Joe Harris",
    "Spencer Dinwiddie",
    "Jeff Green",
    "Landry Shamet",
    "Bruce Brown",
    "Tyler Johnson",
    "Nicolas Claxton",
    "DeAndre Jordan",
    "Chris Chiozza",
    "Alize Johnson",
    "Reggie Perry",
    "Mike James",
    "Timothe Luwawu-Cabarrot"
]
p = []
DIR = r"C:\Users\kchut\PycharmProjects\BrooklynNets-FaceRecognizer\pictures"

features = []
labels = []
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def create_training_set():
    for current_player in players:
        path = os.path.join(DIR, current_player)
        label = players.index(current_player)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            image_array = cv2.imread(image_path)
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            faces_recognition = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_recognition:
                faces_region = gray[y:y + h, x:x + w]
                features.append(faces_region)
                labels.append(label)


create_training_set()
print(len(features))
print(len(labels))

print("done")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
features = numpy.array(features)
labels = numpy.array(labels)

face_recognizer.train(features, labels)
face_recognizer.save("face_trained.yml")
numpy.save("features.npy", features)
numpy.save("labels.npy", labels)

# For adding new players to the detection

# DIR = r"C:\Users\kchut\PycharmProjects\BrooklynNets-FaceRecognizer\pictures"
# current_player = "Timothe Luwawu-Cabarrot"
# path = os.path.join(DIR, current_player)
#
# for current_file in os.listdir(path):
#     # current_img = cv2.imread(r"pictures\Alize Johnson\alize15.jpg")  # Single image
#     current_img = cv2.imread(fr"pictures\{current_player}\{current_file}")  # All images
#     gray = cv2.cvtColor(current_img, cv2.COLOR_BGR2GRAY)
#     faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
#     print(f"# of faces: {len(faces)}")
#     for (x, y, w, h) in faces:
#         face_region = gray[y:y + h, x:x + h]
#         label, confidence = face_recognizer.predict(face_region)
#         print(f"Label = {label} with a confidence of {confidence}")
#         cv2.putText(current_img, str(players[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
#         cv2.rectangle(current_img, (x, y), (x + w, y + h), (255, 0, 255), thickness=2)
#     cv2.imshow("Detected Face", current_img)
#     cv2.waitKey(0)
