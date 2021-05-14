import numpy
import cv2

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
DIR = r"C:\Users\kchut\PycharmProjects\OpenCV-Testing\pictures"

features = []
labels = []
haar_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")


def create_training_set():
    for current_player in players:
        path = os.path.join(DIR, current_player)
        label = players.index(current_player)

        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            image_array = cv.imread(image_path)
            gray = cv.cvtColor(image_array, cv.COLOR_BGR2GRAY)

            faces_recognition = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in faces_recognition:
                faces_region = gray[y:y + h, x:x + w]
                features.append(faces_region)
                labels.append(label)


create_training_set()
print(len(features))
print(len(labels))

print("done")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
features = numpy.array(features)
labels = numpy.array(labels)

face_recognizer.train(features, labels)
face_recognizer.save("face_trained.yml")
numpy.save("features.npy", features)
numpy.save("labels.npy", labels)

#
current_img = cv.imread(r"C:\Users\kchut\PycharmProjects\OpenCV-Testing\pictures\Spencer Dinwiddie\spencer15.png")
gray = cv.cvtColor(current_img, cv.COLOR_BGR2GRAY)
faces = haar_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    face_region = gray[y:y + h, x:x + h]
    label, confidence = face_recognizer.predict(face_region)
    print(f"Label = {label} with a confidence of {confidence}")
    cv.putText(current_img, str(players[label]), (20, 20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(current_img, (x, y), (x + w, y + h), (255, 0, 255), thickness=2)
cv.imshow("Detected Face", current_img)
cv.waitKey(0)
