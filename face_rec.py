import cv2

haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

capture = cv2.VideoCapture(0)
screenshot_counter = 0

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

while True:
    video_frame, webcam = capture.read()

    if not video_frame:
        print("Something went wrong")
        break

    gray = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)
    faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(webcam, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("Webcam", webcam)
    k = cv2.waitKey(1)
    if k % 256 == 27:
        # escape pressed
        capture.release()
        cv2.destroyAllWindows()
        break
    elif k % 256 == 32:
        # space is pressed
        webcam_image = "webcam_screenshot.png"
        faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(webcam, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imwrite(webcam_image, webcam)
        print("Took a screenshot")
        capture.release()
        cv2.destroyAllWindows()
        break

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("face_trained.yml")
gray = cv2.cvtColor(webcam, cv2.COLOR_BGR2GRAY)
faces = haar_cascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    face_region = gray[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(face_region)
    print(f"Label = {label} with a confidence of {confidence}")
    cv2.putText(webcam, str(players[label]), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)
    cv2.rectangle(webcam, (x, y), (x+w, y+h), (255, 0, 255), 2)
cv2.imshow("Result!", webcam)
cv2.waitKey(0)