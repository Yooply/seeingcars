import cv2
import sys
from deepface import DeepFace

cascPath = sys.argv[1]
verifyPath = sys.argv[2]
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
counter = 0
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame

    cv2.imshow('Video', frame)
    if len(faces) > 0:
        cv2.imwrite(f"livestream_frame.jpg", gray)
        obj = DeepFace.verify(verifyPath, "livestream_frame.jpg", model_name='ArcFace', detector_backend="retinaface")
        print(obj["verified"])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

