import cv2
import sys
import threading
from deepface import DeepFace

cascPath = sys.argv[1]
verifyPath = sys.argv[2]
# faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
counter = 0

spacer = 30

def identify(image):
    obj = DeepFace.verify(verifyPath, image, model_name='ArcFace', detector_backend="fastmtcnn")
    if obj["verified"] == True:
        print(obj)
        sys.stdout.write("Match\n")
    else:
        sys.stdout.write("False\n")
    sys.stdout.flush()

threadHolder: list[threading.Thread] = []

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    '''
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    '''

    # Display the resulting frame

    cv2.imshow('Video', frame)
    if spacer == 120:
        spacer = 0
        img = frame.copy(); 
        t = threading.Thread(target=identify, kwargs={"image":img})
        t.start()
        threadHolder.append(t)
    else:
        spacer += 1
        delIdx = []
        for i, t in enumerate(threadHolder):
            t.join(timeout=0.001)
            if not t.is_alive():
                delIdx.append(i)
        for i in delIdx:
            threadHolder.pop(i)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

