import cv2

cam = cv2.VideoCapture(0)
alg = "haarcascade_frontalface_default.xml"
haarcascade = cv2.CascadeClassifier(alg)

while True:
    _,img = cam.read()
    text = "No Person Detected"
    GrayImg1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face = haarcascade.detectMultiScale(GrayImg1,1.4,6)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        text = "Person Detected"
    cv2.putText(img,text,(15,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    print(text)
    cv2.imshow("FaceDetection",img)
    key = cv2.waitKey(1)& 0xFF
    if key == ord("Q") or key == ord("q"):
        break
cv2.destroyAllWindows()