import cv2

camera = cv2.VideoCapture(0)
cv2.namedWindow("cam_bedroom")

 # try to get the first frame
if camera.isOpened():
    ret, frame = camera.read()
else:
    print("Dead Camera :( )")
    ret = False

while ret:
    cv2.imshow("cam_bedroom", frame)
    ret, frame = camera.read()
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyWindow("cam_bedroom")