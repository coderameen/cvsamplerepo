# import cv2

# img = cv2.imread('group.jpg')
# cv2.imshow("face image",img)
# print(img.shape)
# print(img.ndim)
# cv2.waitKey(0)
# # #cv2.waitKey(0)#all keyboard buttons
# # while True:
# #     if cv2.waitKey(1)==27:#ord('x')
# #         break
# cv2.destroyAllWindows()

# #n dim
# '''
# color img: 3 channels ( 3 dim)
# gray img: 2
# binary img : 2
# '''


#to convert RGB image to Gray scale image
# import cv2

# img = cv2.imread("group.jpg",0)
# cv2.imshow("gray image",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import cv2

# img = cv2.imread("group.jpg")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray image",gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#camera
#internal camera(0)
#external camera(1)

#matplotlib : used to visualise the dataset
# import cv2
# import matplotlib.pyplot as plt 
# cap = cv2.VideoCapture(0)#laptop camera access
# ret, frame = cap.read()

# plt.imshow(frame)
# plt.title("my image capture")
# plt.show()


# cap.release()


#to detectedt faces in image
import cv2
cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv2.imread("group.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#snytax: detectMultiScale(image, scaling factor, min_neighbor)
faces = cascade.detectMultiScale(gray, 1.03, 6)

for x,y,w,h in faces:

    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

cv2.imshow("Face detected!!",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

