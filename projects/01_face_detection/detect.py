# detect.py
import cv2

# 加载预训练模型
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 读取图片
img = cv2.imread('test.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 人脸检测
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# 绘制矩形框
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# 保存结果
cv2.imwrite('result.jpg', img)
