# detect.py
import cv2
import sys

# 加载预训练模型
try:
    face_cascade = cv2.CascadeClassifier('D:/work/opencv-practice/projects/01_face_detection/haarcascade_frontalface_default.xml')
except Exception as e:
    print("加载模型失败:", e)
    sys.exit(1)

# 读取图片
try:
    img = cv2.imread('D:/work/opencv-practice/projects/01_face_detection/test.jpg')
    if img is None:
        raise FileNotFoundError("无法加载图片，请检查test.jpg是否存在")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
except Exception as e:
    print("图片处理错误:", e)
    sys.exit(1)

# 人脸检测
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
print(f"检测到 {len(faces)} 张人脸")

# 绘制矩形框
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

# 保存结果
cv2.imwrite('result.jpg', img)

# 显示结果
cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 用Python生成对比图
import numpy as np

original = cv2.imread('D:/work/opencv-practice/projects/01_face_detection/test.jpg', cv2.IMREAD_COLOR)
result = cv2.imread('result.jpg', cv2.IMREAD_COLOR)
if original is None or result is None:
    print("无法加载图片进行对比")
else:
    if original.ndim != result.ndim:
        if original.ndim == 1:
            original = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        elif result.ndim == 1:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    compare = np.hstack([original, result])  # 水平拼接
    cv2.imwrite('compare.png', compare)
