from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2
img_1_path = "main_baiden_front.jpg"
img_2_path = "filtered\simple_for_rec_baiden-zerno.jpg"
# 

img1 = cv2.imread(img_1_path)
img2 = cv2.imread(img_2_path)
model = "ArcFace"
#БЛОК ВСЕХ НЕЙРОСЕТЕЙ
    # "VGG-Face",
    # "Facenet",
    # "Facenet512",
    # "OpenFace",
    # "DeepFace",
    # # "DeepID",
    # # "Dlib",
    # "ArcFace",
    # "SFace",
#БЛОК ПОРОГОВЫХ ЗНАЧЕНИЙ ДЛЯ СЕТОК 
    # thresholds = {
    #     "VGG-Face": {"cosine": 0.40, "euclidean": 0.60, "euclidean_l2": 0.86},
    #     "Facenet": {"cosine": 0.40, "euclidean": 10, "euclidean_l2": 0.80},
    #     "Facenet512": {"cosine": 0.30, "euclidean": 23.56, "euclidean_l2": 1.04},
    #     "ArcFace": {"cosine": 0.68, "euclidean": 4.15, "euclidean_l2": 1.13},
    #     "Dlib": {"cosine": 0.07, "euclidean": 0.6, "euclidean_l2": 0.4},
    #     "SFace": {"cosine": 0.593, "euclidean": 10.734, "euclidean_l2": 1.055},
    #     "OpenFace": {"cosine": 0.10, "euclidean": 0.55, "euclidean_l2": 0.55},
    #     "DeepFace": {"cosine": 0.23, "euclidean": 64, "euclidean_l2": 0.64},
    #     "DeepID": {"cosine": 0.015, "euclidean": 45, "euclidean_l2": 0.17},
    # }

# БЛОК ВЫВОДА ТОЧНОСТЕЙ
print("Opencv")
result = DeepFace.verify(img_1_path,img_2_path,model,"opencv","cosine")
# print("SSD")
# result = DeepFace.verify(img_1_path,img_2_path,model,"ssd","cosine")
print("MTCNN")
result = DeepFace.verify(img_1_path,img_2_path,model,"mtcnn","cosine")
print("RetinaFace")
result = DeepFace.verify(img_1_path,img_2_path,model,"retinaface","cosine")
# print("Yunet")
# result = DeepFace.verify(img_1_path,img_2_path,model,"yunet","cosine")

# accuracy = round(result['threshold']-result['distance'],5)
# print("Максимальная точность: "+str(accuracy)+" "+str(result["detector_backend"]))
# БЛОК ВЫВОДА РАМОК 
x1, y1, w1, h1 = result['facial_areas']['img1']['x'], result['facial_areas']['img1']['y'], result['facial_areas']['img1']['w'], result['facial_areas']['img1']['h']
cv2.rectangle(img1, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
img1 = cv2.resize(img1,(700,450))
x2, y2, w2, h2 = result['facial_areas']['img2']['x'], result['facial_areas']['img2']['y'], result['facial_areas']['img2']['w'], result['facial_areas']['img2']['h']
cv2.rectangle(img2, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
img2 = cv2.resize(img2,(700,450))
cv2.imshow('Image 1', img1)
cv2.imshow("Image 2",img2)
cv2.waitKey(0)
