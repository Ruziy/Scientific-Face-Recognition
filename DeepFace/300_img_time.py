import os
import cv2
from deepface import DeepFace
import time

start_time = time.time() 
folder_path = r"C:\Users\Alex\Desktop\scientific_faceREC-reincornation\Scientific-Face-Recognition\300_face_test\random_300"
reference_image_path = r"C:\Users\Alex\Desktop\scientific_faceREC-reincornation\Scientific-Face-Recognition\photos\main_baiden_front.jpg"

model = "DeepFace"
detector_backend = "yunet"
similarity_metric = "cosine"
threshold = 0.1
good_face = 0
bad_face = 0
# Загружаем эталонное изображение
ref_img = cv2.imread(reference_image_path)
if ref_img is None:
    print(f"Ошибка: не удалось загрузить эталонное изображение {reference_image_path}")
    exit()

# Проходим по всем изображениям в папке
for file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, file)

    # Пропускаем эталонное изображение
    if img_path == reference_image_path:
        continue

    print(f"Обрабатываем изображение: {file}")

    try:
        # Проверяем лицо
        result = DeepFace.verify(reference_image_path, img_path, model, detector_backend, similarity_metric)

        if result:
            print(f"✅ Лицо найдено!")#Совпадение: {result['distance']} (порог: {threshold})
            good_face+=1
        # Проверяем, содержит ли `facial_areas` координаты лиц
        if 'facial_areas' in result and 'img1' in result['facial_areas'] and 'img2' in result['facial_areas']:
            img1 = cv2.imread(reference_image_path)
            img2 = cv2.imread(img_path)

            # Извлекаем координаты лица
            face1 = result['facial_areas']['img1']
            face2 = result['facial_areas']['img2']

            if all(k in face1 for k in ('x', 'y', 'w', 'h')):
                x1, y1, w1, h1 = face1['x'], face1['y'], face1['w'], face1['h']
                cv2.rectangle(img1, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2)

            if all(k in face2 for k in ('x', 'y', 'w', 'h')):
                x2, y2, w2, h2 = face2['x'], face2['y'], face2['w'], face2['h']
                cv2.rectangle(img2, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2)

            img1 = cv2.resize(img1, (700, 450))
            img2 = cv2.resize(img2, (700, 450))
            # cv2.imshow('Image 1', img1)
            # cv2.imshow('Image 2', img2)
            # cv2.waitKey(0)

    except Exception as e:
        bad_face+=1
        print(f"⚠ Ошибка при обработке {file}: {e}")

cv2.destroyAllWindows()

print(f"✅ Результаты по распозанным лицам => {good_face} что составляет =>{round(100/(300/good_face*100),2)}%")
print(f"❌ Результаты по НЕ распозанным лицам => {bad_face} что составляет =>{round(100/(300/bad_face*100),2)}%")
end_time = time.time()  # Конец измерения
execution_time = end_time - start_time  # Вычисляем разницу

print(f"Время выполнения: {execution_time:.6f} секунд")
print("✅ Обработка завершена!")