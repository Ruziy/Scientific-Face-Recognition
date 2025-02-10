import os
import cv2
from deepface import DeepFace
import time

start_time = time.time()
folder_path = r"C:\Users\Alex\Desktop\scientific_faceREC-reincornation\Scientific-Face-Recognition\photos_categories\boys_low_12"
reference_image_path = r"C:\Users\Alex\Desktop\scientific_faceREC-reincornation\Scientific-Face-Recognition\photos\main_baiden_front.jpg"

model = "ArcFace"
detector_backend = "yunet"
threshold = 0.1  

good_face = 0
bad_face = 0

# Проходим по всем изображениям в папке
for file in os.listdir(folder_path):
    img_path = os.path.join(folder_path, file)

    # Пропускаем эталонное изображение
    if img_path == reference_image_path:
        continue

    print(f"Обрабатываем изображение: {file}")

    try:
        # Анализируем лицо
        result = DeepFace.analyze(img_path, actions=['emotion'], detector_backend=detector_backend)

        if result:
            good_face += 1
            print(f"✅ Лицо найдено с вероятностью {result[0]['face_confidence']:.2f}%")
        else:
            bad_face += 1
            print(f"❌ Лицо не найдено.")

    except Exception as e:
        bad_face += 1
        print(f"⚠ Ошибка при обработке {file}: {e}")

cv2.destroyAllWindows()

total_images = good_face + bad_face
good_percent = round((good_face / total_images) * 100, 2) if total_images else 0
bad_percent = round((bad_face / total_images) * 100, 2) if total_images else 0

print(f"✅ Распознанные лица: {good_face} ({good_percent}%)")
print(f"❌ Не распознанные лица: {bad_face} ({bad_percent}%)")
print(f"⏱ Время выполнения: {time.time() - start_time:.6f} секунд")
print("✅ Обработка завершена!")
