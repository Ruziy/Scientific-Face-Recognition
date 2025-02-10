import cv2
import face_recognition
import time
import os

# Указываем путь к папке с изображениями
folder_path =r"C:\Users\Alex\Desktop\scientific_faceREC-reincornation\Scientific-Face-Recognition\300_face_test\random_300"

# Счетчики
recognized_count = 0
not_recognized_count = 0

start_time = time.time()  # Засекаем время

for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"❌ Не удалось загрузить изображение: {filename}")
            continue
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_img)
        
        if face_locations:
            recognized_count += 1
            top, right, bottom, left = face_locations[0]
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
            print(f"✅ Лицо найдено: {filename}")
        else:
            not_recognized_count += 1
            print(f"❌ Лицо не найдено: {filename}")
        
        img = cv2.resize(img, (700, 450))
        # cv2.imshow(filename, img)
        # cv2.waitKey(500)  # Показ каждого изображения на 0.5 секунды
        # cv2.destroyAllWindows()

end_time = time.time()  # Конец измерения
execution_time = end_time - start_time  # Вычисляем разницу

print("\n====== Результаты ======")
print(f"✅ Распознано лиц: {recognized_count}")
print(f"❌ Не распознано лиц: {not_recognized_count}")
print(f"⏳ Время выполнения: {execution_time:.6f} секунд")
print("✅ Обработка завершена!")
