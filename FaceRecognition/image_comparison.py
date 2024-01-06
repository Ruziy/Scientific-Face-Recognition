import cv2
import face_recognition

img = cv2.imread("Popular/Andzhelina-Dzholi-1.jpg")

rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_encoding = face_recognition.face_encodings(rgb_img)[0]
face_locations = face_recognition.face_locations(rgb_img)
top, right, bottom, left = face_locations[0][0],face_locations[0][1],face_locations[0][2],face_locations[0][3]
cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)


img2 = cv2.imread("Popular/andzheliny-dzholi_2.jpg")
rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img_encoding2 = face_recognition.face_encodings(rgb_img2)[0]

face_locations_2 = face_recognition.face_locations(rgb_img2)
top_2, right_2, bottom_2, left_2 = face_locations_2[0][0],face_locations_2[0][1],face_locations_2[0][2],face_locations_2[0][3]
cv2.rectangle(img2, (left_2, top_2), (right_2, bottom_2), (0, 255, 0), 2)

result = face_recognition.compare_faces([img_encoding], img_encoding2)
print("Result: ", result) #list(face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)
img = cv2.resize(img,(700,450))
img2 = cv2.resize(img2,(700,450))
cv2.imshow("Img", img)
cv2.imshow("Img_2", img2)
cv2.waitKey(0)