import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

deteksi_wajah = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                      'haarcascade_frontalface_default.xml')

pipeline_deteksi_usia = pipeline("image-classification", 
                                 model="dima806/facial_age_image_detection")

pipeline_deteksi_jenis_kelamin = pipeline("image-classification", 
                                          model="rizvandwiki/gender-classification")

kamera = cv2.VideoCapture(0)

while True:
    berhasil, frame = kamera.read()
    gambar_abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = deteksi_wajah.detectMultiScale(gambar_abu, scaleFactor=1.1, 
                                           minNeighbors=5, minSize=(30, 30))

    for (x, y, lebar, tinggi) in wajah:
        gambar_wajah = frame[y:y+tinggi, x:x+lebar]
        wajah_rgb = cv2.cvtColor(gambar_wajah, cv2.COLOR_BGR2RGB)
        gambar_pil = Image.fromarray(wajah_rgb)

        prediksi_usia = pipeline_deteksi_usia(gambar_pil)
        usia = prediksi_usia[0]['label']

        prediksi_jenis_kelamin = pipeline_deteksi_jenis_kelamin(gambar_pil)
        jenis_kelamin = prediksi_jenis_kelamin[0]['label']

        cv2.rectangle(frame, (x, y), (x+lebar, y+tinggi), (255, 0, 0), 2)
        cv2.putText(frame, f"{usia}, {jenis_kelamin}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Deteksi Wajah, Usia, dan Jenis Kelamin', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
