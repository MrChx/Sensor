import cv2
import numpy as np
from transformers import pipeline
from PIL import Image

deteksi_wajah = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                      'haarcascade_frontalface_default.xml')

pipeline_deteksi_ekspresi = pipeline("image-classification", 
                                     model="trpakov/vit-face-expression")

# Warna untuk setiap ekspresi
warna_ekspresi = {
    "happy": (0, 255, 0),        # Hijau
    "sad": (255, 0, 0),          # Biru
    "angry": (0, 0, 255),        # Merah
    "surprise": (255, 255, 0),   # Kuning
    "disgust": (255, 100, 0),    # Oranye
    "fear": (255, 0, 0),         # Biru (duplikasi, sesuaikan jika diperlukan)
    "neutral": (255, 255, 255)   # Putih
}

# Membuka kamera
kamera = cv2.VideoCapture(0)

while True:
    berhasil, frame = kamera.read()

    # Mengubah gambar ke skala abu-abu untuk deteksi wajah
    gambar_abu = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Mendeteksi wajah
    wajah = deteksi_wajah.detectMultiScale(gambar_abu, scaleFactor=1.1, 
                                           minNeighbors=5, minSize=(50, 50))

    for (x, y, lebar, tinggi) in wajah:
        # Memotong bagian wajah dari gambar
        gambar_wajah = frame[y:y+tinggi, x:x+lebar]

        # Mengubah gambar wajah ke format RGB
        wajah_rgb = cv2.cvtColor(gambar_wajah, cv2.COLOR_BGR2RGB)

        # Mengubah gambar wajah menjadi format PIL
        gambar_pil = Image.fromarray(wajah_rgb)

        # Mendeteksi ekspresi wajah
        prediksi_ekspresi = pipeline_deteksi_ekspresi(gambar_pil)
        ekspresi = prediksi_ekspresi[0]['label']

        # Menggambar persegi di sekitar wajah dan menampilkan label ekspresi
        cv2.rectangle(frame, (x, y), (x+lebar, y+tinggi), warna_ekspresi[ekspresi], 2)
        cv2.putText(frame, f"{ekspresi}", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    warna_ekspresi[ekspresi], 2)

    # Menampilkan hasil deteksi pada jendela
    cv2.imshow('Deteksi Ekspresi Wajah', frame)

    # Keluar dari program jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera dan jendela
kamera.release()
cv2.destroyAllWindows()
