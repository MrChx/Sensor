import cv2
import mediapipe as mp
import pyautogui

kamera = cv2.VideoCapture(0)

deteksi_tangan = mp.solutions.hands.Hands()
gambar_utilitas = mp.solutions.drawing_utils

lebar_layar, tinggi_layar = pyautogui.size()

index_y = 0

while True:
    berhasil, frame = kamera.read()
    if not berhasil:
        break
    
    tinggi_frame, lebar_frame, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    hasil = deteksi_tangan.process(frame_rgb)
    tangan = hasil.multi_hand_landmarks
    
    if tangan:
        for landmark_tangan in tangan:
            gambar_utilitas.draw_landmarks(frame, landmark_tangan, mp.solutions.hands.HAND_CONNECTIONS)

            landmarks = landmark_tangan.landmark
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * lebar_frame)
                y = int(landmark.y * tinggi_frame)
                
                if id == 8:  # Jari telunjuk
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    index_x = lebar_layar / lebar_frame * x
                    index_y = tinggi_layar / tinggi_frame * y
                    pyautogui.moveTo(index_x, index_y)
                
                if id == 4:  # Ibu jari
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255), thickness=-1)
                    ibu_jari_x = lebar_layar / lebar_frame * x
                    ibu_jari_y = tinggi_layar / tinggi_frame * y
                    
                    if abs(index_y - ibu_jari_y) < 20:
                        print('Klik')
                        pyautogui.click()

    cv2.imshow('Mouse Virtual', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
