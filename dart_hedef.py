import cv2
import numpy as np

# Kamera başlatma
cap = cv2.VideoCapture(3) 

if not cap.isOpened():
    print("Hata: Kamera açılamadı.")
    exit()

print("Kamera başlatıldı. Çıkmak için 'q' tuşuna basın.")

while True:
    ret, frame = cap.read() 
    if not ret:
        print("Kare okunamadı, çıkılıyor...")
        break

    # Görüntü ön işleme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gürültü azaltma için kernel boyutu artırıldı
    blur = cv2.GaussianBlur(gray, (11, 11), 3) 

    # 1. Dart Tahtasını Algılama
    # Yarıçap aralığı dart tahtasının kameradaki boyutuna göre daraltıldı.
    # Bu değerler, tahtanızın kameradaki boyutuna göre AYARLANMALIDIR!
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1,
                               minDist=50, 
                               # Canny eşiği (param1) yanlış kenar algılamasını azaltmak için artırıldı
                               param1=120, param2=50,
                               # Dart tahtasının beklenen piksel yarıçap aralığı
                               minRadius=10, maxRadius=300) 

    dartboard_center = None
    dartboard_radius = None
    dart_detected = False
    
    # Dart tahtası algılandı mı?
    if circles is not None:
        circles = np.uint16(np.around(circles))
        dartboard_circle = circles[0][0]
        dartboard_center_x, dartboard_center_y, dartboard_radius = dartboard_circle
        dartboard_center = (dartboard_center_x, dartboard_center_y)
        
        # Tahtanın merkezini her zaman çiz (referans noktası)
        cv2.circle(frame, dartboard_center,5, (0, 255, 0), -1) 

        # 2. Dartı Algılama (Örnek: Sarı renkli dart ucu varsayımı)
        # BU HSV RENK ARALIKLARI KULLANILAN DARTIN RENGİNE GÖRE HALA KESİN AYARLANMALIDIR!
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_color = np.array([20, 100, 100]) # Sarı için alt eşik
        upper_color = np.array([30, 255, 255]) # Sarı için üst eşik
        mask = cv2.inRange(hsv, lower_color, upper_color)
        
        # Morfolojik işlemler
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if 50 < cv2.contourArea(cnt) < 5000:
                
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    dart_center_x = int(M["m10"] / M["m00"])
                    dart_center_y = int(M["m01"] / M["m00"])
                    
                    dart_detected = True
                    
                    # Dart algılandığında tahta çemberini çiz (İstenilen Kural)
                    cv2.circle(frame, dartboard_center, dartboard_radius, (255, 0, 0), 3)

                    # Dart ucunu (Hedef Noktası) işaretle
                    cv2.circle(frame, (dart_center_x, dart_center_y), 5, (0, 0, 255), -1) 
                    
                    # Konum Hesaplama ve Skor Yorumu
                    distance = np.sqrt((dart_center_x - dartboard_center[0])**2 + (dart_center_y - dartboard_center[1])**2)
                    
                    if distance < dartboard_radius * 0.1:
                        score_text = "Bullseye!"
                        color = (0, 255, 0)
                    elif distance < dartboard_radius:
                        score_text = f"Tahtada! Mesafe: {distance:.0f}"
                        color = (255, 255, 0)
                    else:
                        score_text = "Tahta Disinda!"
                        color = (0, 0, 255)
                        
                    cv2.putText(frame, score_text, (dart_center_x + 10, dart_center_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.line(frame, dartboard_center, (dart_center_x, dart_center_y), (255, 255, 255), 1)

                break 

        # Kural: Dart Tahtası algılandıysa, ancak Dart algılanmadıysa
        if not dart_detected:
            cv2.putText(frame, "Dart yok.", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
    # Tahta algılanamazsa
    else:
        cv2.putText(frame, "Dart algilanamadi.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Sonuçları göster
    cv2.imshow('Canli Dart Tespiti', frame)

    # 'q' tuşuna basıldığında çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
