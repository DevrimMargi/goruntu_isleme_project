import cv2
import pytesseract
import numpy as np
import re

# OCR ayarları (Aynı kalır)
custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=1234'

# --- DÖNGÜ ÖNCESİ TANIMLANMASI GEREKEN DEĞİŞKENLER (Aynı kalır) ---
last_printed_text = None 
print_counter = 0        
none_detection_frames = 0 
RESET_DELAY = 30 
# -------------------------------------------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Hata: Kamera başlatılamadı.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    
    if not ret or frame is None or frame.size == 0:
        continue

    # Kırmızı renk maskeleme ve kontur algılama (Önceki kodunuzla aynı)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 150, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 150, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    # KONTUR TEMİZLİĞİ: 1 metre/Yakın mesafe için optimize edildi
    kernel = np.ones((7,7), np.uint8) # Kernel boyutu 9'dan 7'ye düşürüldü
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.medianBlur(mask, 7) # Median blur boyutu 7

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_text = None 

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        # Alan eşiği 1 metreye göre ayarlandı
        if area < 10000 or area > 300000: # 1 metreden büyük görüneceği için alt eşik artırıldı.
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        if not (0.8 < aspect_ratio < 1.2 and 6 <= len(approx) <= 10):
            continue
            
        # Görüntü İşleme (Kare çizme, ROI, Thresholding)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)
        margin = int(w * 0.20)
        x1 = max(x + margin, 0)
        y1 = max(y + margin, 0)
        x2 = min(x + w - margin, frame.shape[1])
        y2 = min(y + h - margin, frame.shape[0])
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0: continue
        
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) 
        
        # *** EN KRİTİK DEĞİŞİKLİK: Adaptive Thresholding ***
        # Bu, ışık değişimlerini ve gölgeleri otomatik telafi ederek OCR başarısını artırır.
        _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # VEYA:
        # thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                                cv2.THRESH_BINARY_INV, 11, 4)

        # Tesseract OCR
        text = pytesseract.image_to_string(thresh, config=custom_config)
        text = re.sub(r'[^0-9]', '', text).strip() 

        if text in ["1","2","3","4"]:
            detected_text = text
            # Ekrana YEŞİL kare çizme
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            cv2.putText(frame, f"{detected_text} nolu tabela", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
            break 
            
    # --- KONSOL ÇIKTISINI KONTROL EDEN BLOK (Aynı kalır) ---
    
    if detected_text is not None:
        none_detection_frames = 0
        
        if detected_text != last_printed_text:
            last_printed_text = detected_text
            print_counter = 0 

        if print_counter < 2:
            print(f"{detected_text} nolu tabela algılandı (Yazdırma {print_counter + 1}/2)")
            print_counter += 1
            
    else:
        none_detection_frames += 1
        
        if none_detection_frames >= RESET_DELAY:
            last_printed_text = None
            print_counter = 0        

    cv2.imshow("Canli Tabela Okuma", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()