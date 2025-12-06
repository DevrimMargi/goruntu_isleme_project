import cv2
import time
import sys
import numpy as np

# --- Parkur Yapılandırması ---
TASK_ORDER = list(range(1, 12)) 
WARNING_DISTANCE_M = 5.0        
ALIGNMENT_DISTANCE_M = 0.5      
CAMERA_INDEX = 1                # Kamera indeksi 1 olarak sabit

# --- Simülasyon ve Durum ---
current_task_index = 0
task_states = {}
current_simulated_distance_to_label = 7.0 
# OCR için Tesseract kütüphanesini kullanacaksanız buraya import etmelisiniz.
# import pytesseract # Gerekirse

# ... initialize_camera fonksiyonu aynı kalır ...

def initialize_camera():
    """Kamerayı başlatır."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"❌ HATA: Kamera {CAMERA_INDEX} indeksi ile açılamadı.")
        print("Lütfen indeksinizi (0, 2, 3...) kontrol edin veya kamera izinlerinizi kontrol edin.")
        sys.exit(0)
    print(f"✅ Kamera {CAMERA_INDEX} indeksi ile başarıyla başlatıldı.")
    return cap

def detect_label_and_distance(frame, target_task_id):
    """
    Görüntü İşleme Fonksiyonu: Kırmızı daire içindeki target_task_id tabelasını algılar.
    """
    global current_simulated_distance_to_label
    
    # 1. Görüntüyü HSV renk uzayına dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Kırmızı rengin HSV aralıkları (Kırmızı iki aralıkta yer alır)
    # Kırmızı maske için iki maske oluşturup birleştirmek gerekir
    lower_red_1 = np.array([0, 100, 100])
    upper_red_1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)

    lower_red_2 = np.array([170, 100, 100])
    upper_red_2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
    
    # İki maskeyi birleştir
    red_mask = mask1 + mask2

    # 2. Konturları bul (Daire tespiti)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    label_found = False
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Sadece yeterince büyük olan konturları işle
        if area > 100: 
            
            # Kontur etrafına minimum çevreleyen daireyi çiz
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Daire-kontur oranı kontrolü (Şeklin daireye ne kadar benzediğini kontrol eder)
            # Dairenin alanı: pi * r^2. Kontur alanının bu değere yakın olması gerekir.
            circle_area_theoretical = np.pi * radius**2
            
            # Eğer kontur alanı teorik daire alanının %70-%130'u arasında ise daireye benzer kabul edelim.
            if circle_area_theoretical * 0.7 < area < circle_area_theoretical * 1.3:
                
                # Tespiti onaylamak için çevreleyen daireyi çizer (Görsel doğrulama)
                cv2.circle(frame, center, radius, (0, 0, 255), 3)
                
                # 3. OCR (Rakam Tanıma) YERİ:
                # Gerçekte, burada dairenin içindeki bölgeyi (ROI) kesip 
                # Tesseract gibi bir OCR kütüphanesi kullanarak '1' rakamını tanımalısınız.
                
                # Bu aşamada sadece daireyi gördüğümüzü varsayalım:
                label_found = True
                
                # --- MESAFE TAHMİNİ (SADECE SİMÜLASYON YERİNE) ---
                # Mesafeyi tahmin etmek için temel geometri kullanılabilir: 
                # Gerçek mesafe = (Hedefin gerçek genişliği * Odağın odak uzaklığı) / Hedefin piksel genişliği
                
                # Şimdilik simülasyon mantığını koruyalım, ancak artık sadece
                # hedef (1 numara) bulunduğunda mesafe değişecek.
                current_simulated_distance_to_label -= 0.05 
                if current_simulated_distance_to_label < 0:
                    current_simulated_distance_to_label = 0.0
                    
                cv2.putText(frame, f"KIRMIZI DAİRE {target_task_id} BULUNDU: {current_simulated_distance_to_label:.1f}m", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                return label_found, current_simulated_distance_to_label

    # Tabela bulunamadıysa
    if not label_found and target_task_id == 1:
        # Tabela kaybolduğunda simülasyon mesafesini tekrar başlat (yeni görev için)
        current_simulated_distance_to_label = 7.0 
    
    return False, float('inf')


# --- Ana Algoritma Döngüsü ve Başlatma (Aynı Kalır) ---
def run_parkour_with_camera():
    global current_task_index
    global current_simulated_distance_to_label
    
    cap = initialize_camera()

    print("\n--- 🏁 Kırmızı Tabela Takip Başladı ---")
    
    while current_task_index < len(TASK_ORDER):
        
        ret, frame = cap.read()
        if not ret:
            print("❌ Kamera akışı kesildi.")
            break
            
        target_task_id = TASK_ORDER[current_task_index]
        label_detected, distance = detect_label_and_distance(frame, target_task_id)

        task_key = target_task_id
        
        # --- ALGORİTMA MANTIĞI (Aynı) ---
        if label_detected:
            
            if distance <= WARNING_DISTANCE_M and distance > ALIGNMENT_DISTANCE_M:
                if task_states.get(task_key) != "Warned":
                    task_states[task_key] = "Warned"
                    print(f"⚠️ **UZAKTAN ALGILANDI:** {task_key}. Tabela {distance:.1f} metreden görüldü.")
                        
            elif distance <= ALIGNMENT_DISTANCE_M:
                if task_states.get(task_key) != "Started":
                    task_states[task_key] = "Started"
                    print(f"✅ **HİZALANMA TAMAM:** {task_key}. Tabela ile aynı hizaya gelindi (Mesafe: {distance:.1f}m). **{task_key}. Görev başladı.**")
                    
                    current_task_index += 1
                    current_simulated_distance_to_label = 7.0 # Yeni görev için reset
                    
            time.sleep(0.01) 

        cv2.imshow('Parkur Takip', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    if current_task_index >= len(TASK_ORDER):
        print("\n--- 🏆 BİTİŞ: Tüm görevler tamamlandı! ---")

    cap.release()
    cv2.destroyAllWindows()


# --- Programı Başlat ---
run_parkour_with_camera()