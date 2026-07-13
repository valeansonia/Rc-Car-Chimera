import cv2

print("Începem căutarea camerelor conectate...")

# Căutăm pe canalele de la 0 la 5
for i in range(6):
    # Folosim explicit driverul nativ de Linux (CAP_V4L2)
    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"SUCCES! Am găsit o cameră funcțională pe canalul: {i}")
            # Arătăm imaginea pe ecran timp de 3 secunde
            cv2.imshow(f"Test Camera {i}", frame)
            cv2.waitKey(3000)
            cv2.destroyAllWindows()
        else:
            print(f"Canalul {i} s-a deschis, dar nu trimite imagine.")
    else:
        print(f"Nimic pe canalul {i}.")
        
    cap.release()

print("Căutare finalizată!")