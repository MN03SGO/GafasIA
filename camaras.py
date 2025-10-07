import cv2

for i in range(5):  # Prueba hasta 5 cámaras (ajusta si tienes más)
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara {i} - Abierta correctamente")
        # Muestra una captura para identificar visualmente
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Cámara {i}", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"Cámara {i} - No disponible")
