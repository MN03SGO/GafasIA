import time
try:
    from picamera2 import Picamera2
    print("Librería 'picamera2' importada correctamente.")
except ImportError:
    print("¡FALLO! - No se pudo importar 'picamera2'.")
    print("Revisa que esté instalada DENTRO de tu entorno 'venv'.")
    exit()

import cv2
import numpy as np

print("Inicializando Picamera2...")
picam2 = None
try:
    # 1. Inicializar la cámara
    picam2 = Picamera2()

    # 2. Crear una configuración de "preview" (rápida)
    # Usamos "main" para obtener el frame como un array de NumPy
    config = picam2.create_preview_configuration(main={"format": "RGB888"})
    picam2.configure(config)

    # 3. Arrancar la cámara
    picam2.start()
    print("¡Cámara Picamera2 iniciada!")

    # Darle 2 segundos para que se estabilice
    time.sleep(2.0)

    # 4. Capturar un frame como array de NumPy
    frame = picam2.capture_array()

    if frame is not None:
        print("--- ¡ÉXITO! ---")
        print(f"Frame capturado exitosamente. Dimensiones: {frame.shape}")
        print("¡Esta es la forma correcta de leer tu cámara!")
        
        # Opcional: Guardar el frame para verificar
        # cv2.imwrite("test_picam_frame.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # print("Frame de prueba guardado como 'test_picam_frame.jpg'")
        
    else:
        print("¡FALLO! - Se inició la cámara, pero no se pudo capturar el frame.")

except Exception as e:
    print(f"¡FALLO! - Ocurrió un error inesperado al configurar la Picamera2:")
    print(e)
    print("Asegúrate de que la cámara esté bien conectada y habilitada en 'raspi-config'.")

finally:
    # 5. Siempre detener la cámara
    if picam2:
        picam2.stop()
        print("Cámara Picamera2 detenida.")

