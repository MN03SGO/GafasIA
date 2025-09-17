import cv2
import numpy as np
from src.deteccion.detector_objetos import DetectorObjetos
import time

def main():

    print(" Iniciando prueba del detector de objetos...")
    
    try:
    
        detector = DetectorObjetos(
            modelo_path='yolov8n.pt',
            confianza_minima=0.5
        )
        
        print("Conectando con la cámara...")
        cap = cv2.VideoCapture(0)  # Cámara por defecto
        
        if not cap.isOpened():
            print("No se pudo abrir la cámara")
            return
        
        # Configurar cámara para Raspberry Pi
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        
        print("Cámara inicializada. Presiona 'q' para salir, 's' para descripción de audio")
        
        ultimo_tiempo = time.time()
        intervalo_deteccion = 0  # Detectar cada segundo, intenta probar si esta bien en 0 o aumentale de  0 > 10 max. Por el momento me gusta el 0 > 2
        
        while True:
            # Capturar frame
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar imagen")
                break
            
            tiempo_actual = time.time()
            
            # Realizar detección cada cierto intervalo
            if tiempo_actual - ultimo_tiempo >= intervalo_deteccion:
                print("Analizando imagen...")
                
                # Detectar objetos
                detecciones = detector.detectar(frame, solo_prioritarios=True)
                
                # resultados en consola
                print(f"Detecciones encontradas: {len(detecciones)}")
                for det in detecciones:
                    print(f"  - {det['nombre']} ({det['confianza']:.2f}) {det['posicion']}, {det['distancia_relativa']}")
                
                #  descripción para audio
                descripcion = detector.generar_descripcion_audio(detecciones)
                print(f" Descripción de audio: '{descripcion}'")
                
                # Dibujar detecciones en el frame
                frame_con_detecciones = detector.dibujar_detecciones(frame, detecciones)
                
                ultimo_tiempo = tiempo_actual
            else:
                frame_con_detecciones = frame
            
            # Mostrar imagen (útil para desarrollo, quitar en producción)
            cv2.imshow('Gafas IA - Detector de Objetos', frame_con_detecciones)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Saliendo...")
                break
            elif key == ord('s'):
                # Forzar detección 
                print("Generando descripción de audio...")
                detecciones = detector.detectar(frame, solo_prioritarios=True)
                descripcion = detector.generar_descripcion_audio(detecciones)
                print(f" '{descripcion}'")
                # sistema de síntesis de voz
    
    except Exception as e:
        print(f" Error en la aplicación: {e}")
    
    finally:

        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

        # OBVIAR ESTA PARTE POR EL MOMENTO "16 / 10 / 2025"

def probar_con_imagen_estatica(ruta_imagen: str):
    print(f" Probando con imagen: {ruta_imagen}")
    
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        print(" No se pudo cargar la imagen")
        return
    
    detector = DetectorObjetos()
    
    # Detectar objetos
    detecciones = detector.detectar(imagen)
    
    print(f"Objetos detectados: {len(detecciones)}")
    for det in detecciones:
        print(f"  - {det['nombre']} ({det['confianza']:.2f}) {det['posicion']}")
    
    # Descripción de audio
    descripcion = detector.generar_descripcion_audio(detecciones)
    print(f" Descripción: '{descripcion}'")
    
    # Mostrar imagen con detecciones
    imagen_resultado = detector.dibujar_detecciones(imagen, detecciones)
    cv2.imshow('Detecciones', imagen_resultado)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    modo = input("Selecciona modo de prueba:\n1. Cámara en tiempo real\n2. Imagen estática\nOpción (1/2): ")
    
    if modo == "2":
        ruta = input("Ingresa la ruta de la imagen: ")
        probar_con_imagen_estatica(ruta)
    else:
        main()