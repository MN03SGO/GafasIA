# _ejemplo_ocr_integrado.py (Versión final integrada)

import cv2
import time

# Importa las clases que has creado desde la carpeta 'src'
from src.deteccion.detector_objetos import DetectorObjetos
from src.ocr.lector_texto import LectorTexto
from src.audio.sintetizador_voz import SintetizadorVoz

def main():
    print("🚀 Iniciando Gafas IA - by Sigaran 🚀")

    # --- 1. CONFIGURACIÓN INICIAL ---
    modo_actual = "DETECCION"
    TIEMPO_ESPERA_ENTRE_ANUNCIOS = 5.0
    ultimo_anuncio_tiempo = 0

    # --- 2. INICIALIZAR MÓDULOS ---
    detector = DetectorObjetos(modelo_path='yolov8n.pt')
    lector_ocr = LectorTexto(idioma='spa')
    sintetizador = SintetizadorVoz(velocidad=185)

    # --- 3. INICIAR CÁMARA ---
    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        print("Fatal: No se pudo abrir la cámara.")
        sintetizador.decir("Error, no puedo acceder a la cámara.", prioridad=True)
        return

    sintetizador.decir("Gafas IA activadas. Modo detección.", prioridad=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar fotograma. Saliendo.")
                break

            # --- LÓGICA DE MODOS ---
            if modo_actual == "DETECCION":
                detecciones = detector.detectar(frame, solo_prioritarios=True)
                frame_para_mostrar = detector.dibujar_detecciones(frame, detecciones)
                cv2.putText(frame_para_mostrar, "MODO: DETECCION (Pulsa 'r' para leer)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                ahora = time.time()
                if (ahora - ultimo_anuncio_tiempo > TIEMPO_ESPERA_ENTRE_ANUNCIOS) and detecciones:
                    descripcion = detector.generar_descripcion_audio(detecciones)
                    sintetizador.decir(descripcion)
                    ultimo_anuncio_tiempo = ahora

            elif modo_actual == "LECTURA":
                frame_para_mostrar = frame.copy()
                cv2.putText(frame_para_mostrar, "MODO: LECTURA (Pulsa 'ESPACIO' para capturar)", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Gafas IA - Visor", frame_para_mostrar)

            # --- MANEJO DE TECLADO (COMANDOS) ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                if modo_actual != "LECTURA":
                    modo_actual = "LECTURA"
                    sintetizador.decir("Modo lectura. Apunte al texto y presione espacio.", prioridad=True)
            elif key == ord('d'):
                if modo_actual != "DETECCION":
                    modo_actual = "DETECCION"
                    sintetizador.decir("Modo detección.", prioridad=True)
            elif key == ord(' ') and modo_actual == "LECTURA":
                sintetizador.decir("Analizando imagen...", prioridad=True)
                textos_detectados = lector_ocr.detectar_texto(frame)
                
                if textos_detectados:
                    resumen_audio = lector_ocr.generar_descripcion_audio(textos_detectados, modo='resumen')
                    sintetizador.decir(resumen_audio)
                    frame_ocr_resultado = lector_ocr.dibujar_texto_detectado(frame, textos_detectados)
                    cv2.imshow("Resultado OCR", frame_ocr_resultado)
                else:
                    sintetizador.decir("No encontré texto legible.")
    finally:
        # --- 4. LIMPIEZA FINAL ---
        print("Cerrando aplicación...")
        sintetizador.decir("Apagando.", prioridad=True)
        time.sleep(1)
        
        cap.release()
        cv2.destroyAllWindows()
        sintetizador.finalizar()
        print("Recursos liberados. ¡Hasta pronto!")

if __name__ == '__main__':
    main()