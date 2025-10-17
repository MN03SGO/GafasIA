import cv2
import time
import signal
import torch
import sys
from ultralytics.nn.tasks import DetectionModel
from src.deteccion.detector_objetos import DetectorObjetos
from src.ocr.lector_texto import LectorTexto
from src.audio.sintetizador_voz import SintetizadorVoz
from ultralytics import YOLO

class GafasIACompleto:
    def __init__(self):
        self.detector = DetectorObjetos(
            modelo_path='models/deteccion/yolov8n.pt',
            confianza_minima=0.5
        )
        self.lector_ocr = LectorTexto(
            idioma='es',
            confianza_minima=40,
            usar_gpu=torch.cuda.is_available() 
        )
        self.sintetizador = SintetizadorVoz(
            idioma='es',
            velocidad=180,
            volumen=0.8
        )
        self.modo_actual = 'objetos'  # 'objetos', 'texto', 'ambos'
        self.intervalo_deteccion = 3
        self.ultimo_analisis = 0
        self.camara = None
        self.ejecutando = False
        signal.signal(signal.SIGINT, self._manejador_cierre)
        signal.signal(signal.SIGTERM, self._manejador_cierre)
    
    def iniciar_camara(self):
        print("Abriendo cámara")
        
        for indice in [ 0, 1, 2]:
            try:
                self.camara = cv2.VideoCapture(indice)
                if self.camara.isOpened():
                    print(f"Cámara {indice} inicializada")
                    self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camara.set(cv2.CAP_PROP_FPS, 15)
                    self.camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return True
            except Exception as e:
                print(f"Error al abrir la camara en el indice {indice}: {e}")
                if self.camara: self.camara.release()
        print("Error crítico: No se pudo inicializar ninguna cámara.")
        return False
    
    def ejecutar(self, modo_visual: bool = False):
        if not self.iniciar_camara():
            self.sintetizador.decir("Error, no se pudo iniciar la cámara.")
            return
        self.sintetizador.decir_inicio()
        time.sleep(1)
        self.ejecutando = True
        self._mostrar_controles()
        try:
            while self.ejecutando:
                ret, frame = self.camara.read()
                if not ret:
                    print("Error al capturar imagen. Intentando reconectar...")
                    time.sleep(0.5)
                    continue
                
                tiempo_actual = time.time()
                frame_a_mostrar = frame.copy() if modo_visual else None
                if tiempo_actual - self.ultimo_analisis >= self.intervalo_deteccion:
                    if not self.sintetizador.esta_hablando():
                        self._analisis_periodico(frame, frame_a_mostrar, modo_visual)
                        self.ultimo_analisis = tiempo_actual
                if modo_visual:
                    texto_modo = f"Modo: {self.modo_actual.upper()}"
                    cv2.putText(frame_a_mostrar, texto_modo, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow('RASVISION', frame_a_mostrar)
                    
                    self._manejar_teclado(frame)
                    
        except Exception as e:
            print(f"Error inesperado en el bucle principal: {e}")
            self.sintetizador.decir_error()
        finally:
            self._limpiar_recursos()

    def _analisis_periodico(self, frame, frame_a_mostrar, modo_visual):
        if self.modo_actual in ['objetos', 'ambos']:
            detecciones = self._analizar_objetos(frame)
            if modo_visual and detecciones:
                self.detector.dibujar_detecciones(frame_a_mostrar, detecciones)
        if self.modo_actual in ['texto', 'ambos']:
            textos = self._analizar_texto(frame)
            if modo_visual and textos:
                self.lector_ocr.dibujar_texto_detectado(frame_a_mostrar, textos)

    def _manejar_teclado(self, frame):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): self.ejecutando = False
        elif key == ord('o'): self._cambiar_modo('objetos')
        elif key == ord('t'): self._cambiar_modo('texto')
        elif key == ord('b'): self._cambiar_modo('ambos')
        elif key == ord('a'): self._analisis_forzado(frame)
        elif key == ord('v'): self._ajustar_volumen()
        elif key == ord('r'): self._ajustar_velocidad()
    
    def _mostrar_controles(self):
        print("\n--- Controles de RasVision ---")
        print("'o' -> Modo Detección de Objetos")
        print("'t' -> Modo Lectura de Texto (OCR)")
        print("'b' -> Modo Ambos (Objetos + Texto)")
        print("'a' -> Forzar Análisis Inmediato")
        print("'v' -> Ajustar Volumen")
        print("'r' -> Ajustar Velocidad de Voz")
        print("'q' -> Salir")
        print("--------------------------------")

    def _cambiar_modo(self, nuevo_modo: str):
        if self.modo_actual == nuevo_modo: return
        self.modo_actual = nuevo_modo
        print(f"\nModo cambiado a: {nuevo_modo.upper()}")
        anuncios = {'objetos': "Detección de objetos.", 'texto': "Lectura de texto.", 'ambos': "Detección completa."}
        self.sintetizador.decir(anuncios[nuevo_modo], prioridad=True)
    
    def _analizar_objetos(self, frame):
        print("Analizando objetos...")
        detecciones = self.detector.detectar(frame, solo_prioritarios=True)
        if detecciones: self.sintetizador.decir_detecciones(detecciones)
        return detecciones

    def _analizar_texto(self, frame):
        print("Analizando texto...")
        textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
        if textos:
            descripcion = self.lector_ocr.generar_descripcion_audio(textos, modo='resumen')
            self.sintetizador.decir(descripcion)
        return textos
            
    def _analisis_forzado(self, frame):
        print("Análisis forzado iniciado...")
        self.sintetizador.decir("Analizando escena en detalle.", prioridad=True)
        
        detecciones = self.detector.detectar(frame, solo_prioritarios=False)
        textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
        
        if detecciones: self.sintetizador.decir_detecciones(detecciones, incluir_detalles=True)
        if textos:
            descripcion_detallada = self.lector_ocr.generar_descripcion_audio(textos, modo='completo')
            self.sintetizador.decir(descripcion_detallada)
        
        if not detecciones and not textos:
            self.sintetizador.decir("No se encontraron objetos ni texto relevante.")

    def _ajustar_volumen(self):
        nuevo_volumen = float(input("Introduce el nuevo volumen (0.0 a 1.0): "))
        self.sintetizador.configurar(volumen=nuevo_volumen)

    def _ajustar_velocidad(self):
        nueva_velocidad = int(input("Introduce la nueva velocidad (ej. 150 lenta, 180 normal, 220 rápida): "))
        self.sintetizador.configurar(velocidad=nueva_velocidad)

    def _manejador_cierre(self, signal_num, frame):
        print(f"\nSeñal {signal_num} recibida. Cerrando de forma ordenada...")
        self.ejecutando = False
    
    def _limpiar_recursos(self):
        print("Limpiando recursos...")
        if self.camara:
            self.camara.release()
            print("Cámara liberada.")
        cv2.destroyAllWindows()
        if self.sintetizador:
            self.sintetizador.decir("Sistema desactivado.", prioridad=True)
            self.sintetizador.esperar_finalizacion()
            self.sintetizador.finalizar()
        print("Recursos liberados. ¡Adiós!")
def mostrar_menu():
    print("\n--- Bienvenido a RasVision ---")
    print("Iniciar sistema completo (con ventana de video)")
    print("Iniciar sistema completo (solo audio, sin ventana)")
    print("Salir")
    return input("Selecciona una opción: ").strip()

def main():
    gafas = None
    try:
        while True:
            opcion = mostrar_menu()
            if opcion == '1':
                gafas = GafasIACompleto()
                gafas.ejecutar(modo_visual=True)
            elif opcion == '2':
                gafas = GafasIACompleto()
                gafas.ejecutar(modo_visual=False)
            elif opcion == '3':
                break
            else:
                print("Opción no válida. Inténtalo de nuevo.")
            gafas = None 
    except Exception as e:
        print(f"Error crítico en main: {e}")
    finally:
        if gafas and gafas.ejecutando:
            gafas._limpiar_recursos()
        print("\nGracias por usar RasVision.")
if __name__ == "__main__":
    main()
