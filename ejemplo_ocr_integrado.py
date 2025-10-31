import cv2
import time
import signal
import torch
import numpy as np
import sys
import traceback

try:
    from picamera2 import Picamera2
    print("Picamera2 importada exitosamente.")
except ImportError:
    print("No se pudo importar Picamera2.")
    sys.exit(1)
from src.ocr.lector_texto import LectorTexto
from src.audio.sintetizador_voz import SintetizadorVoz
from src.deteccion.analizador_escena import AnalizadorEscena

class GafasIACompleto:
    def __init__(self):
        print("Inicializando de RasVision")
        self.analizador = AnalizadorEscena(
            modelo_custom_path='models/detecciones/Modelo_V4.pt', # Cargando V4
            modelo_seg_path='yolov8n-seg.pt',
            confianza_minima=0.3
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
        self.modo_actual = 'objetos'
        self.intervalo_deteccion = 4
        self.ultimo_analisis = 0
        self.picam2 = None # cambiado de camera
        self.ejecutando = False

        self.jpeg_quality = 50 #calidad de la imagen, pare mejor transmision lo deje en 50
        signal.signal(signal.SIGINT, self._manejador_cierre)
        signal.signal(signal.SIGTERM, self._manejador_cierre)


    def iniciar_camara(self):
        print("Abriendo cámara picamera")
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"format": "RGB888", "size": (640, 480)}
            )
            self.picam2.configure(config)
            self.picam2.start()
            print("Picamera2 inicializada y configurada.")
            time.sleep(1.0) #  un segundo de esperra para  iniciar
            return True
        except Exception as e:
            print(f"Error al inicializar Picamera2: {e}")
            traceback.print_exc()
            if self.picam2:
                self.picam2.stop()
            self.picam2 = None
            return False

    def ejecutar(self, modo_visual: bool = False):
        if not self.picam2 or not self.picam2.started:
            print(" cámara no está lista. Iniciando...")
            if not self.iniciar_camara():
                self.sintetizador.decir("Error, no se pudo iniciar la cámara.")
                return

        self.sintetizador.decir_inicio()
        time.sleep(1)
        self.ejecutando = True
        self._mostrar_controles()
        ultimo_resultado_analisis = None

        try:
            while self.ejecutando:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None:
                    print("Error al capturar imagen o frame nulo. Reintentando...")
                    time.sleep(0.5)
                    continue
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                
                tiempo_actual = time.time()
                frame_a_mostrar = frame_bgr.copy() 

                if tiempo_actual - self.ultimo_analisis >= self.intervalo_deteccion:
                    if not self.sintetizador.esta_hablando():
                        print(f"\n Iniciando análisis (Modo: {self.modo_actual}) ---")
                        ultimo_resultado_analisis = self._analisis_periodico(frame_rgb)
                        self.ultimo_analisis = tiempo_actual

                if modo_visual:
                    if ultimo_resultado_analisis:
                        if self.modo_actual in ['objetos', 'ambos'] and ultimo_resultado_analisis.get('objetos'):
                            frame_a_mostrar = self.analizador.dibujar_analisis(frame_a_mostrar, ultimo_resultado_analisis)
                        if self.modo_actual in ['texto', 'ambos'] and ultimo_resultado_analisis.get('textos'):
                            try:
                                frame_a_mostrar = self.lector_ocr.dibujar_texto_detectado(frame_a_mostrar, ultimo_resultado_analisis['textos'])
                            except Exception as e_ocr_draw:
                                print(f"Error al dibujar texto OCR: {e_ocr_draw}")

                    cv2.imshow('RASVISION', frame_a_mostrar)
                    self._manejar_teclado(frame_rgb) 
                else:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nInterrupción por teclado detectada.")
        except Exception as e:
            import traceback
            print(f"Error FATAL inesperado en el bucle principal: {e}")
            traceback.print_exc()
            self.sintetizador.decir_error()
        finally:
            self._limpiar_recursos()

    def _analisis_periodico(self, frame_rgb: np.ndarray) -> dict:
        resultados_completos = {'objetos': [], 'contexto': [], 'descripcion': '', 'textos': []}
        try:
            if self.modo_actual in ['objetos', 'ambos']:
                print("Analizando escena (objetos y contexto)...")
                analisis_escena = self.analizador.analizar(frame_rgb, solo_prioritarios=True)
                resultados_completos.update(analisis_escena)
                if analisis_escena.get('objetos') or analisis_escena.get('contexto'):
                    print(f"Descripción generada: {analisis_escena['descripcion']}")
                    self.sintetizador.decir(analisis_escena['descripcion'])
                else:
                    print(" No se detectaron objetos/contexto prioritarios.")

            if self.modo_actual in ['texto', 'ambos']:
                print("Analizando texto (OCR)...")
                textos = self.lector_ocr.detectar_texto(frame_rgb, mejorar_imagen=True)
                resultados_completos['textos'] = textos
                if textos:
                    descripcion_texto = self.lector_ocr.generar_descripcion_audio(textos, modo='resumen')
                    print(f"Texto OCR detectado: {descripcion_texto}")
                    self.sintetizador.decir(descripcion_texto)
                else:
                    print("No se detectó texto OCR.")
        except Exception as e:
            print(f"Error durante el análisis periódico: {e}")
            traceback.print_exc()
            self.sintetizador.decir("Error durante el análisis.")
        return resultados_completos

    def _manejar_teclado(self, frame_original_rgb):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Cerrando por petición del usuario (tecla 'q')...")
            self.ejecutando = False
        elif key == ord('o'):
            self._cambiar_modo('objetos')
        elif key == ord('t'):
            self._cambiar_modo('texto')
        elif key == ord('b'):
            self._cambiar_modo('ambos')
        elif key == ord('a'):
            self._analisis_forzado(frame_original_rgb)

    def _mostrar_controles(self):
        print(" Controles de RasVision ")
        print("'o' Modo Detección de Objetos")
        print("'t' Modo Lectura de Texto (OCR)")
        print("'b' Modo Ambos (Objetos + Texto)")
        print("'a'Forzar Análisis Detallado Inmediato")
        print("'q' Salir")
        print(f"Modo actual: {self.modo_actual.upper()}")

    def _cambiar_modo(self, nuevo_modo: str):
        if self.modo_actual == nuevo_modo:
            print(f"Ya estás en modo {nuevo_modo.upper()}.")
            return
        self.modo_actual = nuevo_modo
        print(f"\nModo cambiado a: {nuevo_modo.upper()}")
        anuncios = {'objetos': "Modo detección de objetos.",
                    'texto': "Modo lectura de texto.",
                    'ambos': "Modo detección completa."}
        self.sintetizador.decir(anuncios.get(nuevo_modo, "Modo desconocido."), prioridad=True)

    def _analisis_forzado(self, frame_rgb: np.ndarray):
        print("\n Análisis Forzado Detallado")
        if self.sintetizador.esta_hablando():
            print("Esperando a que termine el audio actual...")
            self.sintetizador.esperar_finalizacion(timeout=5)

        self.sintetizador.decir("Analizando escena en detalle.", prioridad=True)
        try:
            analisis_escena = self.analizador.analizar(frame_rgb, solo_prioritarios=False)
            print(f"Descripción detallada generada: {analisis_escena['descripcion']}")
            self.sintetizador.decir(analisis_escena['descripcion'])

            if self.modo_actual in ['texto', 'ambos']:
                print("Analizando texto (OCR detallado)...")
                textos = self.lector_ocr.detectar_texto(frame_rgb, mejorar_imagen=True)
                if textos:
                    descripcion_detallada_ocr = self.lector_ocr.generar_descripcion_audio(textos, modo='completo')
                    print(f"Texto OCR detallado: {descripcion_detallada_ocr}")
                    self.sintetizador.decir(descripcion_detallada_ocr)
                else:
                    print("No se detectó texto OCR.")

            objetos_encontrados = analisis_escena.get('objetos')
            texto_encontrado = 'textos' in locals() and textos
            if not objetos_encontrados and (self.modo_actual != 'texto' or not texto_encontrado):
                self.sintetizador.decir("No se encontraron elementos relevantes en este análisis.")
        except Exception as e:
            print(f"Error durante el análisis forzado: {e}")
            traceback.print_exc()
            self.sintetizador.decir("Error durante el análisis detallado.")

    def _manejador_cierre(self, signal_num, frame):
        print(f"\nSeñal {signal_num} recibida. Iniciando cierre ordenado...")
        self.ejecutando = False
        
    def _limpiar_recursos(self):
        print("Limpiando recursos...")
        if hasattr(self, 'picam2') and self.picam2:
            self.picam2.stop()
            print("Cámara Picamera2 detenida.")

        cv2.destroyAllWindows()
        print("Ventanas de OpenCV cerradas.")
        if hasattr(self, 'sintetizador') and self.sintetizador:
            print("Solicitando finalización del sintetizador...")
            self.sintetizador.decir("Sistema RasVision desactivado.", prioridad=True)
            self.sintetizador.esperar_finalizacion(timeout=5)
            self.sintetizador.finalizar()
        print("Recursos liberados. ¡Adiós!")

    def generar_frames_flask(self):
        print("Iniciando generador de frames para Flask...")
        if not self.picam2 or not self.picam2.started:
            if not self.iniciar_camara():
                print("No se pudo iniciar la cámara para Flask.")
                return

        # Parámetros de codificación JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]

        self.ejecutando = True
        while self.ejecutando:
            try:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None:
                    continue
                
                resultados_completos = {}
                analisis_escena = self.analizador.analizar(frame_rgb, solo_prioritarios=True)
                resultados_completos.update(analisis_escena)
                textos = self.lector_ocr.detectar_texto(frame_rgb, mejorar_imagen=False) # 'mejorar' es lento
                resultados_completos['textos'] = textos
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frame_a_mostrar = frame_bgr
                if resultados_completos.get('objetos'):
                    frame_a_mostrar = self.analizador.dibujar_analisis(frame_a_mostrar, resultados_completos)
                if resultados_completos.get('textos'):
                    frame_a_mostrar = self.lector_ocr.dibujar_texto_detectado(frame_a_mostrar, resultados_completos['textos'])

                (flag, encodedImage) = cv2.imencode(".jpg", frame_a_mostrar, encode_param)
                if not flag:
                    continue


                yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
                    bytearray(encodedImage) + b'\r\n')
            
            except Exception as e:
                print(f"Error en bucle Flask: {e}")
                traceback.print_exc()
                time.sleep(1.0)
        
        print("Bucle de generación de frames terminado.")
        self._limpiar_recursos()


def mostrar_menu():
    print("\nBienvenido a RasVision ")
    print("1. Iniciar sistema completo (con ventana de video)")
    print("2. Iniciar sistema completo (solo audio, sin ventana)")
    print("3. Salir")
    print("---")
    return input("Selecciona una opción (1-3): ").strip()

def main():
    gafas = None
    try:
        while True:
            opcion = mostrar_menu()
            if opcion == '1':
                print("\nIniciando en modo visual...")
                gafas = GafasIACompleto()
                gafas.ejecutar(modo_visual=True)
                gafas = None 
            elif opcion == '2':
                print("\nIniciando en modo solo audio...")
                gafas = GafasIACompleto()
                gafas.ejecutar(modo_visual=False)
                gafas = None
            elif opcion == '3':
                print("Saliendo...")
                break
            else:
                print("Opción no válida. Inténtalo de nuevo.")
    except Exception as e:
        import traceback
        print(f"Error CRÍTICO en la ejecución principal: {e}")
        traceback.print_exc()
    finally:
        if gafas and gafas.ejecutando:
            print("\nRealizando limpieza final por salida inesperada...")
            gafas._limpiar_recursos()
        print("\nGracias por usar RasVision.")

if __name__ == "__main__":
    main()



