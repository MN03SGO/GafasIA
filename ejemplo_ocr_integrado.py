import cv2
import time
import signal
import torch
import numpy as np
from src.deteccion.analizador_escena import AnalizadorEscena
from src.ocr.lector_texto import LectorTexto
from src.audio.sintetizador_voz import SintetizadorVoz

class GafasIACompleto:
    def __init__(self):
        print("--- Inicializando Componentes de RasVision ---")
        self.analizador = AnalizadorEscena(
            modelo_det_path='yolov8n.pt',
            modelo_seg_path='yolov8n-seg.pt',
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
        self.modo_actual = 'objetos'
        self.intervalo_deteccion = 4
        self.ultimo_analisis = 0
        self.camara = None
        self.ejecutando = False
        
        signal.signal(signal.SIGINT, self._manejador_cierre)
        signal.signal(signal.SIGTERM, self._manejador_cierre)
        print("--- Sistema Listo ---")
    
    def iniciar_camara(self):
        print("Iniciando cámara...")
        for indice in [0, 1, 2]:
            try:
                self.camara = cv2.VideoCapture(indice)
                if self.camara.isOpened():
                    print(f"Cámara en índice {indice} inicializada.")
                    self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.camara.set(cv2.CAP_PROP_FPS, 15)
                    self.camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    return True
            except Exception as e:
                print(f"Error al abrir cámara en índice {indice}: {e}")
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
        
        ultimo_resultado_analisis = None
        
        try:
            while self.ejecutando:
                ret, frame = self.camara.read()
                if not ret:
                    print("Error al capturar imagen. Reintentando...")
                    time.sleep(0.5)
                    continue
                
                tiempo_actual = time.time()
                frame_a_mostrar = frame.copy()

                if tiempo_actual - self.ultimo_analisis >= self.intervalo_deteccion:
                    if not self.sintetizador.esta_hablando():
                        ultimo_resultado_analisis = self._analisis_periodico(frame)
                        self.ultimo_analisis = tiempo_actual
                
                if modo_visual:
                    if ultimo_resultado_analisis:
                        if self.modo_actual in ['objetos', 'ambos']:
                            frame_a_mostrar = self.analizador.dibujar_analisis(frame_a_mostrar, ultimo_resultado_analisis)
                        if self.modo_actual in ['texto', 'ambos'] and 'textos' in ultimo_resultado_analisis:
                            frame_a_mostrar = self.lector_ocr.dibujar_texto_detectado(frame_a_mostrar, ultimo_resultado_analisis['textos'])

                    texto_modo = f"Modo: {self.modo_actual.upper()}"
                    cv2.putText(frame_a_mostrar, texto_modo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('RASVISION', frame_a_mostrar)
                    self._manejar_teclado(frame)
                else:
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"Error inesperado en el bucle principal: {e}")
            self.sintetizador.decir_error()
        finally:
            self._limpiar_recursos()

    def _analisis_periodico(self, frame: np.ndarray) -> dict:
        """
        Realiza el análisis según el modo actual y lo anuncia por voz.
        Devuelve un diccionario con los resultados para poder dibujarlos.
        """
        resultados = {}
        if self.modo_actual in ['objetos', 'ambos']:
            print("Analizando escena (objetos y contexto)...")
            analisis_escena = self.analizador.analizar(frame, solo_prioritarios=True)
            if analisis_escena.get('descripcion'):
                self.sintetizador.decir(analisis_escena['descripcion'])
            resultados.update(analisis_escena)

        if self.modo_actual in ['texto', 'ambos']:
            print("Analizando texto (OCR)...")
            textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
            if textos:
                descripcion_texto = self.lector_ocr.generar_descripcion_audio(textos, modo='resumen')
                self.sintetizador.decir(descripcion_texto)
            resultados['textos'] = textos
        
        return resultados

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
        print("'o' -> Modo Análisis de Escena")
        print("'t' -> Modo Lectura de Texto (OCR)")
        print("'b' -> Modo Completo (Todo)")
        print("'a' -> Forzar Análisis Detallado Inmediato")
        print("'v' -> Ajustar Volumen (en consola)")
        print("'r' -> Ajustar Velocidad de Voz (en consola)")
        print("'q' -> Salir")
        print("--------------------------------")

    def _cambiar_modo(self, nuevo_modo: str):
        if self.modo_actual == nuevo_modo: return
        self.modo_actual = nuevo_modo
        print(f"\nModo cambiado a: {nuevo_modo.upper()}")
        anuncios = {'objetos': "Análisis de escena activado.", 'texto': "Lectura de texto activada.", 'ambos': "Modo completo activado."}
        self.sintetizador.decir(anuncios[nuevo_modo], prioridad=True)
            
    def _analisis_forzado(self, frame):
        print("--- Análisis Forzado Detallado ---")
        self.sintetizador.decir("Analizando escena en detalle.", prioridad=True)
        
        analisis_escena = self.analizador.analizar(frame, solo_prioritarios=False)
        if analisis_escena.get('descripcion'):
            self.sintetizador.decir(analisis_escena['descripcion'])

        textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
        if textos:
            descripcion_detallada = self.lector_ocr.generar_descripcion_audio(textos, modo='completo')
            self.sintetizador.decir(descripcion_detallada)
        
        if not analisis_escena.get('objetos') and not textos:
            self.sintetizador.decir("No se encontraron objetos ni texto relevante.")

    def _ajustar_volumen(self):
        try:
            nuevo_volumen = float(input("Introduce el nuevo volumen (0.0 a 1.0): "))
            self.sintetizador.configurar(volumen=nuevo_volumen)
        except ValueError:
            print("Entrada no válida.")

    def _ajustar_velocidad(self):
        try:
            nueva_velocidad = int(input("Introduce la nueva velocidad (ej. 150, 180, 220): "))
            self.sintetizador.configurar(velocidad=nueva_velocidad)
        except ValueError:
            print("Entrada no válida.")

    def _manejador_cierre(self, signal_num, frame):
        print(f"\nSeñal {signal_num} recibida. Cerrando de forma ordenada...")
        self.ejecutando = False
    
    def _limpiar_recursos(self):
        print("Limpiando recursos...")
        if self.camara and self.camara.isOpened():
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
    print("1. Iniciar sistema completo (con ventana de video)")
    print("2. Iniciar sistema completo (solo audio, sin ventana)")
    print("3. Salir")
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

