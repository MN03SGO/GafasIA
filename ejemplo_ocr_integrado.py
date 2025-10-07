# ejemplo_ocr_integrado.py - Sistema completo con detección de objetos y OCR
import cv2
import time
import signal
import sys
from src.deteccion.detector_objetos import DetectorObjetos
from src.ocr.lector_texto import LectorTexto
from src.audio.sintetizador_voz import SintetizadorVoz

class GafasIACompleto:
    def __init__(self):
        self.detector = DetectorObjetos(
            modelo_path='yolov8n.pt',
            confianza_minima=0.5
        )
        self.lector_ocr = LectorTexto(
            idioma='spa',
            confianza_minima=40
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
        
        print("Rasvision inciado")
    
    def iniciar_camara(self):
        print("Inicializando cámara...")
        
        for indice in [2, 0, 1]:
            try:
                self.camara = cv2.VideoCapture(indice)
                if self.camara.isOpened():
                    print(f"Cámara {indice} inicializada")
                    break
                else:
                    self.camara.release()
            except:
                continue
        else:
            raise Exception("No se pudo inicializar ninguna cámara")
        
        # Configuracion PARA LA RASPBY   
        self.camara.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camara.set(cv2.CAP_PROP_FPS, 15)
        self.camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        return True
    
    def ejecutar(self, modo_visual: bool = False):
        try:
            self.iniciar_camara()
            self.sintetizador.decir_inicio()
            time.sleep(2)
            self.ejecutando = True
            self._mostrar_controles()
            while self.ejecutando:
                ret, frame = self.camara.read()
                if not ret:
                    print("Error al capturar imagen")
                    break
                
                tiempo_actual = time.time()
                frame_a_mostrar = frame.copy()
                
                # Análisis automático según el modo
                if tiempo_actual - self.ultimo_analisis >= self.intervalo_deteccion:
                    if self.modo_actual == 'objetos':
                        detecciones = self._analizar_objetos(frame)
                        if modo_visual and detecciones:
                            frame_a_mostrar = self.detector.dibujar_detecciones(frame, detecciones)
                    elif self.modo_actual == 'texto':
                        textos = self._analizar_texto(frame)
                        if modo_visual and textos:
                            frame_a_mostrar = self.lector_ocr.dibujar_texto_detectado(frame, textos)
                    
                    elif self.modo_actual == 'ambos':
                        detecciones = self._analizar_objetos(frame)
                        textos = self._analizar_texto(frame)
                        
                        if modo_visual:
                            if detecciones:
                                frame_a_mostrar = self.detector.dibujar_detecciones(frame_a_mostrar, detecciones)
                            if textos:
                                frame_a_mostrar = self.lector_ocr.dibujar_texto_detectado(frame_a_mostrar, textos)
                    
                    self.ultimo_analisis = tiempo_actual
                
                if modo_visual:
                    texto_modo = f"Modo: {self.modo_actual.upper()}"
                    cv2.putText(frame_a_mostrar, texto_modo, (10, 30), #    CAMBIAR
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    cv2.imshow('RASVISION', frame_a_mostrar)
                    key = cv2.waitKey(1) & 0xFF
                    # Controles de teclado
                    if key == ord('q'):
                        print("Cerrando sistema...")
                        break
                    elif key == ord('o'):  #solo objetos
                        self._cambiar_modo('objetos')
                    elif key == ord('t'):  # solo texot
                        self._cambiar_modo('texto')
                    elif key == ord('b'):  # opcionde ambos modos
                        self._cambiar_modo('ambos')
                    elif key == ord('a'):  # analisi forzado
                        self._analisis_forzado(frame, modo_visual)
                    elif key == ord('l'):  # lectura continua
                        self._lectura_continua(frame)
                    elif key == ord('v'):  #  volumen
                        self._ajustar_volumen()
                    elif key == ord('r'):  #  velocidad
                        self._ajustar_velocidad()
                else:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nInterrupción por teclado")
        except Exception as e:
            print(f"Error en el sistema: {e}")
            self.sintetizador.decir_error()
        finally:
            self._limpiar_recursos()
    
    def _mostrar_controles(self):
        print("\nControles disponibles:")
        print("'o' → Modo detección de objetos")
        print("'t' → Modo lectura de texto (OCR)")
        print("'b' → Modo ambos (objetos + texto)")
        print("'a' → Análisis forzado inmediato")
        print("'l' → Lectura continua de documento")
        print("'v' → Ajustar volumen")
        print("'r' → Ajustar velocidad de habla")
        print("'q' → Salir del sistema")
        print(f"\nModo actual: {self.modo_actual.upper()}")
    
    def _cambiar_modo(self, nuevo_modo: str):
        modos_validos = ['objetos', 'texto', 'ambos']
        if nuevo_modo in modos_validos:
            self.modo_actual = nuevo_modo
            print(f"Modo cambiado a: {nuevo_modo.upper()}")
            
            # Anunciar cambio por voz
            if nuevo_modo == 'objetos':
                self.sintetizador.decir("Detección de objetos activado", prioridad=True)
            elif nuevo_modo == 'texto':
                self.sintetizador.decir("Lectura de texto activado", prioridad=True)
            else:
                self.sintetizador.decir("Detección completa activado", prioridad=True)
    
    def _analizar_objetos(self, frame):
        try:
            print("Analizando objetos...")
            detecciones = self.detector.detectar(frame, solo_prioritarios=True)
            
            if detecciones:
                self.sintetizador.decir_detecciones(detecciones, incluir_detalles=False)
                print(f"{len(detecciones)} objetos detectados")
            
            return detecciones
            
        except Exception as e:
            print(f"Error en análisis de objetos: {e}")
            return []
    def _analizar_texto(self, frame):
        try:
            print("Analizando texto...")
            textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
            
            if textos:
                descripcion = self.lector_ocr.generar_descripcion_audio(textos, modo='resumen')
                self.sintetizador.decir(descripcion)
                print(f"{len(textos)} textos detectados")
                
                # Mostrar textos en consola para debugging
                for i, texto in enumerate(textos[:3], 1):
                    print(f"  {i}. '{texto['texto']}' ({texto['confianza']}%) - {texto['categoria']}")
            
            return textos
            
        except Exception as e:
            print(f"Error en análisis de texto: {e}")
            return []
    
    def _analisis_forzado(self, frame, modo_visual):
        print("Análisis forzado iniciado...")
        
        if self.modo_actual == 'objetos' or self.modo_actual == 'ambos':
            print("Analizando objetos (detallado)...")
            detecciones = self.detector.detectar(frame, solo_prioritarios=False)  # Todos los objetos
            if detecciones:
                self.sintetizador.decir_detecciones(detecciones, incluir_detalles=True)
                if modo_visual:
                    frame_detecciones = self.detector.dibujar_detecciones(frame, detecciones)
                    cv2.imshow('Gafas IA Completo', frame_detecciones)
        
        if self.modo_actual == 'texto' or self.modo_actual == 'ambos':
            print("Analizando texto (detallado)...")
            textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
            if textos:
                # En análisis forzado, dar más detalles
                descripcion = self.lector_ocr.generar_descripcion_audio(textos, modo='completo')
                self.sintetizador.decir(descripcion)
                if modo_visual:
                    frame_texto = self.lector_ocr.dibujar_texto_detectado(frame, textos)
                    cv2.imshow('RasVision Completo', frame_texto)
    
    def _lectura_continua(self, frame):
        print("Iniciando lectura continua...")
        self.sintetizador.decir("Iniciando lectura de documento", prioridad=True)
        time.sleep(2)
        try:
            # detectar texto con máxima precisión
            textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True)
            if not textos:
                self.sintetizador.decir("No se detectó texto legible en el documento")
                return
            # fragmentos para lectura continua
            fragmentos = self.lector_ocr.leer_texto_continuo(frame, velocidad_lectura='normal')
            print(f"Leyendo {len(fragmentos)} fragmentos de texto...")
            
            for i, fragmento in enumerate(fragmentos, 1):
                print(f"Fragmento {i}/{len(fragmentos)}: {fragmento[:50]}...")
                self.sintetizador.decir(f"Fragmento {i}. {fragmento}")
                self.sintetizador.esperar_finalizacion(timeout=30)
                time.sleep(1)
            
            self.sintetizador.decir("Lectura de documento completada")
            
        except Exception as e:
            print(f"Error en lectura continua: {e}")
            self.sintetizador.decir("Error durante la lectura del documento")
    
    def _ajustar_volumen(self):
        print("Ajuste de volumen - Presiona:")
        print("1 → Bajo (50%), 2 → Normal (80%), 3 → Alto (100%)")
        self.sintetizador.decir("Ajuste de volumen. Presiona 1 para bajo, 2 para normal, 3 para alto", prioridad=True)
    def _ajustar_velocidad(self):
        print("Ajuste de velocidad - Presiona:")
        print("1 → Lenta (150), 2 → Normal (180), 3 → Rápida (220)")
        self.sintetizador.decir("Ajuste de velocidad. Presiona 1 para lenta, 2 para normal, 3 para rápida", prioridad=True)
    def _manejador_cierre(self, signal_num, frame):
        print(f"\nSeñal {signal_num} recibida, cerrando sistema...")
        self.ejecutando = False
    def _limpiar_recursos(self):
        print("Limpiando recursos...")
        if self.camara:
            self.camara.release()
            print("Cámara liberada")
        cv2.destroyAllWindows()
        if self.sintetizador:
            self.sintetizador.decir("Sistema desactivado", prioridad=True)
            time.sleep(2)
            self.sintetizador.finalizar()
        print("Recursos liberados correctamente")
    def modo_prueba_ocr(self, ruta_imagen: str = None):
        print("Prueba OCR activado")
        
        if ruta_imagen:
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                print(f"No se pudo cargar la imagen: {ruta_imagen}")
                return
        else:
            # Tomar foto con la camara
            self.iniciar_camara()
            ret, imagen = self.camara.read()
            if not ret:
                print("No se pudo capturar imagen")
                return
            
            print("Foto capturada para análisis OCR")
            if self.camara:
                self.camara.release()
        
        # Probar detección de texto
        print("Probando detección de texto...")
        textos = self.lector_ocr.detectar_texto(imagen, mejorar_imagen=True)
        
        if textos:
            print(f"{len(textos)} textos detectados:")
            for i, texto in enumerate(textos, 1):
                print(f"  {i}. '{texto['texto']}' ({texto['confianza']}%) - {texto['posicion']}")
            
            descripcion = self.lector_ocr.generar_descripcion_audio(textos, modo='completo')
            print(f"Descripción de audio: '{descripcion}'")
            
            if self.sintetizador.disponible:
                self.sintetizador.decir(descripcion)
                time.sleep(len(descripcion) * 0.1)
            
            imagen_resultado = self.lector_ocr.dibujar_texto_detectado(imagen, textos)
            cv2.imshow('OCR - Texto Detectado', imagen_resultado)
            print("Presiona cualquier tecla para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        else:
            print("No se detectó texto en la imagen")
            self.sintetizador.decir("No se detectó texto legible en la imagen")
        
        print("Prueba OCR completada")


def main():
    print("Rasvision")
    print("Detección de Objetos + Reconocimiento de Texto (OCR)")
    gafas = GafasIACompleto()
    try:
        print("\nModos disponibles :")  #RECORDAR ARRGLAR
        print("Sistema completo (modo visual)")
        print("Sistema completo (solo audio)")
        print("Prueba OCR con imagen")
        print("Prueba OCR con cámara")
        
        opcion = input("\n ").strip()
        
        if opcion == '1':
            print("\nIniciando en modo visual completo...")
            gafas.ejecutar(modo_visual=True)
        elif opcion == '2':
            print("\nIniciando en modo solo audio...")
            gafas.ejecutar(modo_visual=False)
        elif opcion == '3':
            ruta = input("Ingresa la ruta de la imagen: ").strip()
            gafas.modo_prueba_ocr(ruta_imagen=ruta)
        elif opcion == '4':
            print("\nCapturando imagen para prueba OCR...")
            gafas.modo_prueba_ocr()
        else:
            print("Opción no válida")
            
    except Exception as e:
        print(f"Error crítico: {e}")
    finally:
        print("\n¡Gracias por usar Gafas IA!")


if __name__ == "__main__":
    main()