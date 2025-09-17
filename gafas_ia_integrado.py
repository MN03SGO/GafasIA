import cv2
import time
import signal
import sys
from src.deteccion.detector_objetos import DetectorObjetos
from src.audio.sintetizador_voz import SintetizadorVoz

class GafasIA:
    def __init__(self):
        print(" Iniciando RasVision...")
        
    
        self.detector = DetectorObjetos(
            modelo_path='yolov8n.pt',
            confianza_minima=0.5
        )
        
        self.sintetizador = SintetizadorVoz(
            idioma='es',
            velocidad=180,  # Velocidad cómoda para personas con discapacidad visual
            volumen=0.8
        )
        
        self.intervalo_deteccion = 3  # Detectar cada 3 segundos
        self.ultimo_analisis = 0
        self.camara = None
        self.ejecutando = False
        

        signal.signal(signal.SIGINT, self._manejador_cierre)
        signal.signal(signal.SIGTERM, self._manejador_cierre)
        
        print("Sistema RasVision inicializado correctamente")
    
    def iniciar_camara(self):
    
        print("Inicializando cámara...")
    
        for indice in [0, 1, 2]:
            try:
                self.camara = cv2.VideoCapture(indice)
                if self.camara.isOpened():
                    print(f" Cámara {indice} inicializada correctamente")
                    break
                else:
                    self.camara.release()
            except:
                continue
        else:
            raise Exception("No se pudo inicializar ninguna cámara")
        
        #Raspberry Pi
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
            print("Sistema en funcionamiento. Controles:")
            print("   - Presiona 'a' para análisis inmediato")
            print("   - Presiona 'v' para ajustar volumen")
            print("   - Presiona 'r' para ajustar velocidad")
            print("   - Presiona 'q' para salir")
            
            while self.ejecutando:
                # Capturar frame
                ret, frame = self.camara.read()
                if not ret:
                    print("Error al capturar imagen de la cámara")
                    break
                
                tiempo_actual = time.time()
                
                # analisis automático cada cierto tiempo
                if tiempo_actual - self.ultimo_analisis >= self.intervalo_deteccion:
                    self._analizar_frame(frame)
                    self.ultimo_analisis = tiempo_actual
                
                if modo_visual:
                    cv2.imshow('Gafas IA', frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    #  comandos de teclado
                    if key == ord('q'):
                        print(" Cerrando sistema...")
                        break
                    elif key == ord('a'):
                        print("🔍 Análisis forzado...")
                        self._analizar_frame(frame, forzado=True)
                    elif key == ord('v'):
                        self._ajustar_volumen()
                    elif key == ord('r'):
                        self._ajustar_velocidad()
                else:
                    time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n Interrupción por teclado recibida")
        except Exception as e:
            print(f"Error en el sistema: {e}")
            self.sintetizador.decir_error()
        finally:
            self._limpiar_recursos()
    
    def _analizar_frame(self, frame, forzado=False):
        try:
            if forzado:
                print("Análisis forzado iniciado...")
            else:
                print("Análisis automático iniciado...")
            
            # Detectar objetos
            inicio_deteccion = time.time()
            detecciones = self.detector.detectar(frame, solo_prioritarios=True)
            tiempo_deteccion = time.time() - inicio_deteccion
            
            print(f"⏱Detección completada en {tiempo_deteccion:.2f}s - {len(detecciones)} objetos")
            
            if forzado:
                # En análisis forzado
                self.sintetizador.decir_detecciones(detecciones, incluir_detalles=True)
            else:
                # En análisis automático
                self.sintetizador.decir_detecciones(detecciones, incluir_detalles=False)
            
            # Logs
            if detecciones:
                print("Detecciones encontradas:")
                for i, det in enumerate(detecciones[:5], 1):
                    print(f"  {i}. {det['nombre']} - {det['confianza']:.2f} - {det['posicion']} - {det['distancia_relativa']}")
                    
                    if len(detecciones) > 5:
                        print(f"  ... y {len(detecciones)-5} más")
                        break
            
        except Exception as e:
            print(f"Error en análisis: {e}")
            self.sintetizador.decir("Error al analizar la imagen")
    
    def _ajustar_volumen(self):
        """Permite ajustar el volumen del sistema"""
        print("Ajuste de volumen:")
        print("1. Bajo (50%)")
        print("2. Normal (80%)")
        print("3. Alto (100%)")
        
        try:
            opcion = input("Selecciona opción (1-3): ")
            
            if opcion == '1':
                self.sintetizador.configurar(volumen=0.5)
            elif opcion == '2':
                self.sintetizador.configurar(volumen=0.8)
            elif opcion == '3':
                self.sintetizador.configurar(volumen=1.0)
            else:
                self.sintetizador.decir("Opción no válida")
                
        except:
            print("Error ajustando volumen")
    
    def _ajustar_velocidad(self):
    
        print("Ajuste de velocidad:")
        print("1. Lenta (150 ppm)")
        print("2. Normal (180 ppm)")
        print("3. Rápida (220 ppm)")
        
        try:
            opcion = input("Selecciona opción (1-3): ")
            
            if opcion == '1':
                self.sintetizador.configurar(velocidad=150)
            elif opcion == '2':
                self.sintetizador.configurar(velocidad=180)
            elif opcion == '3':
                self.sintetizador.configurar(velocidad=220)
            else:
                self.sintetizador.decir("Opción no válida")
                
        except:
            print("Error ajustando velocidad")
    
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
            time.sleep(2) #tiempo de repodruccion
            self.sintetizador.finalizar()
        
        print("Recursos liberados correctamente")
    
    def modo_prueba(self):
        print("Modo de prueba activado")
        
        
        self.sintetizador.probar_voz()
        time.sleep(3)
        
        # Simular detecciones
        detecciones_simuladas = [
            {'clase_id': 56, 'nombre': 'silla', 'confianza': 0.89, 'posicion': 'a la derecha', 'distancia_relativa': 'cerca'},
            {'clase_id': 0, 'nombre': 'persona', 'confianza': 0.95, 'posicion': 'en el centro', 'distancia_relativa': 'muy cerca'}
        ]
        
        print("Simulando detecciones...")
        self.sintetizador.decir_detecciones(detecciones_simuladas, incluir_detalles=True)
        
        time.sleep(5)
        
        print("Prueba completada...")


def main():
    """Función principal"""
    print("=" * 50)
    print(" RasVision ")
    print("   Desarrollado para personas con discapacidad visual")
    print("=" * 50)
    
    gafas = GafasIA()
    
    try:
        #modo de operación
        print("\nModos disponibles:")
        print("1. Modo normal (solo audio)")
        print("2. Modo visual (con ventana de video)")
        print("3. Modo prueba (sin cámara)")
        
        opcion = input("\nSelecciona modo (1-3): ").strip()
        
        if opcion == '1':
            gafas.ejecutar(modo_visual=False)
        elif opcion == '2':
            gafas.ejecutar(modo_visual=True)
        elif opcion == '3':
            gafas.modo_prueba()
        else:
            print("Opción no válida")
            
    except Exception as e:
        print(f" Error crítico: {e}")
    finally:
        print("\n¡Gracias por usar Gafas IA!")


if __name__ == "__main__":
    main()
