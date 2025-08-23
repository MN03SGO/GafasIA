import cv2
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import signal
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilidades.config import Config
from utilidades.logger import Logger
from utilidades.camara import GestorCamara
from deteccion.detector_objetos import DetectorObjetos
from audio.texto_a_voz import TextoAVoz

class GestorPrincipal:
    def __init__(self, config_path: str = "config/ajuste.yaml"):
        self.config = Config(config_path)
        self.logger = Logger().get_logger()
        self._ejecutando = False
        self._pausado = False
        self._hilo_principal = None
        self._lock = threading.Lock()
        #Subsistemas
        self.camara = None
        self.detector = None
        self.tts = None
        # Detecciones
        self.ultima_deteccion = None
        self.intervalo_deteccion = self.config.get("deteccion.intervalo_frames", 30)
        self.contador_frames = 0
        self.ultima_vez_anunciado = {}
        self.cooldown_objetos = self.config.get("tts.cooldown_segundos", 3.0)
        # Metricas
        self.metricas = {
            'frames_procesados': 0,
            'detecciones_realizadas': 0,
            'objetos_detectados': 0,
            'mensajes_tts': 0,
            'tiempo_inicio': None,
            'fps_promedio': 0.0
        }

        self._configurar_senales()

        self.logger.info("GestorPrincipal inicializado")
    def _configurar_senales(self):
        def signal_handler(signum, frame):
            self.logger.info(f"Señal recibida: {signum}")
            self.detener()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def inicializar_subsistemas(self) -> bool:
        
        try:
            self.logger.info("Inicializando subsistemas...")
            # Camara
            self.logger.info("Inicializando cámara...")
            self.camara = GestorCamara()
            if not self.camara.inicializar():
                self.logger.error("Error al inicializar cámara")
                return False
            # Detector de objetos
            self.logger.info("Inicializando detector de objetos...")
            self.detector = DetectorObjetos()
            if not self.detector.inicializar():
                self.logger.error("Error al inicializar detector")
                return False
            # TTS
            self.logger.info("Inicializando sistema de texto a voz...")
            self.tts = TextoAVoz()
            if not self.tts.inicializar():
                self.logger.error("Error al inicializar TTS")
                return False

            self.tts.decir("Sistema RasVision iniciado correctamente", prioridad=1)
            self.logger.info("Todos los subsistemas inicializados correctamente")
            return True

        except Exception as e:
            self.logger.error(f"Error al inicializar subsistemas: {e}")
            return False

    def iniciar(self) -> bool:
        
        if self._ejecutando:
            self.logger.warning("El sistema ya está en ejecución")
            return True
        if not self.inicializar_subsistemas():
            return False
        with self._lock:
            self._ejecutando = True
            self._pausado = False
            self.metricas['tiempo_inicio'] = datetime.now()
        #hilo principal
        self._hilo_principal = threading.Thread(target=self._bucle_principal, daemon=True)
        self._hilo_principal.start()

        self.logger.info("Sistema GafasIA iniciado")
        return True

    def _bucle_principal(self):
        self.logger.info("Iniciando bucle principal de procesamiento")

        tiempo_ultimo_fps = time.time()
        frames_fps = 0

        try:
            while self._ejecutando:
                if self._pausado:
                    time.sleep(0.1)
                    continue
                # Capturar frame
                frame = self.camara.capturar_frame()
                if frame is None:
                    self.logger.warning("No se pudo capturar frame")
                    time.sleep(0.1)
                    continue
                #  metricas de frames
                self.metricas['frames_procesados'] += 1
                frames_fps += 1
                # Calcular FPS cada segundo
                tiempo_actual = time.time()
                if tiempo_actual - tiempo_ultimo_fps >= 1.0:
                    fps = frames_fps / (tiempo_actual - tiempo_ultimo_fps)
                    self.metricas['fps_promedio'] = fps
                    frames_fps = 0
                    tiempo_ultimo_fps = tiempo_actual

                self._procesar_frame(frame)

                # breack para no saturar la gpu
                time.sleep(0.01)

        except Exception as e:
            self.logger.error(f"Error en bucle principal: {e}")
        finally:
            self.logger.info("Bucle principal terminado")

    def _procesar_frame(self, frame: np.ndarray):
        
        self.contador_frames += 1
        if self.contador_frames % self.intervalo_deteccion != 0:
            return
        try:
            #  detección
            detecciones = self.detector.detectar(frame)
            self.metricas['detecciones_realizadas'] += 1

            if detecciones:
                self.metricas['objetos_detectados'] += len(detecciones)
                self._procesar_detecciones(detecciones)
                self.ultima_deteccion = detecciones

        except Exception as e:
            self.logger.error(f"Error al procesar frame: {e}")

    def _procesar_detecciones(self, detecciones: List[Dict]):
        if not detecciones:
            return

        #  objetos  filtrados cooldown
        detecciones_filtradas = []
        tiempo_actual = time.time()

        for det in detecciones:
            clase = det.get('clase', 'objeto')
            confianza = det.get('confianza', 0)

            # Verificar cooldown
            if clase in self.ultima_vez_anunciado:
                tiempo_transcurrido = tiempo_actual - self.ultima_vez_anunciado[clase]
                if tiempo_transcurrido < self.cooldown_objetos:
                    continue

            # Filtrar por confianza mínima
            confianza_minima = self.config.get("deteccion.confianza_minima", 0.5)
            if confianza < confianza_minima:
                continue

            detecciones_filtradas.append(det)
            self.ultima_vez_anunciado[clase] = tiempo_actual

        #  mensaje TTS
        if detecciones_filtradas:
            mensaje = self._generar_mensaje_detecciones(detecciones_filtradas)
            if mensaje:
                self.tts.limpiar_cola()
                #  prioridad basada en distancia/posición
                prioridad = self._calcular_prioridad_mensaje(detecciones_filtradas)

                self.tts.decir(mensaje, prioridad=prioridad)
                self.metricas['mensajes_tts'] += 1

                self.logger.info(f"Anunciando: {mensaje}")

    def _generar_mensaje_detecciones(self, detecciones: List[Dict]) -> str:
        if not detecciones:
            return ""
        conteo_clases = {}
        distancias_cercanas = []

        for det in detecciones:
            clase = det.get('clase', 'objeto')
            distancia = det.get('distancia_relativa', 'media')

            if clase in conteo_clases:
                conteo_clases[clase] += 1
            else:
                conteo_clases[clase] = 1

            # objetos muy cercanos
            if distancia == 'muy_cerca':
                distancias_cercanas.append(clase)

        partes_mensaje = []

        # Objetos muy cercanos tienen prioridad
        if distancias_cercanas:
            if len(distancias_cercanas) == 1:
                partes_mensaje.append(f"¡{distancias_cercanas[0]} muy cerca!")
            else:
                partes_mensaje.append(f"¡Objetos muy cerca: {', '.join(distancias_cercanas)}!")

        # Resto de objetos
        objetos_normales = []
        for clase, cantidad in conteo_clases.items():
            if clase not in distancias_cercanas:
                if cantidad == 1:
                    objetos_normales.append(clase)
                else:
                    objetos_normales.append(f"{cantidad} {clase}s")

        if objetos_normales:
            if len(objetos_normales) == 1:
                partes_mensaje.append(f"Detectado: {objetos_normales[0]}")
            else:
                partes_mensaje.append(f"Detectados: {', '.join(objetos_normales)}")

        return ". ".join(partes_mensaje)

    def _calcular_prioridad_mensaje(self, detecciones: List[Dict]) -> int:
        # Prioridad 1 (alta): objetos muy cerca o peligrosos
        objetos_peligrosos = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        for det in detecciones:
            clase = det.get('clase', '')
            distancia = det.get('distancia_relativa', 'media')
            # Muy alta prioridad para objetos cercanos o peligrosos
            if distancia == 'muy_cerca' or clase in objetos_peligrosos:
                return 1
        # Prioridad 2 (media): objetos cercanos
        for det in detecciones:
            if det.get('distancia_relativa') == 'cerca':
                return 2
        # Prioridad 3 (baja): otros objetos
        return 3

    def pausar(self):
        with self._lock:
            self._pausado = True
        self.logger.info("Sistema pausado")
        self.tts.decir("Sistema pausado", prioridad=1)

    def reanudar(self):
        with self._lock:
            self._pausado = False
        self.logger.info("Sistema reanudado")
        self.tts.decir("Sistema reanudado", prioridad=1)

    def esta_pausado(self) -> bool:
        with self._lock:
            return self._pausado

    def detener(self):
        self.logger.info("Deteniendo sistema...")

        with self._lock:
            self._ejecutando = False
        # hilo principal
        if self._hilo_principal and self._hilo_principal.is_alive():
            self._hilo_principal.join(timeout=2.0)
        if self.tts:
            self.tts.decir("Cerrando sistema GafasIA", prioridad=1)
            time.sleep(1)  #Tiempo a que termine de hablar
            self.tts.detener()

        if self.detector:
            self.detector.limpiar()

        if self.camara:
            self.camara.liberar()

        self.logger.info("Sistema detenido completamente")

    def obtener_estado(self) -> Dict[str, Any]:
        with self._lock:
            estado = {
                'ejecutando': self._ejecutando,
                'pausado': self._pausado,
                'subsistemas': {
                    'camara': self.camara.esta_disponible() if self.camara else False,
                    'detector': self.detector.esta_inicializado() if self.detector else False,
                    'tts': self.tts.esta_activo() if self.tts else False
                },
                'metricas': self.metricas.copy(),
                'ultima_deteccion': self.ultima_deteccion
            }

        # Tiempo de ejecución
        if self.metricas['tiempo_inicio']:
            tiempo_ejecucion = datetime.now() - self.metricas['tiempo_inicio']
            estado['tiempo_ejecucion'] = str(tiempo_ejecucion).split('.')[0]

        return estado

    def cambiar_configuracion(self, clave: str, valor: Any):
        try:
            self.config.set(clave, valor)
            if clave.startswith('tts.'):
                if self.tts:
                    self.tts.actualizar_configuracion()
            elif clave.startswith('deteccion.'):
                if 'intervalo_frames' in clave:
                    self.intervalo_deteccion = valor
                elif 'confianza_minima' in clave and self.detector:
                    self.detector.actualizar_configuracion()

            self.logger.info(f"Configuración actualizada: {clave} = {valor}")
            return True

        except Exception as e:
            self.logger.error(f"Error al cambiar configuración {clave}: {e}")
            return False

    def obtener_estadisticas(self) -> Dict[str, Any]:
        estado = self.obtener_estado()

        estadisticas = {
            'rendimiento': {
                'fps_promedio': self.metricas['fps_promedio'],
                'frames_procesados': self.metricas['frames_procesados'],
                'detecciones_por_minuto': 0,
                'objetos_por_deteccion': 0
            },
            'detecciones': {
                'total_detecciones': self.metricas['detecciones_realizadas'],
                'total_objetos': self.metricas['objetos_detectados'],
                'mensajes_tts': self.metricas['mensajes_tts']
            },
            'sistema': estado
        }

        # Metricas derivadas
        if self.metricas['tiempo_inicio']:
            minutos = (datetime.now() - self.metricas['tiempo_inicio']).total_seconds() / 60
            if minutos > 0:
                estadisticas['rendimiento']['detecciones_por_minuto'] = \
                    self.metricas['detecciones_realizadas'] / minutos

        if self.metricas['detecciones_realizadas'] > 0:
            estadisticas['rendimiento']['objetos_por_deteccion'] = \
                self.metricas['objetos_detectados'] / self.metricas['detecciones_realizadas']

        return estadisticas

# Función principal para ejecutar desde línea de comandos
def main():
    print("Iniciando GafasIA - Asistente Visual Inteligente")
    print("=" * 50)

    gestor = GestorPrincipal()

    try:
        if not gestor.iniciar():
            print("Error al inicializar el sistema")
            return 1

        print("RasVision iniciado correctamente")
        print("Comandos disponibles:")
        print("  p - Pausar/Reanudar")
        print("  s - Ver estado")
        print("  e - Ver estadísticas")
        print("  q - Salir")
        print("=" * 50)

        while True:
            try:
                comando = input().strip().lower()

                if comando == 'q':
                    break
                elif comando == 'p':
                    if gestor.esta_pausado():
                        gestor.reanudar()
                        print("Sistema reanudado")
                    else:
                        gestor.pausar()
                        print("Sistema pausado")
                elif comando == 's':
                    estado = gestor.obtener_estado()
                    print("\nEstado del Sistema:")
                    print(f"  Ejecutando: {estado['ejecutando']}")
                    print(f"  Pausado: {estado['pausado']}")
                    print(f"  FPS: {estado['metricas']['fps_promedio']:.1f}")
                    print(f"  Frames: {estado['metricas']['frames_procesados']}")
                    print(f"  Detecciones: {estado['metricas']['detecciones_realizadas']}")
                    if 'tiempo_ejecucion' in estado:
                        print(f"  Tiempo: {estado['tiempo_ejecucion']}")
                    print()
                elif comando == 'e':
                    stats = gestor.obtener_estadisticas()
                    print("\n📈 Estadísticas Detalladas:")
                    print(f"  FPS promedio: {stats['rendimiento']['fps_promedio']:.1f}")
                    print(f"  Detecciones/min: {stats['rendimiento']['detecciones_por_minuto']:.1f}")
                    print(f"  Objetos/detección: {stats['rendimiento']['objetos_por_deteccion']:.1f}")
                    print(f"  Total mensajes TTS: {stats['detecciones']['mensajes_tts']}")
                    print()
                elif comando:
                    print("Comando no reconocido. Usa 'q' para salir.")

            except KeyboardInterrupt:
                break
            except EOFError:
                break

    except KeyboardInterrupt:
        print("\nInterrupción recibida...")
    except Exception as e:
        print(f"Error inesperado: {e}")
        return 1
    finally:
        print("Cerrando sistema...")
        gestor.detener()
        print("¡Hasta luego!")

    return 0


if __name__ == "__main__":
    exit(main())
