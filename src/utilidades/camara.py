import cv2 
import time 
import threading
from queue import Queue, Empty
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

try:
    from .logger import obtener_logger_modulo
    from .config import obtener_configuracion
    logger = obtener_logger_modulo("Gestor de camara")
except ImportError:
    import logging
    logger = logging.getLogger("Gestor de camara")

#imoirtacion de la raspbi
PICAMERA2_DISPONIBLE = False
try: 
    from picamera2 import Picamera2
    from libcamera import controls
    PICAMERA2_DISPONIBLE = True
    logger.info("Picamera2 disponible para Raspberry Pi")
except ImportError:
    logger.debug("Picamera2 no disponible - usando OpenCV estándar")
    
class DetectorCamaras:
    @staticmethod
    @staticmethod
    def listar_camaras_disponibles(max_camaras=10) -> List[Dict[str, Any]]:
        camaras_encontradas = []
        logger.info(f"Camaras disponibles (máximo {max_camaras})...")
        
        for i in range(max_camaras):
            cap = None
            try:
                # Intentar abrir cámara con diferentes backends
                backends = [cv2.CAP_ANY, cv2.CAP_V4L2, cv2.CAP_DSHOW]
                for backend in backends:
                    try:
                        cap= cv2.VideoCapture(i, backend)
                        if cap.is0pened():
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                info_camara = {
                                    'indice': i,
                                    'backend': DetectorCamaras._obtener_nombre_backend(backend),
                                    'ancho': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                    'alto': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                    'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                                    'nombre': DetectorCamaras._obtener_nombre_camara(i),
                                    'activa': True
                                }
                                camaras_encontradas.append(info_camara)
                                logger.info(f" Cámara {i}: {info_camara['ancho']}x{info_camara['alto']} @{info_camara['fps']}fps")
                                break
                    except Exception:
                        continue
                        
            except Exception as e:
                logger.debug(f"Error probando cámara {i}: {e}")
            finally:
                if cap is not None:
                    cap.release()
        
        if not camaras_encontradas:
            logger.warning("No se encontraron cámaras disponibles")
        else:
            logger.info(f"Encontradas {len(camaras_encontradas)} cámaras")
            
        return camaras_encontradas
    
    @staticmethod
    def _obtener_nombre_backend(backend_id):
        #Obtiene el nombre del backend de OpenCV
        backends = {
            cv2.CAP_ANY: "ANY",
            cv2.CAP_V4L2: "V4L2", 
            cv2.CAP_DSHOW: "DSHOW",
            cv2.CAP_GSTREAMER: "GSTREAMER"
        }
        return backends.get(backend_id, f"UNKNOWN_{backend_id}")
    
    @staticmethod
    def _obtener_nombre_camara(indice):
        try:
            if Path(f"/sys/class/video4linux/video{indice}/name").exists():
                with open(f"/sys/class/video4linux/video{indice}/name", 'r') as f:
                    return f.read().strip()
        except Exception:
            pass
        
        return f"Cámara {indice}"


class ConfiguradorCamara:

    @staticmethod
    def aplicar_configuracion_opencv(cap: cv2.VideoCapture, config: Dict[str, Any]) -> bool:
        try:
            # configuracion de la resolucion
            resolucion = config.get('resolucion', {})
            ancho = resolucion.get('ancho', 640)
            alto = resolucion.get('alto', 480)
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
            
            # fps
            fps = config.get('fps', 30)
            cap.set(cv2.CAP_PROP_FPS, fps)
            
            #
            backend = config.get('backend', 'any')
            if backend.lower() != 'any':
                logger.info(f"Backend configurado: {backend}")
            
            # parametros de las img
            brillo = config.get('brillo', 0)
            if brillo != 0:
                cap.set(cv2.CAP_PROP_BRIGHTNESS, brillo)
            
            contraste = config.get('contraste', 0)
            if contraste != 0:
                cap.set(cv2.CAP_PROP_CONTRAST, contraste)
            
            saturacion = config.get('saturacion', 0)
            if saturacion != 0:
                cap.set(cv2.CAP_PROP_SATURATION, saturacion)
            
            exposicion = config.get('exposicion', -1)
            if exposicion != -1:
                cap.set(cv2.CAP_PROP_EXPOSURE, exposicion)
            
            
            ancho_real = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            alto_real = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_real = int(cap.get(cv2.CAP_PROP_FPS))
            
            logger.info(f"Cámara configurada: {ancho_real}x{alto_real} @{fps_real}fps")
            
            return True
            
        except Exception as e:
            logger.error(f"Error aplicando configuración de cámara: {e}")
            return False
    
    @staticmethod
    def aplicar_configuracion_picamera(picam, config: Dict[str, Any]) -> bool:
        
        try:
            # configuracion de resolucion
            resolucion = config.get('resolucion', {})
            ancho = resolucion.get('ancho', 640)
            alto = resolucion.get('alto', 480)
            
            # configuracion de la picamera
            config_picam = picam.create_still_configuration(
                main={"size": (ancho, alto)},
                buffer_count=config.get('raspberry_pi', {}).get('buffer_size', 1)
            )
            
            picam.configure(config_picam)
            
            # Configurar controles de imagen
            controles = {}
            
            if config.get('brillo', 0) != 0:
                controles[controls.Brightness] = config['brillo']
            
            if config.get('contraste', 0) != 0:
                controles[controls.Contrast] = config['contraste']
            
            if config.get('saturacion', 0) != 0:
                controles[controls.Saturation] = config['saturacion']
            
            if controles:
                picam.set_controls(controles)
            
            logger.info(f"Picamera2 configurada: {ancho}x{alto}")
            return True
            
        except Exception as e:
            logger.error(f"Error configurando Picamera2: {e}")
            return False

class CapturadorFrames:
    
    def __init__(self, buffer_size: int = 1):

        self.buffer_size = buffer_size
        self.buffer = Queue(maxsize=buffer_size)
        self.capturando = False
        self.hilo_captura = None
        self.cap = None
        self.stats = {
            'frames_capturados': 0,
            'frames_perdidos': 0,
            'tiempo_inicio': None,
            'ultimo_frame': None
        }
        
    def iniciar_captura(self, cap: cv2.VideoCapture):
        
        self.cap = cap
        self.capturando = True
        self.stats['tiempo_inicio'] = time.time()
        
        self.hilo_captura = threading.Thread(target=self._captura_continua, daemon=True)
        self.hilo_captura.start()
        
        logger.info(f"Captura de frames iniciada (buffer: {self.buffer_size})")
    
    def _captura_continua(self):
        
        while self.capturando and self.cap is not None:
            try:
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Intentar agregar frame al buffer
                    try:
                        self.buffer.put(frame, block=False)
                        self.stats['frames_capturados'] += 1
                        self.stats['ultimo_frame'] = time.time()
                    except:
                        # Buffer lleno, descartar frame
                        self.stats['frames_perdidos'] += 1
                        # Limpiar buffer para mantener frames más recientes
                        try:
                            self.buffer.get_nowait()
                            self.buffer.put(frame, block=False)
                        except Empty:
                            pass
                else:
                    time.sleep(0.001)  # Pequeña pausa si no hay frame
                    
            except Exception as e:
                logger.error(f"Error en captura continua: {e}")
                break
    
    def obtener_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
    
        try:
            return self.buffer.get(timeout=timeout)
        except Empty:
            return None
    
    def detener_captura(self):

        self.capturando = False
        
        if self.hilo_captura and self.hilo_captura.is_alive():
            self.hilo_captura.join(timeout=1.0)
        
        # Limpiar buffer
        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except Empty:
                break
        
        logger.info("Captura de frames detenida")
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
    
        if self.stats['tiempo_inicio']:
            tiempo_transcurrido = time.time() - self.stats['tiempo_inicio']
            fps_promedio = self.stats['frames_capturados'] / tiempo_transcurrido if tiempo_transcurrido > 0 else 0
        else:
            fps_promedio = 0
        
        return {
            'frames_capturados': self.stats['frames_capturados'],
            'frames_perdidos': self.stats['frames_perdidos'],
            'fps_promedio': round(fps_promedio, 2),
            'buffer_ocupacion': self.buffer.qsize(),
            'ultimo_frame_hace': time.time() - self.stats['ultimo_frame'] if self.stats['ultimo_frame'] else None
        }


class GestorCapturas:
    
    def __init__(self, config_capturas: Dict[str, Any]):
    
        self.config = config_capturas
        self.ruta_capturas = Path(config_capturas.get('ruta', 'assets/images/'))
        self.formato = config_capturas.get('formato', 'jpg').lower()
        self.calidad = config_capturas.get('calidad', 85)
        self.incluir_timestamp = config_capturas.get('incluir_timestamp', True)
        
        #directorio de capturas
        self.ruta_capturas.mkdir(parents=True, exist_ok=True)
        
        logger.info(f" Gestor de capturas inicializado: {self.ruta_capturas}")
    
    def guardar_captura(self, frame: np.ndarray, nombre_archivo: Optional[str] = None) -> Optional[str]:
    
        try:
        
            if nombre_archivo is None:
                if self.incluir_timestamp:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    nombre_archivo = f"captura_{timestamp}.{self.formato}"
                else:
                    nombre_archivo = f"captura.{self.formato}"
            
            ruta_completa = self.ruta_capturas / nombre_archivo
            
            # parametros guardados
            params = []
            if self.formato in ['jpg', 'jpeg']:
                params = [cv2.IMWRITE_JPEG_QUALITY, self.calidad]
            elif self.formato == 'png':
                params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (self.calidad // 10)]
            
            # Guardar imagen
            exito = cv2.imwrite(str(ruta_completa), frame, params)
            
            if exito:
                logger.info(f" Captura guardada: {nombre_archivo}")
                return str(ruta_completa)
            else:
                logger.error(f" Error guardando captura: {nombre_archivo}")
                return None
                
        except Exception as e:
            logger.error(f" Error en guardar_captura: {e}")
            return None


class GestorCamara:

    def __init__(self, config_manager=None):

        # Obtener configuración
        self.config_manager = config_manager or obtener_configuracion()
        self.config_camara = self.config_manager.obtener('camara', {})
        self.config_raspberry = self.config_manager.obtener('raspberry_pi', {})
        
        # estado gestor 
        self.cap = None
        self.picam = None
        self.usando_picamera = False
        self.activa = False
        
        # Componentes
        self.capturador = None
        self.gestor_capturas = None
        
        # Configurar gestor de capturas si está habilitado
        if self.config_raspberry.get('capturas', {}).get('guardar_automatico', False):
            self.gestor_capturas = GestorCapturas(self.config_raspberry['capturas'])
        
        # Estadísticas
        self.estadisticas = {
            'frames_procesados': 0,
            'tiempo_inicio': None,
            'errores': 0,
            'ultima_captura': None
        }
        
        logger.info("Gestor de cámara inicializado")
    
    def inicializar_camara(self) -> bool:
        logger.info("Inicializando cámara...")

        # Decidir si usar Picamera2 o OpenCV
        es_raspberry = self.config_manager.es_raspberry_pi()
        usar_picamera = (es_raspberry and 
                        PICAMERA2_DISPONIBLE and 
                        self.config_raspberry.get('usar_picamera', False))
        
        if usar_picamera:
            return self._inicializar_picamera()
        else:
            return self._inicializar_opencv()
    
    def _inicializar_opencv(self) -> bool:
        # **Inicializa cámara usando OpenCV**
        try:
            # Obtener índice de cámara y backend
            indice_camara = self.config_camara.get('indice', 0)
            backend_str = self.config_camara.get('backend', 'any').upper()
            
            # Mapear backend string a constante OpenCV
            backends = {
                'ANY': cv2.CAP_ANY,
                'V4L2': cv2.CAP_V4L2,
                'DSHOW': cv2.CAP_DSHOW,
                'GSTREAMER': cv2.CAP_GSTREAMER
            }
            backend = backends.get(backend_str, cv2.CAP_ANY)
            
            # Abrir cámara
            self.cap = cv2.VideoCapture(indice_camara, backend)
            
            if not self.cap.isOpened():
                logger.error(f"No se pudo abrir la cámara {indice_camara}")
                return False
            
            exito_config = ConfiguradorCamara.aplicar_configuracion_opencv(
                self.cap, self.config_camara
            )
            
            if not exito_config:
                logger.warning("Algunas configuraciones no se pudieron aplicar")
            
            # Verificar que la cámara funciona
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("No se pudo capturar frame de prueba")
                self.cap.release()
                return False
            
            # Inicializar capturador de frames
            buffer_size = self.config_manager.obtener('rendimiento.buffer_frames', 1)
            self.capturador = CapturadorFrames(buffer_size)
            self.capturador.iniciar_captura(self.cap)
            
            self.activa = True
            self.usando_picamera = False
            self.estadisticas['tiempo_inicio'] = time.time()
            
            logger.info(f"Cámara OpenCV inicializada (índice: {indice_camara}, backend: {backend_str})")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando OpenCV: {e}")
            return False
    
    def _inicializar_picamera(self) -> bool:
        try:
            self.picam = Picamera2()
            
            exito_config = ConfiguradorCamara.aplicar_configuracion_picamera(
                self.picam, self.config_camara
            )
            
            if not exito_config:
                logger.warning("Algunas configuraciones no se pudieron aplicar")
            
            # Iniciar cámara
            self.picam.start()
            
            # Captura de prueba
            frame = self.picam.capture_array()
            if frame is None:
                logger.error("No se pudo capturar frame de prueba con Picamera2")
                return False
            
            self.activa = True
            self.usando_picamera = True
            self.estadisticas['tiempo_inicio'] = time.time()
            
            logger.info("Picamera2 inicializada correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando Picamera2: {e}")
            return False
    
    def obtener_frame(self) -> Optional[np.ndarray]:
        
        if not self.activa:
            return None
        
        try:
            if self.usando_picamera:
                # Captura directa con Picamera2
                frame = self.picam.capture_array()
                
                # Aplicar transformaciones si están configuradas
                if self.config_raspberry.get('flip_horizontal', False):
                    frame = cv2.flip(frame, 1)
                if self.config_raspberry.get('flip_vertical', False):
                    frame = cv2.flip(frame, 0)
                
            else:
                # Usar capturador con buffer para OpenCV
                if self.capturador:
                    frame = self.capturador.obtener_frame()
                else:
                    # Fallback a captura directa
                    ret, frame = self.cap.read()
                    frame = frame if ret else None
            
            if frame is not None:
                self.estadisticas['frames_procesados'] += 1
                self.estadisticas['ultima_captura'] = time.time()
            
            return frame
            
        except Exception as e:
            logger.error(f"Error obteniendo frame: {e}")
            self.estadisticas['errores'] += 1
            return None
    
    def capturar_imagen(self, nombre_archivo: Optional[str] = None) -> Optional[str]:
    
        if not self.gestor_capturas:
            logger.warning(" Gestor de capturas no está configurado")
            return None
        
        frame = self.obtener_frame()
        if frame is None:
            logger.error("No se pudo obtener frame para captura")
            return None
        
        return self.gestor_capturas.guardar_captura(frame, nombre_archivo)
    
    def obtener_propiedades_camara(self) -> Dict[str, Any]:
    
        if not self.activa:
            return {}
        
        propiedades = {}
        
        try:
            if self.usando_picamera:
                # Propiedades de Picamera2
                propiedades = {
                    'tipo': 'Picamera2',
                    'ancho': self.config_camara.get('resolucion', {}).get('ancho', 'Desconocido'),
                    'alto': self.config_camara.get('resolucion', {}).get('alto', 'Desconocido'),
                    'fps': 'Variable',
                    'backend': 'Picamera2'
                }
            else:
                # Propiedades de OpenCV
                propiedades = {
                    'tipo': 'OpenCV',
                    'ancho': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'alto': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                    'brillo': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                    'contraste': self.cap.get(cv2.CAP_PROP_CONTRAST),
                    'saturacion': self.cap.get(cv2.CAP_PROP_SATURATION),
                    'exposicion': self.cap.get(cv2.CAP_PROP_EXPOSURE),
                    'backend': self.config_camara.get('backend', 'any')
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo propiedades: {e}")
        
        return propiedades
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
    
        stats = dict(self.estadisticas)
        
        # Calcular FPS promedio
        if self.estadisticas['tiempo_inicio']:
            tiempo_transcurrido = time.time() - self.estadisticas['tiempo_inicio']
            stats['fps_promedio'] = (
                self.estadisticas['frames_procesados'] / tiempo_transcurrido 
                if tiempo_transcurrido > 0 else 0
            )
        else:
            stats['fps_promedio'] = 0
        
        
        if self.capturador:
            stats['capturador'] = self.capturador.obtener_estadisticas()
        
        return stats
    
    def esta_activa(self) -> bool:
    
        return self.activa
    
    def finalizar(self):
        logger.info("Finalizando gestor de cámara...")
        
        self.activa = False
        
        # Detener capturador
        if self.capturador:
            self.capturador.detener_captura()
            self.capturador = None
        
        # Liberar OpenCV
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Liberar Picamera2
        if self.picam:
            try:
                self.picam.stop()
            except Exception as e:
                logger.debug(f"Error deteniendo Picamera2: {e}")
            self.picam = None
        
        logger.info("Gestor de cámara finalizado")


# Funciones de utilidad para uso directo
def listar_camaras_sistema() -> List[Dict[str, Any]]:
   
    return DetectorCamaras.listar_camaras_disponibles()


def crear_gestor_camara(config_manager=None) -> GestorCamara:
    
    return GestorCamara(config_manager)




# Ejemplo de uso
if __name__ == "__main__":
    print("Probando el gestor de cámara GafasIA...")
    
    try:
        # Listar cámaras disponibles
        print("\nCámaras disponibles:")
        camaras = listar_camaras_sistema()
        for camara in camaras:
            print(f"Cámara {camara['indice']}: {camara['nombre']} ({camara['ancho']}x{camara['alto']})")
        
        if not camaras:
            print("No se encontraron cámaras")
            exit(1)
        
        # Crear gestor de cámara
        gestor = crear_gestor_camara()
        
        # Inicializar cámara
        if gestor.inicializar_camara():
            print("Cámara inicializada correctamente")
            
            # Mostrar propiedades de la cámara
            propiedades = gestor.obtener_propiedades_camara()
            print(f"Propiedades: {propiedades}")
            
            # Capturar algunos frames de prueba
            print("\n Capturando frames de prueba...")
            for i in range(5):
                frame = gestor.obtener_frame()
                if frame is not None:
                    print(f"   Frame {i+1}: {frame.shape}")
                    time.sleep(0.1)
                else:
                    print(f"  Frame {i+1}: Error")
            
            # Mostrar estadísticas
            stats = gestor.obtener_estadisticas()
            print(f"\n Estadísticas:")
            print(f"  Frames procesados: {stats['frames_procesados']}")
            print(f"  FPS promedio: {stats['fps_promedio']:.2f}")
            print(f"  Errores: {stats['errores']}")
            
            # Finalizar
            gestor.finalizar()
            print("Gestor finalizado correctamente")
            
        else:
            print(" Error inicializando la cámara")
            
    except Exception as e:
        print(f" Error en prueba: {e}")
    