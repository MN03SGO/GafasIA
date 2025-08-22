import cv2
import numpy as np
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import queue

ULTRALYTICS_DISPONIBLE = False
TORCH_DISPONIBLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_DISPONIBLE = True
except ImportError:
    try:
        import torch
        TORCH_DISPONIBLE = True
    except ImportError:
        pass

try:
    from ..utilidades.logger import obtener_logger_modulo
    from ..utilidades.config import obtener_configuracion
    logger = obtener_logger_modulo("DetectorObjetos")
except ImportError:
    import logging
    logger = logging.getLogger("DetectorObjetos")


class TipoDeteccion(Enum):
    """Tipos de detección para categorizar objetos"""
    PERSONA = "persona"
    VEHICULO = "vehiculo"
    OBSTACULO = "obstaculo"
    OBJETO_COTIDIANO = "objeto_cotidiano"
    ELECTRODOMESTICO = "electrodomestico"
    ANIMAL = "animal"
    OTRO = "otro"


@dataclass
class Deteccion:
    """
    Clase que representa una detección de objeto
    """
    # Identificación
    clase_id: int
    clase_nombre: str
    clase_nombre_es: str
    confianza: float
    
    # Posición y tamaño
    x1: int
    y1: int
    x2: int
    y2: int
    centro_x: int
    centro_y: int
    ancho: int
    alto: int
    area: int
    
    tipo_deteccion: TipoDeteccion
    prioridad: int
    
    distancia_estimada: Optional[float] = None
    tracking_id: Optional[int] = None
    tiempo_deteccion: float = 0.0
    
    def __post_init__(self):
        #Cálculos automáticos después de inicialización
        self.centro_x = int((self.x1 + self.x2) / 2)
        self.centro_y = int((self.y1 + self.y2) / 2)
        self.ancho = abs(self.x2 - self.x1)
        self.alto = abs(self.y2 - self.y1)
        self.area = self.ancho * self.alto
        self.tiempo_deteccion = time.time()
    
    def obtener_rectangulo(self) -> Tuple[int, int, int, int]:
        # coordenadas como tupla (x1, y1, x2, y2)
        return (self.x1, self.y1, self.x2, self.y2)
    
    def obtener_centro(self) -> Tuple[int, int]:
        #retorna el centro como tupla (x, y)
        return (self.centro_x, self.centro_y)
    
    def esta_cerca_de(self, otra_deteccion: 'Deteccion', umbral_distancia: float = 50.0) -> bool:

        distancia = math.sqrt(
            (self.centro_x - otra_deteccion.centro_x) ** 2 +
            (self.centro_y - otra_deteccion.centro_y) ** 2
        )
        return distancia <= umbral_distancia
    
    def calcular_iou(self, otra_deteccion: 'Deteccion') -> float:
    
        # Calcular área de intersección
        x1_intersec = max(self.x1, otra_deteccion.x1)
        y1_intersec = max(self.y1, otra_deteccion.y1)
        x2_intersec = min(self.x2, otra_deteccion.x2)
        y2_intersec = min(self.y2, otra_deteccion.y2)
        
        if x1_intersec >= x2_intersec or y1_intersec >= y2_intersec:
            return 0.0
        
        area_intersec = (x2_intersec - x1_intersec) * (y2_intersec - y1_intersec)
        area_union = self.area + otra_deteccion.area - area_intersec
        
        return area_intersec / area_union if area_union > 0 else 0.0


class ClasificadorObjetos:

    # Mapeo de clases COCO a tipos de detección
    MAPEO_TIPOS = {
        # Personas
        'person': TipoDeteccion.PERSONA,
        
        # Vehículos
        'bicycle': TipoDeteccion.VEHICULO,
        'car': TipoDeteccion.VEHICULO,
        'motorcycle': TipoDeteccion.VEHICULO,
        'airplane': TipoDeteccion.VEHICULO,
        'bus': TipoDeteccion.VEHICULO,
        'train': TipoDeteccion.VEHICULO,
        'truck': TipoDeteccion.VEHICULO,
        'boat': TipoDeteccion.VEHICULO,
        
        # Obstáculos/Mobiliario
        'chair': TipoDeteccion.OBSTACULO,
        'couch': TipoDeteccion.OBSTACULO,
        'bed': TipoDeteccion.OBSTACULO,
        'dining table': TipoDeteccion.OBSTACULO,
        'bench': TipoDeteccion.OBSTACULO,
        'traffic light': TipoDeteccion.OBSTACULO,
        'stop sign': TipoDeteccion.OBSTACULO,
        
        # Objetos cotidianos
        'bottle': TipoDeteccion.OBJETO_COTIDIANO,
        'cup': TipoDeteccion.OBJETO_COTIDIANO,
        'fork': TipoDeteccion.OBJETO_COTIDIANO,
        'knife': TipoDeteccion.OBJETO_COTIDIANO,
        'spoon': TipoDeteccion.OBJETO_COTIDIANO,
        'bowl': TipoDeteccion.OBJETO_COTIDIANO,
        'book': TipoDeteccion.OBJETO_COTIDIANO,
        'laptop': TipoDeteccion.OBJETO_COTIDIANO,
        'cell phone': TipoDeteccion.OBJETO_COTIDIANO,
        'remote': TipoDeteccion.OBJETO_COTIDIANO,
        'keyboard': TipoDeteccion.OBJETO_COTIDIANO,
        'mouse': TipoDeteccion.OBJETO_COTIDIANO,
        
        # Electrodomésticos
        'microwave': TipoDeteccion.ELECTRODOMESTICO,
        'oven': TipoDeteccion.ELECTRODOMESTICO,
        'toaster': TipoDeteccion.ELECTRODOMESTICO,
        'sink': TipoDeteccion.ELECTRODOMESTICO,
        'refrigerator': TipoDeteccion.ELECTRODOMESTICO,
        'tv': TipoDeteccion.ELECTRODOMESTICO,
        
        # Animales
        'bird': TipoDeteccion.ANIMAL,
        'cat': TipoDeteccion.ANIMAL,
        'dog': TipoDeteccion.ANIMAL,
        'horse': TipoDeteccion.ANIMAL,
        'sheep': TipoDeteccion.ANIMAL,
        'cow': TipoDeteccion.ANIMAL,
        'elephant': TipoDeteccion.ANIMAL,
        'bear': TipoDeteccion.ANIMAL,
        'zebra': TipoDeteccion.ANIMAL,
        'giraffe': TipoDeteccion.ANIMAL,
    }
    
    # Prioridades por tipo (1 = más prioritario)
    PRIORIDADES = {
        TipoDeteccion.PERSONA: 1,
        TipoDeteccion.VEHICULO: 2,
        TipoDeteccion.OBSTACULO: 3,
        TipoDeteccion.ANIMAL: 4,
        TipoDeteccion.ELECTRODOMESTICO: 5,
        TipoDeteccion.OBJETO_COTIDIANO: 6,
        TipoDeteccion.OTRO: 7
    }
    
    def __init__(self, traducciones: Dict[str, str]):
        
        self.traducciones = traducciones
        
    def clasificar_objeto(self, clase_nombre: str) -> Tuple[TipoDeteccion, int]:
        
        tipo_deteccion = self.MAPEO_TIPOS.get(clase_nombre, TipoDeteccion.OTRO)
        prioridad = self.PRIORIDADES[tipo_deteccion]
        
        return tipo_deteccion, prioridad
    
    def traducir_nombre(self, clase_nombre: str) -> str:
        
        return self.traducciones.get(clase_nombre, clase_nombre)


class EstimadorDistancia:
    
    # Tamaños promedio de objetos en centímetros (altura)
    TAMANIOS_REALES = {
        'person': 170.0,      # Altura promedio de persona
        'car': 150.0,         # Altura promedio de auto
        'chair': 80.0,        # Altura promedio de silla
        'bottle': 25.0,       # Altura promedio de botella
        'laptop': 2.0,        # Grosor de laptop cerrada
        'cell phone': 15.0,   # Altura de teléfono
        'cup': 10.0,          # Altura promedio de taza
    }
    
    def __init__(self, distancia_focal_mm: float = 4.0, sensor_altura_mm: float = 2.7):

        self.distancia_focal_mm = distancia_focal_mm
        self.sensor_altura_mm = sensor_altura_mm
        
    def estimar_distancia(self, deteccion: Deteccion, altura_imagen: int) -> Optional[float]:
        
    
        tamanio_real_cm = self.TAMANIOS_REALES.get(deteccion.clase_nombre)
        if not tamanio_real_cm:
            return None
        
        #distancia usando la fórmula
        # Distancia = (Tamaño_Real * Distancia_Focal * Altura_Imagen) / (Tamaño_Imagen * Altura_Sensor)
        try:
            distancia_cm = (
                tamanio_real_cm * self.distancia_focal_mm * altura_imagen
            ) / (deteccion.alto * self.sensor_altura_mm)
            
            # Convertir a metros
            distancia_m = distancia_cm / 100.0
            
            # Limitar rangos razonables (0.3m a 50m)
            if 0.3 <= distancia_m <= 50.0:
                return distancia_m
            
        except (ZeroDivisionError, ValueError):
            pass
        
        return None


class FiltroNMS:
    
    @staticmethod
    def aplicar_nms(detecciones: List[Deteccion], umbral_iou: float = 0.4) -> List[Deteccion]:
        
        if not detecciones:
            return []
        
        # Ordenar por confianza (descendente)
        detecciones_ordenadas = sorted(detecciones, key=lambda x: x.confianza, reverse=True)
        detecciones_filtradas = []
        
        for deteccion_actual in detecciones_ordenadas:
        
            mantener_deteccion = True
            
            for deteccion_seleccionada in detecciones_filtradas:
            
                if deteccion_actual.clase_id == deteccion_seleccionada.clase_id:
                    iou = deteccion_actual.calcular_iou(deteccion_seleccionada)
                    
                    if iou > umbral_iou:
                        mantener_deteccion = False
                        break
            
            if mantener_deteccion:
                detecciones_filtradas.append(deteccion_actual)
        
        return detecciones_filtradas


class CargadorModelo:
    
    def __init__(self):
        self.modelo = None
        self.nombres_clases = []
        self.tipo_modelo = None
        
    def cargar_modelo(self, ruta_modelo: str, usar_gpu: bool = False) -> bool: 

        try:
            ruta_archivo = Path(ruta_modelo)
            
            if not ruta_archivo.exists():
                logger.error(f"Archivo de modelo no encontrado: {ruta_modelo}")
                return False
            
            #  cargar con Ultralytics YOLO
            if ULTRALYTICS_DISPONIBLE:
                return self._cargar_ultralytics(ruta_modelo, usar_gpu)
            
            # Fallback a OpenCV DNN
            elif ruta_archivo.suffix.lower() in ['.onnx', '.pb']:
                return self._cargar_opencv_dnn(ruta_modelo)
            
            else:
                logger.error("No hay backends disponibles para cargar el modelo")
                return False
                
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            return False
    
    def _cargar_ultralytics(self, ruta_modelo: str, usar_gpu: bool) -> bool:
        
        try:
            # Cargar modelo
            self.modelo = YOLO(ruta_modelo)
            
            # Configurar dispositivo
            if usar_gpu and TORCH_DISPONIBLE:
                import torch
                if torch.cuda.is_available():
                    self.modelo.to('cuda')
                    logger.info("Modelo cargado en GPU")
                else:
                    logger.warning("GPU solicitada pero no disponible, usando CPU")
            else:
                logger.info("Modelo cargado en CPU")
            
            # Obtener nombres de clases
            self.nombres_clases = list(self.modelo.names.values())
            self.tipo_modelo = "ultralytics"
            
            logger.info(f"Modelo Ultralytics cargado: {len(self.nombres_clases)} clases")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo Ultralytics: {e}")
            return False
    
    def _cargar_opencv_dnn(self, ruta_modelo: str) -> bool:
        
        try:
            # Cargar modelo con OpenCV DNN
            if ruta_modelo.endswith('.onnx'):
                self.modelo = cv2.dnn.readNetFromONNX(ruta_modelo)
            else:
                logger.error("Formato de modelo no soportado para OpenCV DNN")
                return False
            
            # Nombres de clases COCO por defecto
            self.nombres_clases = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
                'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
            
            self.tipo_modelo = "opencv_dnn"
            logger.info(f"Modelo OpenCV DNN cargado: {len(self.nombres_clases)} clases")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo OpenCV DNN: {e}")
            return False
    
    def predecir(self, imagen: np.ndarray, confianza: float = 0.5) -> List[Tuple]:
        
        if self.modelo is None:
            return []
        
        try:
            if self.tipo_modelo == "ultralytics":
                return self._predecir_ultralytics(imagen, confianza)
            elif self.tipo_modelo == "opencv_dnn":
                return self._predecir_opencv_dnn(imagen, confianza)
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return []
    
    def _predecir_ultralytics(self, imagen: np.ndarray, confianza: float) -> List[Tuple]:
    
        resultados = self.modelo(imagen, conf=confianza, verbose=False)
        
        detecciones = []
        for resultado in resultados:
            if resultado.boxes is not None:
                for box in resultado.boxes:
                    # Extraer información
                    coords = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    clase_id = int(box.cls[0].cpu().numpy())
                    
                    detecciones.append((
                        clase_id, conf,
                        int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    ))
        
        return detecciones
    
    def _predecir_opencv_dnn(self, imagen: np.ndarray, confianza: float) -> List[Tuple]:
    
        blob = cv2.dnn.blobFromImage(imagen, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.modelo.setInput(blob)
        
        outputs = self.modelo.forward()
        
        # Procesar resultados (implementación básica)
        detecciones = []
        altura_img, ancho_img = imagen.shape[:2]
        
        for output in outputs[0]:
            scores = output[5:]
            clase_id = np.argmax(scores)
            conf = scores[clase_id]
            
            if conf > confianza:
                # coordenadas normalizadas a píxeles
                center_x, center_y, width, height = output[0:4]
                center_x = int(center_x * ancho_img)
                center_y = int(center_y * altura_img)
                width = int(width * ancho_img)
                height = int(height * altura_img)
                
                x1 = int(center_x - width/2)
                y1 = int(center_y - height/2)
                x2 = int(center_x + width/2)
                y2 = int(center_y + height/2)
                
                detecciones.append((clase_id, float(conf), x1, y1, x2, y2))
        
        return detecciones


class DetectorObjetos:
    def __init__(self, config_manager=None):
        
        # Configuración
        self.config_manager = config_manager or obtener_configuracion()
        self.config = self.config_manager.obtener('deteccion_objetos', {})
        
        # Componentes
        self.cargador_modelo = CargadorModelo()
        self.clasificador = ClasificadorObjetos(
            self.config.get('traducciones', {})
        )
        self.estimador_distancia = EstimadorDistancia()
        
        # Estado
        self.modelo_cargado = False
        self.activo = self.config.get('activa', True)
        
        # Parámetros de detección
        self.confianza_minima = self.config.get('confianza_minima', 0.5)
        self.iou_threshold = self.config.get('iou_threshold', 0.4)
        self.max_detecciones = self.config.get('max_detecciones', 10)
        self.clases_prioritarias = set(self.config.get('clases_prioritarias', []))
        
        # Estadísticas
        self.estadisticas = {
            'detecciones_totales': 0,
            'detecciones_filtradas': 0,
            'tiempo_promedio_inferencia': 0.0,
            'ultimo_frame_procesado': 0.0,
            'objetos_detectados_sesion': {}
        }
        
        logger.info(f"Detector de objetos inicializado (activo: {self.activo})")
    
    def cargar_modelo(self) -> bool:
        
        if not self.activo:
            logger.info("Detector desactivado - omitiendo carga de modelo")
            return True
        
        # configuración del modelo
        ruta_modelo = self.config.get('modelo', 'yolov8n.pt')
        usar_gpu = self.config_manager.obtener('rendimiento.pc_desarrollo.usar_gpu', False)
        
        # Raspberry Pi, forzar CPU
        if self.config_manager.es_raspberry_pi():
            usar_gpu = False
            logger.info("Raspberry Pi detectado - usando CPU")
        
        logger.info(f"Cargando modelo: {ruta_modelo}")
        
        # Cargar modelo
        self.modelo_cargado = self.cargador_modelo.cargar_modelo(ruta_modelo, usar_gpu)
        
        if self.modelo_cargado:
            logger.info(f"Modelo cargado exitosamente: {len(self.cargador_modelo.nombres_clases)} clases")
        else:
            logger.error("Error cargando modelo - detector desactivado")
            self.activo = False
        
        return self.modelo_cargado
    
    def detectar_objetos(self, imagen: np.ndarray) -> List[Deteccion]:

        if not self.activo or not self.modelo_cargado:
            return []
        
        tiempo_inicio = time.time()
        
        try:
            # Realizar predicción
            predicciones = self.cargador_modelo.predecir(imagen, self.confianza_minima)
            
            # Convertir predicciones a objetos Deteccion
            detecciones = []
            altura_img, ancho_img = imagen.shape[:2]
            
            for clase_id, confianza, x1, y1, x2, y2 in predicciones:
                # Validar coordenadas
                x1 = max(0, min(x1, ancho_img))
                y1 = max(0, min(y1, altura_img))
                x2 = max(0, min(x2, ancho_img))
                y2 = max(0, min(y2, altura_img))
                
                # Obtener nombre de clase
                if clase_id < len(self.cargador_modelo.nombres_clases):
                    clase_nombre = self.cargador_modelo.nombres_clases[clase_id]
                else:
                    continue
                
                if self.clases_prioritarias and clase_nombre not in self.clases_prioritarias:
                    continue
                
                # Clasificar objeto
                tipo_deteccion, prioridad = self.clasificador.clasificar_objeto(clase_nombre)
                clase_nombre_es = self.clasificador.traducir_nombre(clase_nombre)
                
                # Crear detección
                deteccion = Deteccion(
                    clase_id=clase_id,
                    clase_nombre=clase_nombre,
                    clase_nombre_es=clase_nombre_es,
                    confianza=confianza,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    centro_x=0, centro_y=0,  # Se calculan automáticamente
                    ancho=0, alto=0, area=0,  # Se calculan automáticamente
                    tipo_deteccion=tipo_deteccion,
                    prioridad=prioridad
                )
                
                # Estimar distancia
                distancia = self.estimador_distancia.estimar_distancia(deteccion, altura_img)
                deteccion.distancia_estimada = distancia
                
                detecciones.append(deteccion)
            
            # Aplicar Non-Maximum Suppression
            detecciones_filtradas = FiltroNMS.aplicar_nms(detecciones, self.iou_threshold)
            
            # Limitar número máximo de detecciones
            if len(detecciones_filtradas) > self.max_detecciones:
                # Ordenar por prioridad y confianza
                detecciones_filtradas = sorted(
                    detecciones_filtradas,
                    key=lambda x: (x.prioridad, -x.confianza)
                )[:self.max_detecciones]
            
            # Actualizar estadísticas
            tiempo_inferencia = time.time() - tiempo_inicio
            self._actualizar_estadisticas(detecciones_filtradas, tiempo_inferencia)
            
            return detecciones_filtradas
            
        except Exception as e:
            logger.error(f"Error en detección de objetos: {e}")
            return []
    
    def _actualizar_estadisticas(self, detecciones: List[Deteccion], tiempo_inferencia: float):
        """Actualiza estadísticas de rendimiento"""
        self.estadisticas['detecciones_totales'] += len(detecciones)
        self.estadisticas['ultimo_frame_procesado'] = time.time()
        
        # Calcular tiempo promedio de inferencia
        if self.estadisticas['tiempo_promedio_inferencia'] == 0:
            self.estadisticas['tiempo_promedio_inferencia'] = tiempo_inferencia
        else:
            # Promedio móvil
            alpha = 0.1
            self.estadisticas['tiempo_promedio_inferencia'] = (
                alpha * tiempo_inferencia + 
                (1 - alpha) * self.estadisticas['tiempo_promedio_inferencia']
            )
        
        # objetos detectados por tipo
        for deteccion in detecciones:
            clase = deteccion.clase_nombre_es
            if clase not in self.estadisticas['objetos_detectados_sesion']:
                self.estadisticas['objetos_detectados_sesion'][clase] = 0
            self.estadisticas['objetos_detectados_sesion'][clase] += 1
    
    def filtrar_detecciones_por_area(self, detecciones: List[Deteccion], 
            area_minima: int = 100) -> List[Deteccion]:
    
        return [d for d in detecciones if d.area >= area_minima]
    
    def obtener_detecciones_por_tipo(self, detecciones: List[Deteccion], 
                                tipo: TipoDeteccion) -> List[Deteccion]:
    
        return [d for d in detecciones if d.tipo_deteccion == tipo]
    
    def obtener_detecciones_cercanas(self, detecciones: List[Deteccion],
                                distancia_maxima: float = 2.0) -> List[Deteccion]:
        

        detecciones_cercanas = []
        for deteccion in detecciones:
            if (deteccion.distancia_estimada is not None and 
                deteccion.distancia_estimada <= distancia_maxima):
                detecciones_cercanas.append(deteccion)
        
        return detecciones_cercanas
    
    def generar_descripcion_detecciones(self, detecciones: List[Deteccion]) -> str:
    
        if not detecciones:
            return "No se detectaron objetos"
        
        # Agrupar por tipo y contar
        contador_tipos = {}
        objetos_cercanos = []
        
        for deteccion in detecciones:
            tipo_es = deteccion.clase_nombre_es
            
            if tipo_es not in contador_tipos:
                contador_tipos[tipo_es] = 0
            contador_tipos[tipo_es] += 1
            
            # Objetos cercanos (menos de 1.5 metros)
            if (deteccion.distancia_estimada is not None and 
                deteccion.distancia_estimada < 1.5):
                distancia_texto = f"{deteccion.distancia_estimada:.1f} metros"
                objetos_cercanos.append(f"{tipo_es} a {distancia_texto}")
        

        descripcion_partes = []
        
        # Mencionar objetos cercanos primero
        if objetos_cercanos:
            if len(objetos_cercanos) == 1:
                descripcion_partes.append(f"Objeto cercano: {objetos_cercanos[0]}")
            else:
                descripcion_partes.append(f"Objetos cercanos: {', '.join(objetos_cercanos)}")
        
        # Mencionar otros objetos detectados
        otros_objetos = []
        for tipo, cantidad in contador_tipos.items():
            if cantidad == 1:
                otros_objetos.append(tipo)
            else:
                otros_objetos.append(f"{cantidad} {tipo}s")
        
        if otros_objetos:
            if len(otros_objetos) <= 3:
                objetos_texto = ", ".join(otros_objetos)
            else:
                objetos_texto = f"{', '.join(otros_objetos[:3])} y {len(otros_objetos)-3} más"
            
            if not objetos_cercanos:  # objetos cercanos
                descripcion_partes.append(f"Detectado: {objetos_texto}")
        
        return ". ".join(descripcion_partes) if descripcion_partes else "Objetos detectados"
    
    def obtener_estadisticas(self) -> Dict[str, Any]:
    
        stats = dict(self.estadisticas)
        
        # Calcular FPS promedio
        if self.estadisticas['tiempo_promedio_inferencia'] > 0:
            stats['fps_promedio'] = 1.0 / self.estadisticas['tiempo_promedio_inferencia']
        else:
            stats['fps_promedio'] = 0.0
        
        # Información del modelo
        stats['modelo_info'] = {
            'cargado': self.modelo_cargado,
            'tipo': self.cargador_modelo.tipo_modelo if self.modelo_cargado else None,
            'num_clases': len(self.cargador_modelo.nombres_clases) if self.modelo_cargado else 0,
            'activo': self.activo
        }
        
        # Configuración actual
        stats['configuracion'] = {
            'confianza_minima': self.confianza_minima,
            'iou_threshold': self.iou_threshold,
            'max_detecciones': self.max_detecciones,
            'clases_prioritarias': len(self.clases_prioritarias)
        }
        
        return stats
    
    def cambiar_confianza(self, nueva_confianza: float):
        
        if 0.0 <= nueva_confianza <= 1.0:
            self.confianza_minima = nueva_confianza
            logger.info(f"Confianza mínima cambiada a: {nueva_confianza}")
        else:
            logger.warning(f"Valor de confianza inválido: {nueva_confianza}")
    
    def alternar_estado(self) -> bool:
        
        self.activo = not self.activo
        estado_texto = "activado" if self.activo else "desactivado"
        logger.info(f"Detector de objetos {estado_texto}")
        return self.activo
    
    def esta_activo(self) -> bool:
        
        return self.activo and self.modelo_cargado
    
    def reiniciar_estadisticas(self):
        self.estadisticas = {
            'detecciones_totales': 0,
            'detecciones_filtradas': 0,
            'tiempo_promedio_inferencia': 0.0,
            'ultimo_frame_procesado': 0.0,
            'objetos_detectados_sesion': {}
        }
        logger.info("Estadísticas reiniciadas")


class VisualizadorDetecciones:
    
    # Colores por tipo de detección (BGR)
    COLORES_TIPO = {
        TipoDeteccion.PERSONA: (0, 255, 0),        # Verde
        TipoDeteccion.VEHICULO: (255, 0, 0),       # Azul  
        TipoDeteccion.OBSTACULO: (0, 0, 255),      # Rojo
        TipoDeteccion.OBJETO_COTIDIANO: (255, 255, 0),  # Cian
        TipoDeteccion.ELECTRODOMESTICO: (255, 0, 255),  # Magenta
        TipoDeteccion.ANIMAL: (0, 255, 255),       # Amarillo
        TipoDeteccion.OTRO: (128, 128, 128)        # Gris
    }
    
    def __init__(self, config_interfaz: Dict[str, Any]):
    
        self.config = config_interfaz
        self.mostrar_confianza = config_interfaz.get('mostrar_confianza', True)
        self.mostrar_nombre_clases = config_interfaz.get('mostrar_nombre_clases', True)
        self.mostrar_coordenadas = config_interfaz.get('mostrar_coordenadas', False)
        
        # Configuración de fuente
        fuente_config = config_interfaz.get('fuente', {})
        self.fuente = fuente_config.get('tipo', cv2.FONT_HERSHEY_SIMPLEX)
        self.escala_fuente = fuente_config.get('escala', 0.6)
        self.grosor_fuente = fuente_config.get('grosor', 2)
        
    def dibujar_detecciones(self, imagen: np.ndarray, 
                        detecciones: List[Deteccion]) -> np.ndarray:
    
        imagen_resultado = imagen.copy()
        
        for deteccion in detecciones:
        
            color = self.COLORES_TIPO.get(deteccion.tipo_deteccion, (128, 128, 128))
            
        
            cv2.rectangle(
                imagen_resultado,
                (deteccion.x1, deteccion.y1),
                (deteccion.x2, deteccion.y2),
                color,
                2
            )
            
            # Preparar texto
            textos = []
            
            if self.mostrar_nombre_clases:
                textos.append(deteccion.clase_nombre_es)
            
            if self.mostrar_confianza:
                textos.append(f"{deteccion.confianza:.2f}")
            
            if deteccion.distancia_estimada is not None:
                textos.append(f"{deteccion.distancia_estimada:.1f}m")
            
            if self.mostrar_coordenadas:
                textos.append(f"({deteccion.centro_x},{deteccion.centro_y})")
            
            # Dibujar texto
            if textos:
                texto_completo = " | ".join(textos)
                
                # Calcular tamaño del texto
                (texto_ancho, texto_alto), _ = cv2.getTextSize(
                    texto_completo, self.fuente, self.escala_fuente, self.grosor_fuente
                )
                
                # Fondo para el texto
                cv2.rectangle(
                    imagen_resultado,
                    (deteccion.x1, deteccion.y1 - texto_alto - 10),
                    (deteccion.x1 + texto_ancho, deteccion.y1),
                    color,
                    -1
                )
                
                # Texto
                cv2.putText(
                    imagen_resultado,
                    texto_completo,
                    (deteccion.x1, deteccion.y1 - 5),
                    self.fuente,
                    self.escala_fuente,
                    (255, 255, 255),  # Blanco
                    self.grosor_fuente
                )
        
        return imagen_resultado
    
    def dibujar_estadisticas(self, imagen: np.ndarray, 
                        estadisticas: Dict[str, Any]) -> np.ndarray:
    
        imagen_resultado = imagen.copy()
        altura_img, ancho_img = imagen.shape[:2]
        
        # Preparar texto de estadísticas
        textos_stats = [
            f"FPS: {estadisticas.get('fps_promedio', 0):.1f}",
            f"Detecciones: {estadisticas.get('detecciones_totales', 0)}",
            f"Tiempo: {estadisticas.get('tiempo_promedio_inferencia', 0)*1000:.1f}ms"
        ]
        
        # Dibujar en esquina superior derecha
        y_inicial = 30
        for i, texto in enumerate(textos_stats):
            y_pos = y_inicial + (i * 25)
            
            # Calcular posición x para alinear a la derecha
            (texto_ancho, _), _ = cv2.getTextSize(
                texto, self.fuente, self.escala_fuente, self.grosor_fuente
            )
            x_pos = ancho_img - texto_ancho - 10
            
            # Fondo
            cv2.rectangle(
                imagen_resultado,
                (x_pos - 5, y_pos - 20),
                (x_pos + texto_ancho + 5, y_pos + 5),
                (0, 0, 0),
                -1
            )
            
            # Texto
            cv2.putText(
                imagen_resultado,
                texto,
                (x_pos, y_pos),
                self.fuente,
                self.escala_fuente,
                (255, 255, 255),
                self.grosor_fuente
            )
        
        return imagen_resultado



def crear_detector_objetos(config_manager=None) -> DetectorObjetos:

    return DetectorObjetos(config_manager)


def crear_visualizador(config_manager=None) -> VisualizadorDetecciones:

    config_manager = config_manager or obtener_configuracion()
    config_interfaz = config_manager.obtener('interfaz', {})
    return VisualizadorDetecciones(config_interfaz)


# Ejemplo de uso y pruebas
if __name__ == "__main__":
    print("Probando el detector de objetos YOLO de GafasIA...")
    
    try:
        # Verificar dependencias
        print(f"Ultralytics disponible: {ULTRALYTICS_DISPONIBLE}")
        print(f"PyTorch disponible: {TORCH_DISPONIBLE}")
        
        # Crear detector
        detector = crear_detector_objetos()
        
        # Cargar modelo
        if detector.cargar_modelo():
            print("Modelo cargado correctamente")
            
            # Mostrar estadísticas iniciales
            stats = detector.obtener_estadisticas()
            print(f"Información del modelo:")
            print(f"  Tipo: {stats['modelo_info']['tipo']}")
            print(f"  Clases: {stats['modelo_info']['num_clases']}")
            print(f"  Activo: {stats['modelo_info']['activo']}")
            
    
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        print("Procesando frame de prueba...")
                        
                        # Detectar objetos
                        detecciones = detector.detectar_objetos(frame)
                        
                        print(f"Detectados {len(detecciones)} objetos:")
                        for det in detecciones[:5]:  # Mostrar solo los primeros 5
                            distancia_texto = f"{det.distancia_estimada:.1f}m" if det.distancia_estimada else "N/A"
                            print(f"  - {det.clase_nombre_es}: {det.confianza:.2f} ({distancia_texto})")
                        
                        # Generar descripción
                        descripcion = detector.generar_descripcion_detecciones(detecciones)
                        print(f"Descripción: {descripcion}")
                        
                        # Crear visualizador y mostrar imagen (opcional)
                        visualizador = crear_visualizador()
                        imagen_con_detecciones = visualizador.dibujar_detecciones(frame, detecciones)
                        
                        # Mostrar estadísticas finales
                        stats_final = detector.obtener_estadisticas()
                        print(f"⚡ Rendimiento:")
                        print(f"  FPS promedio: {stats_final['fps_promedio']:.2f}")
                        print(f"  Tiempo inferencia: {stats_final['tiempo_promedio_inferencia']*1000:.1f}ms")
                        
                    cap.release()
                else:
                    print("No se pudo acceder a la cámara para prueba")
                    
            except Exception as e:
                print(f"Error en prueba con cámara: {e}")
                
        else:
            print("Error cargando modelo")
            
        # Probar funciones de utilidad
        print("\nProbando funciones de utilidad...")
        
        # Crear detección de prueba
        deteccion_prueba = Deteccion(
            clase_id=0,
            clase_nombre="person",
            clase_nombre_es="persona",
            confianza=0.85,
            x1=100, y1=100, x2=200, y2=300,
            centro_x=0, centro_y=0, ancho=0, alto=0, area=0,
            tipo_deteccion=TipoDeteccion.PERSONA,
            prioridad=1
        )
        
        print(f"Detección de prueba creada: {deteccion_prueba.clase_nombre_es}")
        print(f"  Centro: {deteccion_prueba.obtener_centro()}")
        print(f"  Área: {deteccion_prueba.area} píxeles")
        
        # Probar clasificador
        clasificador = ClasificadorObjetos({'person': 'persona', 'car': 'automóvil'})
        tipo, prioridad = clasificador.clasificar_objeto('person')
        traduccion = clasificador.traducir_nombre('person')
        
        print(f"Clasificador probado:")
        print(f"  Tipo: {tipo}")
        print(f"  Prioridad: {prioridad}")
        print(f"  Traducción: {traduccion}")
        
    except Exception as e:
        print(f"Error en prueba: {e}")
    
    print("\nPrueba del detector de objetos completada")
    