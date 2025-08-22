import cv2
import numpy as np
import face_recognition
import time
from typing import List, Dict, Any, Optional, Tuple
import logging
import threading
import queue

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from .detector_personas import DetectorObjetos
from ..utilidades.config import ConfigManager
from ..utilidades.logger import setup_logger

class DetectorPersonas:
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        self.config_manager = config_manager or ConfigManager()
        self.logger = setup_logger(__name__)
        
        self.configurar_detector()
        
        self.detector_base = DetectorObjetos(config_manager)
        
        self._inicializar_mediapipe()
        self._inicializar_reconocimiento_facial()
        
        # Seguimiento de personas
        self.personas_conocidas = {}
        self.siguiente_id = 1
        self.historial_detecciones = []
        self.max_historial = 10
        
        # Estadísticas
        self.stats = {
            'total_personas_detectadas': 0,
            'personas_reconocidas': 0,
            'tiempo_promedio_procesamiento': 0.0,
            'detecciones_por_minuto': 0
        }
        
        self.logger.info("Detector de personas inicializado correctamente")
    
    def configurar_detector(self):
        """Configura parámetros específicos para detección de personas"""
        try:
            # Configuración general
            config = self.config_manager.get_config('deteccion.personas', {})
            
            self.habilitado = config.get('habilitado', True)
            self.detectar_caras = config.get('detectar_caras', True)
            self.analizar_pose = config.get('analizar_pose', True)
            self.seguimiento = config.get('seguimiento', True)
            
            # Configuración de reconocimiento facial
            facial_config = config.get('reconocimiento_facial', {})
            self.tolerancia_facial = facial_config.get('tolerancia', 0.6)
            self.modelo_facial = facial_config.get('modelo', 'large')  # small, large
            self.max_caras_por_frame = facial_config.get('max_caras_por_frame', 10)
            
            # Configuración de pose
            pose_config = config.get('estimacion_pose', {})
            self.confianza_pose = pose_config.get('confianza_minima', 0.5)
            self.detectar_manos = pose_config.get('detectar_manos', False)
            self.detectar_cuerpo = pose_config.get('detectar_cuerpo', True)
            
            # Configuración de seguimiento
            seguimiento_config = config.get('seguimiento', {})
            self.max_distancia_seguimiento = seguimiento_config.get('max_distancia', 100)
            self.frames_perdida_max = seguimiento_config.get('frames_perdida_max', 5)
            
            # Configuración de análisis
            analisis_config = config.get('analisis', {})
            self.detectar_actividad = analisis_config.get('detectar_actividad', True)
            self.analizar_direccion = analisis_config.get('analizar_direccion', True)
            self.estimar_edad_genero = analisis_config.get('estimar_edad_genero', False)
            
            self.logger.info(f"Detector de personas configurado - Caras: {self.detectar_caras}, "
                        f"Pose: {self.analizar_pose}, Seguimiento: {self.seguimiento}")
            
        except Exception as e:
            self.logger.error(f"Error limpiando personas perdidas: {e}")
    
    def _actualizar_estadisticas(self, personas: List[Dict[str, Any]], tiempo_procesamiento: float):
    
        try:
            self.stats['total_personas_detectadas'] += len(personas)
            
            # Contar personas reconocidas
            personas_reconocidas = 0
            for persona in personas:
                if 'facial' in persona.get('analisis', {}):
                    facial_info = persona['analisis']['facial']
                    for reconocida in facial_info.get('personas_reconocidas', []):
                        if reconocida.get('nombre') not in ['Desconocido', 'Sin referencia']:
                            personas_reconocidas += 1
            
            self.stats['personas_reconocidas'] += personas_reconocidas
            
            # Actualizar tiempo promedio
            if self.stats['tiempo_promedio_procesamiento'] == 0:
                self.stats['tiempo_promedio_procesamiento'] = tiempo_procesamiento
            else:
                self.stats['tiempo_promedio_procesamiento'] = (
                    self.stats['tiempo_promedio_procesamiento'] * 0.9 + 
                    tiempo_procesamiento * 0.1
                )
            
        except Exception as e:
            self.logger.error(f"Error actualizando estadísticas: {e}")
    
    def generar_descripcion_personas(self, personas: List[Dict[str, Any]]) -> str:
    
        if not personas:
            return ""
        
        try:
            descripcion_partes = []
            
            # Contar personas
            num_personas = len(personas)
            if num_personas == 1:
                descripcion_partes.append("1 persona")
            else:
                descripcion_partes.append(f"{num_personas} personas")
            
            # Información específica de personas reconocidas
            personas_reconocidas = []
            personas_cerca = 0
            actividades = []
            
            for persona in personas:
                # Verificar proximidad
                distancia = persona.get('distancia', 100)
                if distancia < 3:  # Menos de 3 metros
                    personas_cerca += 1
                
                # Información facial
                if 'facial' in persona.get('analisis', {}):
                    facial_info = persona['analisis']['facial']
                    for reconocida in facial_info.get('personas_reconocidas', []):
                        nombre = reconocida.get('nombre', '')
                        if nombre not in ['Desconocido', 'Sin referencia'] and reconocida.get('confianza', 0) > 0.7:
                            personas_reconocidas.append(nombre)
                
                # Información de actividad
                if 'actividad' in persona.get('analisis', {}):
                    actividad_info = persona['analisis']['actividad']
                    actividad = actividad_info.get('actividad_principal', '')
                    if actividad and actividad not in ['desconocida', 'de_pie']:
                        actividades.append(actividad)
            
            # Construir descripción
            if personas_cerca > 0:
                if personas_cerca == 1:
                    descripcion_partes.append("una cerca")
                else:
                    descripcion_partes.append(f"{personas_cerca} cerca")
            
            # Añadir personas reconocidas
            if personas_reconocidas:
                nombres_unicos = list(set(personas_reconocidas))
                if len(nombres_unicos) == 1:
                    descripcion_partes.append(f"reconocido: {nombres_unicos[0]}")
                else:
                    descripcion_partes.append(f"reconocidos: {', '.join(nombres_unicos[:2])}")
            
            if actividades:
                actividades_unicas = list(set(actividades))
                if actividades_unicas:
                    actividad_texto = actividades_unicas[0].replace('_', ' ')
                    descripcion_partes.append(f"actividad: {actividad_texto}")
            
            return " - ".join(descripcion_partes)
            
        except Exception as e:
            self.logger.error(f"Error generando descripción: {e}")
            return f"{len(personas)} personas detectadas"
    
    def dibujar_analisis_personas(self, frame: np.ndarray, personas: List[Dict[str, Any]]) -> np.ndarray:
    
        if not personas:
            return frame
        
        try:
            frame_resultado = frame.copy()
            
            for persona in personas:
                # Dibujar bbox principal
                bbox = persona['bbox']
                x, y, w, h = bbox
                
                # Color basado en seguimiento
                id_seguimiento = persona.get('id_seguimiento', 0)
                colores = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                color = colores[id_seguimiento % len(colores)]
                
                cv2.rectangle(frame_resultado, (x, y), (x + w, y + h), color, 2)
                
                # Información principal
                info_principal = []
                confianza = persona.get('confianza', 0)
                info_principal.append(f"Persona {id_seguimiento} ({confianza:.2f})")
                
                if 'distancia' in persona:
                    info_principal.append(f"Dist: {persona['distancia']:.1f}m")
                
                # Dibujar información principal
                y_texto = y - 10
                for i, texto in enumerate(info_principal):
                    cv2.putText(frame_resultado, texto, (x, y_texto - i * 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Dibujar análisis facial
                self._dibujar_analisis_facial(frame_resultado, persona)
                
                # Dibujar análisis de pose
                self._dibujar_analisis_pose(frame_resultado, persona)
                
                # Dibujar información de actividad
                self._dibujar_informacion_actividad(frame_resultado, persona)
            
            # Dibujar estadísticas generales
            self._dibujar_estadisticas_generales(frame_resultado)
            
            return frame_resultado
            
        except Exception as e:
            self.logger.error(f"Error dibujando análisis: {e}")
            return frame
    
    def _dibujar_analisis_facial(self, frame: np.ndarray, persona: Dict[str, Any]):
        if 'facial' not in persona.get('analisis', {}):
            return
        
        try:
            facial_info = persona['analisis']['facial']
            
            # Dibujar caras detectadas
            for coords in facial_info.get('coordenadas_caras', []):
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Dibujar nombres reconocidos
            bbox = persona['bbox']
            y_nombre = bbox[1] + bbox[3] + 20
            
            for i, reconocida in enumerate(facial_info.get('personas_reconocidas', [])):
                nombre = reconocida.get('nombre', 'Desconocido')
                confianza = reconocida.get('confianza', 0)
                
                if nombre != 'Sin referencia':
                    color_texto = (0, 255, 0) if nombre != 'Desconocido' else (0, 165, 255)
                    texto = f"{nombre} ({confianza:.2f})"
                    cv2.putText(frame, texto, (bbox[0], y_nombre + i * 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_texto, 1)
        
        except Exception as e:
            self.logger.error(f"Error dibujando análisis facial: {e}")
    
    def _dibujar_analisis_pose(self, frame: np.ndarray, persona: Dict[str, Any]):
        if 'pose' not in persona.get('analisis', {}) or not MEDIAPIPE_AVAILABLE:
            return
        
        try:
            pose_info = persona['analisis']['pose']
            
            if not pose_info.get('pose_detectada'):
                return
            
            bbox = persona['bbox']
            landmarks = pose_info.get('landmarks', [])
            
            if len(landmarks) >= 33:  # MediaPipe pose landmarks
                # Dibujar algunos landmarks clave
                altura_frame, ancho_frame = frame.shape[:2]
                
                # Convertir coordenadas normalizadas a píxeles
                for i, landmark in enumerate(landmarks):
                    if i in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:  # Puntos clave
                        x = int(landmark['x'] * bbox[2] + bbox[0])
                        y = int(landmark['y'] * bbox[3] + bbox[1])
                        
                        if 0 <= x < ancho_frame and 0 <= y < altura_frame:
                            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)
                
                # Mostrar actividad estimada
                actividad = pose_info.get('actividad_estimada', 'desconocida')
                if actividad != 'desconocida':
                    cv2.putText(frame, f"Act: {actividad.replace('_', ' ')}", 
                            (bbox[0], bbox[1] + bbox[3] + 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        except Exception as e:
            self.logger.error(f"Error dibujando análisis de pose: {e}")
    
    def _dibujar_informacion_actividad(self, frame: np.ndarray, persona: Dict[str, Any]):
        
        if 'actividad' not in persona.get('analisis', {}):
            return
        
        try:
            bbox = persona['bbox']
            actividad_info = persona['analisis']['actividad']
            
            # Información de movimiento
            movimiento = actividad_info.get('movimiento', 'estatico')
            velocidad = actividad_info.get('velocidad_estimada', 0)
            
            if movimiento != 'estatico':
                texto_movimiento = f"Mov: {movimiento} ({velocidad:.1f})"
                cv2.putText(frame, texto_movimiento, 
                        (bbox[0], bbox[1] + bbox[3] + 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Información de dirección
            if 'direccion' in persona.get('analisis', {}):
                direccion_info = persona['analisis']['direccion']
                orientacion = direccion_info.get('orientacion_cuerpo', 'frontal')
                
                if orientacion != 'frontal':
                    cv2.putText(frame, f"Dir: {orientacion}", 
                            (bbox[0], bbox[1] + bbox[3] + 100), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 128, 0), 1)
    
        except Exception as e:
            self.logger.error(f"Error dibujando información de actividad: {e}")
    
    def _dibujar_estadisticas_generales(self, frame: np.ndarray):
        try:
            altura_frame = frame.shape[0]
            
            # Estadísticas básicas
            stats_texto = [
                f"Personas detectadas: {self.stats['total_personas_detectadas']}",
                f"Reconocidas: {self.stats['personas_reconocidas']}",
                f"Tiempo proc: {self.stats['tiempo_promedio_procesamiento']:.3f}s",
                f"Seguimiento activo: {len(self.personas_conocidas)} personas"
            ]
            
            for i, texto in enumerate(stats_texto):
                cv2.putText(frame, texto, (10, altura_frame - 80 + i * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        except Exception as e:
            self.logger.error(f"Error dibujando estadísticas: {e}")
    
    def agregar_cara_conocida(self, nombre: str, imagen: np.ndarray) -> bool:

        try:
            if len(imagen.shape) == 3:
                rgb_imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                rgb_imagen = imagen
            
            # Obtener encoding de la cara
            encodings = face_recognition.face_encodings(rgb_imagen)
            
            if encodings:
                self.caras_conocidas_encodings.append(encodings[0])
                self.caras_conocidas_nombres.append(nombre)
                self.logger.info(f"Nueva cara conocida agregada: {nombre}")
                return True
            else:
                self.logger.warning(f"No se encontró cara en la imagen para: {nombre}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error agregando cara conocida: {e}")
            return False
    
    def get_estadisticas(self) -> Dict[str, Any]:
    
        stats_completas = self.stats.copy()
        stats_completas.update({
            'caras_conocidas': len(self.caras_conocidas_nombres),
            'personas_en_seguimiento': len(self.personas_conocidas),
            'componentes_activos': {
                'deteccion_facial': self.detectar_caras,
                'analisis_pose': self.analizar_pose and MEDIAPIPE_AVAILABLE,
                'seguimiento': self.seguimiento,
                'mediapipe_disponible': MEDIAPIPE_AVAILABLE
            }
        })
        return stats_completas
    
    def reiniciar_seguimiento(self):
    
        try:
            self.personas_conocidas.clear()
            self.historial_detecciones.clear()
            self.siguiente_id = 1
            self.logger.info("Sistema de seguimiento reiniciado")
        except Exception as e:
            self.logger.error(f"Error reiniciando seguimiento: {e}")
    
    def detener(self):
    
        try:
            # Liberar recursos de MediaPipe
            if hasattr(self, 'pose_detector'):
                self.pose_detector.close()
            
            if hasattr(self, 'hands_detector'):
                self.hands_detector.close()
            
            # Limpiar datos
            self.personas_conocidas.clear()
            self.historial_detecciones.clear()
            
            self.logger.info("Detector de personas detenido")
            
        except Exception as e:
            self.logger.error(f"Error deteniendo detector: {e}")
    
    def __del__(self):
        try:
            self.detener()
        except Exception as e:   
            print(f"Error configurando detector de personas: {e}")
    # Valores por defecto
        self.habilitado = True
        self.detectar_caras = True
        self.analizar_pose = True
    
    def _inicializar_mediapipe(self):
    
        self.mp_pose = None
        self.mp_hands = None
        self.mp_face = None
        self.mp_drawing = None
        
        if not MEDIAPIPE_AVAILABLE or not self.analizar_pose:
            self.logger.warning("MediaPipe no disponible o deshabilitado")
            return
        
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_hands = mp.solutions.hands
            self.mp_face = mp.solutions.face_detection
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Inicializar detectores
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=self.confianza_pose,
                min_tracking_confidence=0.5
            )
            
            if self.detectar_manos:
                self.hands_detector = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=4,  # Máximo 4 manos (2 personas)
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
            
            self.logger.info("MediaPipe inicializado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando MediaPipe: {e}")
            self.analizar_pose = False
    
    def _inicializar_reconocimiento_facial(self):
        
        self.caras_conocidas_encodings = []
        self.caras_conocidas_nombres = []
        
        if not self.detectar_caras:
            return
        
        try:
            # Cargar caras conocidas desde configuración
            caras_config = self.config_manager.get_config('deteccion.personas.caras_conocidas', {})
            
            for nombre, datos in caras_config.items():
                ruta_imagen = datos.get('imagen')
                if ruta_imagen:
                    self._cargar_cara_conocida(nombre, ruta_imagen)
            
            self.logger.info(f"Reconocimiento facial inicializado - {len(self.caras_conocidas_nombres)} caras conocidas")
            
        except Exception as e:
            self.logger.error(f"Error inicializando reconocimiento facial: {e}")
    
    def _cargar_cara_conocida(self, nombre: str, ruta_imagen: str):
    
        try:
            imagen = face_recognition.load_image_file(ruta_imagen)
            encodings = face_recognition.face_encodings(imagen)
            
            if encodings:
                self.caras_conocidas_encodings.append(encodings[0])
                self.caras_conocidas_nombres.append(nombre)
                self.logger.info(f"Cara conocida cargada: {nombre}")
            else:
                self.logger.warning(f"No se encontró cara en imagen: {ruta_imagen}")
                
        except Exception as e:
            self.logger.error(f"Error cargando cara {nombre}: {e}")
    
    def detectar_personas(self, frame: np.ndarray) -> List[Dict[str, Any]]:
    
        if not self.habilitado or frame is None:
            return []
        
        inicio_tiempo = time.time()
        
        try:
            # Detección base de personas usando YOLO
            detecciones_base = self.detector_base.detectar_objetos(frame)
            personas_yolo = [det for det in detecciones_base if det.get('tipo') == 'persona']
            
            if not personas_yolo:
                return []
            
            #  Análisis especializado de cada persona
            personas_analizadas = []
            
            for i, persona in enumerate(personas_yolo):
                persona_analizada = self._analizar_persona(frame, persona, i)
                if persona_analizada:
                    personas_analizadas.append(persona_analizada)
            
            # Seguimiento de personas
            if self.seguimiento:
                personas_analizadas = self._aplicar_seguimiento(personas_analizadas)
            
            # Actualizar estadísticas
            self._actualizar_estadisticas(personas_analizadas, time.time() - inicio_tiempo)
            
            return personas_analizadas
            
        except Exception as e:
            self.logger.error(f"Error detectando personas: {e}")
            return []
    
    def _analizar_persona(self, frame: np.ndarray, deteccion: Dict[str, Any], indice: int) -> Optional[Dict[str, Any]]:
    
        try:
            # Extraer región de la persona
            x, y, w, h = deteccion['bbox']
            roi_persona = frame[y:y+h, x:x+w]
            
            if roi_persona.size == 0:
                return None
            
            # Estructura base del análisis
            persona = {
                **deteccion,  # Incluir datos base de YOLO
                'id_temporal': f"persona_{indice}",
                'timestamp': time.time(),
                'analisis': {}
            }
            
            # Análisis facial
            if self.detectar_caras:
                analisis_facial = self._analizar_cara(roi_persona, (x, y))
                persona['analisis']['facial'] = analisis_facial
            
            # Análisis de pose
            if self.analizar_pose and MEDIAPIPE_AVAILABLE:
                analisis_pose = self._analizar_pose(roi_persona)
                persona['analisis']['pose'] = analisis_pose
            
            # Análisis de actividad
            if self.detectar_actividad:
                actividad = self._detectar_actividad(persona)
                persona['analisis']['actividad'] = actividad
            
            # Análisis de dirección
            if self.analizar_direccion:
                direccion = self._analizar_direccion(persona)
                persona['analisis']['direccion'] = direccion
            
            return persona
            
        except Exception as e:
            self.logger.error(f"Error analizando persona: {e}")
            return None
    
    def _analizar_cara(self, roi_persona: np.ndarray, offset: Tuple[int, int]) -> Dict[str, Any]:
    
        resultado = {
            'caras_detectadas': 0,
            'personas_reconocidas': [],
            'coordenadas_caras': [],
            'confianzas': []
        }
        
        try:
            # Convertir a RGB para face_recognition
            rgb_roi = cv2.cvtColor(roi_persona, cv2.COLOR_BGR2RGB)
            
            # Detectar caras
            ubicaciones_caras = face_recognition.face_locations(rgb_roi, model=self.modelo_facial)
            
            if not ubicaciones_caras:
                return resultado
            
            resultado['caras_detectadas'] = len(ubicaciones_caras)
            
            # Limitar número de caras a procesar
            if len(ubicaciones_caras) > self.max_caras_por_frame:
                ubicaciones_caras = ubicaciones_caras[:self.max_caras_por_frame]
            
            # Obtener encodings de caras
            encodings_caras = face_recognition.face_encodings(rgb_roi, ubicaciones_caras)
            
            # Reconocimiento de caras conocidas
            for i, encoding in enumerate(encodings_caras):
                top, right, bottom, left = ubicaciones_caras[i]
                
                # Ajustar coordenadas al frame completo
                coords_ajustadas = (
                    left + offset[0], 
                    top + offset[1], 
                    right + offset[0], 
                    bottom + offset[1]
                )
                resultado['coordenadas_caras'].append(coords_ajustadas)
                
                # Comparar con caras conocidas
                if self.caras_conocidas_encodings:
                    coincidencias = face_recognition.compare_faces(
                        self.caras_conocidas_encodings, 
                        encoding, 
                        tolerance=self.tolerancia_facial
                    )
                    
                    distancias = face_recognition.face_distance(
                        self.caras_conocidas_encodings, 
                        encoding
                    )
                    
                    mejor_coincidencia = np.argmin(distancias)
                    
                    if coincidencias[mejor_coincidencia]:
                        nombre = self.caras_conocidas_nombres[mejor_coincidencia]
                        confianza = 1.0 - distancias[mejor_coincidencia]
                        
                        resultado['personas_reconocidas'].append({
                            'nombre': nombre,
                            'confianza': float(confianza),
                            'coordenadas': coords_ajustadas
                        })
                        resultado['confianzas'].append(float(confianza))
                    else:
                        resultado['personas_reconocidas'].append({
                            'nombre': 'Desconocido',
                            'confianza': 0.0,
                            'coordenadas': coords_ajustadas
                        })
                        resultado['confianzas'].append(0.0)
                else:
                    resultado['personas_reconocidas'].append({
                        'nombre': 'Sin referencia',
                        'confianza': 0.0,
                        'coordenadas': coords_ajustadas
                    })
                    resultado['confianzas'].append(0.0)
            
        except Exception as e:
            self.logger.error(f"Error en análisis facial: {e}")
        
        return resultado
    
    def _analizar_pose(self, roi_persona: np.ndarray) -> Dict[str, Any]:
    
        resultado = {
            'pose_detectada': False,
            'landmarks': [],
            'confianza_pose': 0.0,
            'actividad_estimada': 'desconocida',
            'manos': []
        }
        
        if not self.analizar_pose or not MEDIAPIPE_AVAILABLE:
            return resultado
        
        try:
            # Convertir a RGB
            rgb_roi = cv2.cvtColor(roi_persona, cv2.COLOR_BGR2RGB)
            
            # Detectar pose
            resultados_pose = self.pose_detector.process(rgb_roi)
            
            if resultados_pose.pose_landmarks:
                resultado['pose_detectada'] = True
                
                # Extraer landmarks
                landmarks = []
                for landmark in resultados_pose.pose_landmarks.landmark:
                    landmarks.append({
                        'x': landmark.x,
                        'y': landmark.y,
                        'z': landmark.z,
                        'visibility': landmark.visibility
                    })
                
                resultado['landmarks'] = landmarks
                
                # Calcular confianza promedio
                visibilidades = [lm['visibility'] for lm in landmarks]
                resultado['confianza_pose'] = float(np.mean(visibilidades))
                
                # Estimar actividad basada en pose
                resultado['actividad_estimada'] = self._estimar_actividad_desde_pose(landmarks)
            
            # Detectar manos 
            if self.detectar_manos and hasattr(self, 'hands_detector'):
                resultados_manos = self.hands_detector.process(rgb_roi)
                
                if resultados_manos.multi_hand_landmarks:
                    for hand_landmarks in resultados_manos.multi_hand_landmarks:
                        mano = []
                        for landmark in hand_landmarks.landmark:
                            mano.append({
                                'x': landmark.x,
                                'y': landmark.y,
                                'z': landmark.z
                            })
                        resultado['manos'].append(mano)
            
        except Exception as e:
            self.logger.error(f"Error en análisis de pose: {e}")
        
        return resultado
    
    def _estimar_actividad_desde_pose(self, landmarks: List[Dict]) -> str:
    
        try:
            if len(landmarks) < 33:  # MediaPose tiene 33 landmarks
                return 'pose_incompleta'
            
            # Puntos clave para análisis
            nariz = landmarks[0]
            hombro_izq = landmarks[11]
            hombro_der = landmarks[12]
            cadera_izq = landmarks[23]
            cadera_der = landmarks[24]
            rodilla_izq = landmarks[25]
            rodilla_der = landmarks[26]
            
            # Calcular ángulos y posiciones relativas
            # Ángulo de inclinación del torso
            centro_hombros_y = (hombro_izq['y'] + hombro_der['y']) / 2
            centro_caderas_y = (cadera_izq['y'] + cadera_der['y']) / 2
            
            # Diferencia de altura entre cabeza y cuerpo
            altura_relativa = nariz['y'] - centro_caderas_y
            
            # Posición de rodillas relativa a caderas
            altura_rodillas = (rodilla_izq['y'] + rodilla_der['y']) / 2 - centro_caderas_y
            
            # Clasificación simple de actividades
            if altura_relativa > -0.3:  # Cabeza muy cerca del cuerpo
                return 'agachado'
            elif altura_rodillas > 0.1:  # Rodillas muy altas
                return 'sentado'
            elif abs(centro_hombros_y - centro_caderas_y) < 0.2:  # Torso muy horizontal
                return 'acostado'
            else:
                return 'de_pie'
                
        except Exception as e:
            self.logger.error(f"Error estimando actividad: {e}")
            return 'desconocida'
    
    def _detectar_actividad(self, persona: Dict[str, Any]) -> Dict[str, Any]:
    
        actividad = {
            'movimiento': 'estatico',
            'velocidad_estimada': 0.0,
            'direccion_movimiento': 'ninguna',
            'actividad_principal': 'desconocida'
        }
        
        try:
            # Obtener actividad desde pose 
            if 'pose' in persona.get('analisis', {}):
                pose_info = persona['analisis']['pose']
                if pose_info.get('actividad_estimada'):
                    actividad['actividad_principal'] = pose_info['actividad_estimada']
            
            # Análisis de movimiento basado en historial 
            if hasattr(self, 'personas_conocidas') and persona.get('id_seguimiento'):
                id_seguimiento = persona['id_seguimiento']
                if id_seguimiento in self.personas_conocidas:
                    persona_anterior = self.personas_conocidas[id_seguimiento]
                    
                    # Calcular movimiento
                    bbox_actual = persona['bbox']
                    bbox_anterior = persona_anterior.get('bbox')
                    
                    if bbox_anterior:
                        # Calcular desplazamiento
                        dx = (bbox_actual[0] + bbox_actual[2]/2) - (bbox_anterior[0] + bbox_anterior[2]/2)
                        dy = (bbox_actual[1] + bbox_actual[3]/2) - (bbox_anterior[1] + bbox_anterior[3]/2)
                        
                        distancia = np.sqrt(dx**2 + dy**2)
                        tiempo_transcurrido = persona['timestamp'] - persona_anterior.get('timestamp', persona['timestamp'])
                        
                        if tiempo_transcurrido > 0:
                            velocidad = distancia / tiempo_transcurrido
                            actividad['velocidad_estimada'] = float(velocidad)
                            
                            # Clasificar movimiento
                            if velocidad > 50:  # pixels/segundo
                                actividad['movimiento'] = 'rapido'
                            elif velocidad > 10:
                                actividad['movimiento'] = 'lento'
                            else:
                                actividad['movimiento'] = 'estatico'
                            
                            # Dirección de movimiento
                            if abs(dx) > abs(dy):
                                actividad['direccion_movimiento'] = 'derecha' if dx > 0 else 'izquierda'
                            else:
                                actividad['direccion_movimiento'] = 'abajo' if dy > 0 else 'arriba'
            
        except Exception as e:
            self.logger.error(f"Error detectando actividad: {e}")
        
        return actividad
    
    def _analizar_direccion(self, persona: Dict[str, Any]) -> Dict[str, Any]:
        
        direccion = {
            'mirando_hacia': 'desconocido',
            'orientacion_cuerpo': 'frontal',
            'confianza_direccion': 0.0
        }
        
        try:
            # Usar información facial 
            if 'facial' in persona.get('analisis', {}):
                facial_info = persona['analisis']['facial']
                
                # Si hay caras detectadas, usar su posición para estimar dirección
                if facial_info.get('coordenadas_caras'):
                    bbox_persona = persona['bbox']
                    cara_coords = facial_info['coordenadas_caras'][0]  # Primera cara
                    
                    # Posición relativa de la cara en el bbox de la persona
                    centro_persona_x = bbox_persona[0] + bbox_persona[2] / 2
                    centro_cara_x = cara_coords[0] + (cara_coords[2] - cara_coords[0]) / 2
                    
                    offset_x = centro_cara_x - centro_persona_x
                    
                    # Determinar orientación basada en offset
                    if abs(offset_x) < bbox_persona[2] * 0.1:  # Cara centrada
                        direccion['orientacion_cuerpo'] = 'frontal'
                        direccion['mirando_hacia'] = 'camara'
                    elif offset_x > 0:
                        direccion['orientacion_cuerpo'] = 'perfil_derecho'
                        direccion['mirando_hacia'] = 'derecha'
                    else:
                        direccion['orientacion_cuerpo'] = 'perfil_izquierdo'
                        direccion['mirando_hacia'] = 'izquierda'
                    
                    direccion['confianza_direccion'] = 0.7
            
            # Usar información de pose si está disponible
            if 'pose' in persona.get('analisis', {}) and persona['analisis']['pose'].get('landmarks'):
                landmarks = persona['analisis']['pose']['landmarks']
                
                if len(landmarks) >= 12:  # Necesitamos hombros
                    hombro_izq = landmarks[11]
                    hombro_der = landmarks[12]
                    
                    # Calcular orientación basada en hombros
                    diff_x = hombro_der['x'] - hombro_izq['x']
                    
                    if abs(diff_x) < 0.05:  # Hombros alineados
                        direccion['orientacion_cuerpo'] = 'frontal'
                    elif diff_x > 0:
                        direccion['orientacion_cuerpo'] = 'perfil_izquierdo'
                    else:
                        direccion['orientacion_cuerpo'] = 'perfil_derecho'
                    
                    direccion['confianza_direccion'] = max(direccion['confianza_direccion'], 0.8)
            
        except Exception as e:
            self.logger.error(f"Error analizando dirección: {e}")
        
        return direccion
    
    def _aplicar_seguimiento(self, personas_actuales: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    
        if not self.seguimiento:
            return personas_actuales
        
        try:
            # Actualizar historial
            self.historial_detecciones.append(personas_actuales)
            if len(self.historial_detecciones) > self.max_historial:
                self.historial_detecciones.pop(0)
            
            # Asignar IDs de seguimiento
            for persona in personas_actuales:
                mejor_id = self._encontrar_mejor_coincidencia(persona)
                
                if mejor_id is not None:
                    persona['id_seguimiento'] = mejor_id
                    self.personas_conocidas[mejor_id] = persona
                else:
                    # Nueva persona
                    persona['id_seguimiento'] = self.siguiente_id
                    self.personas_conocidas[self.siguiente_id] = persona
                    self.siguiente_id += 1
            
            # Limpiar personas perdidas
            self._limpiar_personas_perdidas(personas_actuales)
            
            return personas_actuales
            
        except Exception as e:
            self.logger.error(f"Error en seguimiento: {e}")
            return personas_actuales
    
    def _encontrar_mejor_coincidencia(self, persona_actual: Dict[str, Any]) -> Optional[int]:
    
        try:
            bbox_actual = persona_actual['bbox']
            centro_actual = (
                bbox_actual[0] + bbox_actual[2] / 2,
                bbox_actual[1] + bbox_actual[3] / 2
            )
            
            mejor_distancia = float('inf')
            mejor_id = None
            
            for id_persona, persona_conocida in self.personas_conocidas.items():
                bbox_conocida = persona_conocida['bbox']
                centro_conocido = (
                    bbox_conocida[0] + bbox_conocida[2] / 2,
                    bbox_conocida[1] + bbox_conocida[3] / 2
                )
                
                # Calcular distancia euclidiana
                distancia = np.sqrt(
                    (centro_actual[0] - centro_conocido[0])**2 + 
                    (centro_actual[1] - centro_conocido[1])**2
                )
                
                if distancia < mejor_distancia and distancia < self.max_distancia_seguimiento:
                    mejor_distancia = distancia
                    mejor_id = id_persona
            
            return mejor_id
            
        except Exception as e:
            self.logger.error(f"Error encontrando coincidencia: {e}")
            return None
    
    def _limpiar_personas_perdidas(self, personas_actuales: List[Dict[str, Any]]):
    
        try:
            ids_actuales = {p.get('id_seguimiento') for p in personas_actuales}
            ids_a_eliminar = []
            
            for id_persona, persona in self.personas_conocidas.items():
                if id_persona not in ids_actuales:
                    # Verificar si ha estado perdida por mucho tiempo
                    tiempo_actual = time.time()
                    tiempo_ultima_deteccion = persona.get('timestamp', tiempo_actual)
                    
                    if tiempo_actual - tiempo_ultima_deteccion > self.frames_perdida_max:
                        ids_a_eliminar.append(id_persona)
            
            # Eliminar personas perdidas
            for id_persona in ids_a_eliminar:
                del self.personas_conocidas[id_persona]
                
        except Exception as e:
            self.logger.error(f"Error configurando detector de personas: {e}")