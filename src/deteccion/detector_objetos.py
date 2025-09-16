# src/deteccion/detector_objetos.py
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict, Tuple
import time

class DetectorObjetos:
    def __init__(self, modelo_path: str = 'yolov8n.pt', confianza_minima: float = 0.5):
        """
        Inicializa el detector de objetos con YOLOv8
        
        Args:
            modelo_path: Ruta al modelo YOLO (usa 'yolov8n.pt' para descarga automática)
            confianza_minima: Umbral mínimo de confianza para detecciones
        """
        print("🔄 Inicializando detector de objetos...")
        
        # Configuración del dispositivo (CPU para Raspberry Pi)
        self.dispositivo = 'cpu'  # Cambiar a 'cuda' si tienes GPU
        
        # Cargar modelo YOLO
        self.modelo = YOLO(modelo_path)
        self.confianza_minima = confianza_minima
        
        # Diccionario de etiquetas en español (COCO dataset)
        self.etiquetas_es = {
            0: 'persona', 1: 'bicicleta', 2: 'automóvil', 3: 'motocicleta', 4: 'avión',
            5: 'autobús', 6: 'tren', 7: 'camión', 8: 'barco', 9: 'semáforo',
            10: 'boca de incendios', 11: 'señal de alto', 12: 'parquímetro', 13: 'banco', 14: 'pájaro',
            15: 'gato', 16: 'perro', 17: 'caballo', 18: 'oveja', 19: 'vaca',
            20: 'elefante', 21: 'oso', 22: 'cebra', 23: 'jirafa', 24: 'mochila',
            25: 'paraguas', 26: 'bolso', 27: 'corbata', 28: 'maleta', 29: 'frisbee',
            30: 'esquís', 31: 'tabla de snowboard', 32: 'pelota deportiva', 33: 'cometa', 34: 'bate de béisbol',
            35: 'guante de béisbol', 36: 'patineta', 37: 'tabla de surf', 38: 'raqueta de tenis', 39: 'botella',
            40: 'copa de vino', 41: 'taza', 42: 'tenedor', 43: 'cuchillo', 44: 'cuchara',
            45: 'tazón', 46: 'plátano', 47: 'manzana', 48: 'sándwich', 49: 'naranja',
            50: 'brócoli', 51: 'zanahoria', 52: 'perro caliente', 53: 'pizza', 54: 'dona',
            55: 'pastel', 56: 'silla', 57: 'sofá', 58: 'planta en maceta', 59: 'cama',
            60: 'mesa de comedor', 61: 'inodoro', 62: 'televisor', 63: 'computadora portátil', 64: 'ratón de computadora',
            65: 'control remoto', 66: 'teclado', 67: 'teléfono celular', 68: 'microondas', 69: 'horno',
            70: 'tostadora', 71: 'fregadero', 72: 'refrigerador', 73: 'libro', 74: 'reloj',
            75: 'florero', 76: 'tijeras', 77: 'osito de peluche', 78: 'secador de cabello', 79: 'cepillo de dientes'
        }
        
        # Objetos prioritarios para casa (más útiles para personas con discapacidad visual)
        self.objetos_prioritarios = {
            56: 'silla',           # Importante para sentarse
            57: 'sofá',            # Mueble principal
            59: 'cama',            # Orientación en dormitorio
            60: 'mesa de comedor', # Superficie para comer
            61: 'inodoro',         # Higiene personal
            62: 'televisor',       # Entretenimiento
            39: 'botella',         # Hidratación
            41: 'taza',            # Bebidas calientes
            67: 'teléfono celular', # Comunicación
            73: 'libro',           # Lectura (aunque mejor OCR)
            0: 'persona'           # Personas cercanas
        }
        
        print("✅ Detector inicializado correctamente")
    
    def detectar(self, imagen: np.ndarray, solo_prioritarios: bool = True) -> List[Dict]:
        """
        Detecta objetos en una imagen
        
        Args:
            imagen: Imagen en formato numpy array (BGR)
            solo_prioritarios: Si True, solo reporta objetos prioritarios
            
        Returns:
            Lista de diccionarios con información de detecciones
        """
        try:
            # Ejecutar detección
            resultados = self.modelo(imagen, conf=self.confianza_minima, verbose=False)
            
            detecciones = []
            
            for resultado in resultados:
                boxes = resultado.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extraer información de la detección
                        clase_id = int(box.cls[0])
                        confianza = float(box.conf[0])
                        
                        # Filtrar solo objetos prioritarios si se solicita
                        if solo_prioritarios and clase_id not in self.objetos_prioritarios:
                            continue
                        
                        # Coordenadas de la caja delimitadora
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        # Calcular centro y área
                        centro_x = (x1 + x2) / 2
                        centro_y = (y1 + y2) / 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        # Determinar posición relativa
                        altura_img, ancho_img = imagen.shape[:2]
                        posicion = self._calcular_posicion(centro_x, centro_y, ancho_img, altura_img)
                        
                        # Estimar distancia relativa basada en el área
                        distancia_relativa = self._estimar_distancia(area, ancho_img * altura_img)
                        
                        deteccion = {
                            'clase_id': clase_id,
                            'nombre': self.etiquetas_es.get(clase_id, f'objeto_{clase_id}'),
                            'confianza': round(confianza, 2),
                            'posicion': posicion,
                            'distancia_relativa': distancia_relativa,
                            'coordenadas': {
                                'x1': int(x1), 'y1': int(y1),
                                'x2': int(x2), 'y2': int(y2)
                            },
                            'centro': {'x': int(centro_x), 'y': int(centro_y)},
                            'area': int(area)
                        }
                        
                        detecciones.append(deteccion)
            
            return detecciones
            
        except Exception as e:
            print(f"❌ Error en detección: {e}")
            return []
    
    def _calcular_posicion(self, centro_x: float, centro_y: float, ancho: int, alto: int) -> str:
        """Calcula la posición relativa del objeto en la imagen"""
        # Dividir imagen en 9 regiones (3x3)
        tercio_ancho = ancho / 3
        tercio_alto = alto / 3
        
        if centro_x < tercio_ancho:
            horizontal = "izquierda"
        elif centro_x < 2 * tercio_ancho:
            horizontal = "centro"
        else:
            horizontal = "derecha"
        
        if centro_y < tercio_alto:
            vertical = "arriba"
        elif centro_y < 2 * tercio_alto:
            vertical = "medio"
        else:
            vertical = "abajo"
        
        if horizontal == "centro" and vertical == "medio":
            return "en el centro"
        elif horizontal == "centro":
            return f"en el {vertical}"
        elif vertical == "medio":
            return f"a la {horizontal}"
        else:
            return f"{vertical} a la {horizontal}"
    
    def _estimar_distancia(self, area_objeto: float, area_total: float) -> str:
        """Estima distancia relativa basada en el área del objeto"""
        porcentaje_area = (area_objeto / area_total) * 100
        
        if porcentaje_area > 25:
            return "muy cerca"
        elif porcentaje_area > 10:
            return "cerca"
        elif porcentaje_area > 5:
            return "a distancia media"
        else:
            return "lejos"
    
    def generar_descripcion_audio(self, detecciones: List[Dict]) -> str:
        """
        Genera una descripción en texto natural para síntesis de voz
        
        Args:
            detecciones: Lista de detecciones del método detectar()
            
        Returns:
            Texto descriptivo para convertir a audio
        """
        if not detecciones:
            return "No se detectan objetos en este momento"
        
        if len(detecciones) == 1:
            det = detecciones[0]
            return f"Detecto {det['nombre']} {det['posicion']}, {det['distancia_relativa']}"
        
        # Múltiples detecciones
        descripciones = []
        for det in detecciones:
            descripciones.append(f"{det['nombre']} {det['posicion']}")
        
        if len(descripciones) <= 3:
            return f"Detecto: {', '.join(descripciones[:-1])} y {descripciones[-1]}"
        else:
            return f"Detecto {len(descripciones)} objetos: {', '.join(descripciones[:2])} y otros más"
    
    def dibujar_detecciones(self, imagen: np.ndarray, detecciones: List[Dict]) -> np.ndarray:
        """
        Dibuja las detecciones en la imagen para debugging visual
        
        Args:
            imagen: Imagen original
            detecciones: Lista de detecciones
            
        Returns:
            Imagen con las detecciones dibujadas
        """
        imagen_resultado = imagen.copy()
        
        for det in detecciones:
            coords = det['coordenadas']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            
            # Dibujar rectángulo
            color = (0, 255, 0)  # Verde
            cv2.rectangle(imagen_resultado, (x1, y1), (x2, y2), color, 2)
            
            # Etiqueta con nombre y confianza
            etiqueta = f"{det['nombre']} {det['confianza']:.2f}"
            
            # Fondo para el texto
            (w_texto, h_texto), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(imagen_resultado, (x1, y1 - h_texto - 10), (x1 + w_texto, y1), color, -1)
            
            # Texto
            cv2.putText(imagen_resultado, etiqueta, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return imagen_resultado