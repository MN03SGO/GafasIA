#10/08/2025 PROPENSO A CAMBIOS 
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict, Optional
import os
from pathlib import Path

class ObjectDetector:
    def __init__(self, model_path: str = "models/yolov8n.pt", confidence: float = 0.5):
    
        self.model_path = model_path
        self.confidence = confidence
        self.model = None
        self.class_names = []
        
        # Diccionario para traducir nombres de objetos al español
        self.translations = {
            'person': 'persona', 'bicycle': 'bicicleta', 'car': 'coche', 'motorcycle': 'motocicleta',
            'airplane': 'avión', 'bus': 'autobús', 'train': 'tren', 'truck': 'camión',
            'boat': 'barco', 'traffic light': 'semáforo', 'fire hydrant': 'hidrante',
            'stop sign': 'señal de alto', 'parking meter': 'parquímetro', 'bench': 'banco',
            'bird': 'pájaro', 'cat': 'gato', 'dog': 'perro', 'horse': 'caballo',
            'sheep': 'oveja', 'cow': 'vaca', 'elephant': 'elefante', 'bear': 'oso',
            'zebra': 'cebra', 'giraffe': 'jirafa', 'backpack': 'mochila', 'umbrella': 'paraguas',
            'handbag': 'bolso', 'tie': 'corbata', 'suitcase': 'maleta', 'frisbee': 'frisbee',
            'skis': 'esquís', 'snowboard': 'snowboard', 'sports ball': 'pelota',
            'kite': 'cometa', 'baseball bat': 'bate de béisbol', 'baseball glove': 'guante de béisbol',
            'skateboard': 'monopatín', 'surfboard': 'tabla de surf', 'tennis racket': 'raqueta de tenis',
            'bottle': 'botella', 'wine glass': 'copa de vino', 'cup': 'taza', 'fork': 'tenedor',
            'knife': 'cuchillo', 'spoon': 'cuchara', 'bowl': 'cuenco', 'banana': 'banana',
            'apple': 'manzana', 'sandwich': 'sándwich', 'orange': 'naranja', 'broccoli': 'brócoli',
            'carrot': 'zanahoria', 'hot dog': 'perrito caliente', 'pizza': 'pizza',
            'donut': 'dona', 'cake': 'pastel', 'chair': 'silla', 'couch': 'sofá',
            'potted plant': 'planta en maceta', 'bed': 'cama', 'dining table': 'mesa de comedor',
            'toilet': 'inodoro', 'tv': 'televisión', 'laptop': 'portátil', 'mouse': 'ratón',
            'remote': 'control remoto', 'keyboard': 'teclado', 'cell phone': 'teléfono móvil',
            'microwave': 'microondas', 'oven': 'horno', 'toaster': 'tostadora',
            'sink': 'fregadero', 'refrigerator': 'refrigerador', 'book': 'libro',
            'clock': 'reloj', 'vase': 'jarrón', 'scissors': 'tijeras', 'teddy bear': 'osito de peluche'
        }
    
    def load_model(self) -> bool:
        """Carga el modelo YOLO"""
        try:
            # Crear directorio de modelos si no existe
            os.makedirs("models", exist_ok=True)
            
            print(f"Cargando modelo YOLO desde: {self.model_path}")
            
            # Si no existe el modelo, YOLO lo descargará automáticamente
            if not os.path.exists(self.model_path):
                print("Descargando modelo YOLOv8n (primera vez)...")
            
            self.model = YOLO(self.model_path)
            self.class_names = self.model.names  # Diccionario {id: nombre}
            
            print(f"Modelo cargado exitosamente")
            print(f"Clases disponibles: {len(self.class_names)}")
            return True
            
        except Exception as e:
            print(f"Error cargando modelo YOLO: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
  
           # Lista de detecciones con formato:
           # [{'name': str, 'confidence': float, 'bbox': [x1,y1,x2,y2], 'center': [x,y]}]
     
        if self.model is None:
            return []
        
        try:
            # Realizar detección
            results = self.model(frame, conf=self.confidence, verbose=False)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extraer información de la detección
                        xyxy = box.xyxy[0].cpu().numpy()  # Coordenadas [x1,y1,x2,y2]
                        conf = float(box.conf[0].cpu().numpy())  # Confianza
                        cls = int(box.cls[0].cpu().numpy())  # Clase
                        
                        # Nombre del objeto en inglés y español
                        name_en = self.class_names[cls]
                        name_es = self.translations.get(name_en, name_en)
                        
                        # Calcular centro del objeto
                        center_x = int((xyxy[0] + xyxy[2]) / 2)
                        center_y = int((xyxy[1] + xyxy[3]) / 2)
                        
                        detection = {
                            'name': name_es,
                            'name_en': name_en,
                            'confidence': conf,
                            'bbox': xyxy.astype(int).tolist(),
                            'center': [center_x, center_y],
                            'class_id': cls
                        }
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            print(f"Error en detección: {e}")
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
      
        frame_copy = frame.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            name = detection['name']
            confidence = detection['confidence']
            center = detection['center']
            
            # Coordenadas del bounding box
            x1, y1, x2, y2 = bbox
            
            # Color basado en la confianza (verde más intenso = mayor confianza)
            color_intensity = int(255 * confidence)
            color = (0, color_intensity, 0)  # Verde
            
            # Dibujar bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Punto central
            cv2.circle(frame_copy, tuple(center), 5, (0, 0, 255), -1)  # Rojo
            
            # Texto con nombre y confianza
            label = f"{name} {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Fondo para el texto
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                        (x1 + label_size[0], y1), color, -1)
            
            # Texto
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame_copy
    
    def get_detection_summary(self, detections: List[Dict]) -> str:
    
        if not detections:
            return "No se detectaron objetos"
        
        # Contar objetos únicos
        object_counts = {}
        for detection in detections:
            name = detection['name']
            object_counts[name] = object_counts.get(name, 0) + 1
        
        # Generar resumen
        if len(object_counts) == 1:
            obj_name, count = list(object_counts.items())[0]
            if count == 1:
                return f"Veo un {obj_name}"
            else:
                return f"Veo {count} {obj_name}s"
        else:
            summary_parts = []
            for obj_name, count in object_counts.items():
                if count == 1:
                    summary_parts.append(f"un {obj_name}")
                else:
                    summary_parts.append(f"{count} {obj_name}s")
            
            if len(summary_parts) <= 2:
                return f"Veo {' y '.join(summary_parts)}"
            else:
                return f"Veo {', '.join(summary_parts[:-1])} y {summary_parts[-1]}"
    
    def get_closest_object(self, detections: List[Dict], frame_center: Tuple[int, int]) -> Optional[Dict]:
 
        if not detections:
            return None
        
        min_distance = float('inf')
        closest_object = None
        
        for detection in detections:
            center = detection['center']
            distance = np.sqrt((center[0] - frame_center[0])**2 + (center[1] - frame_center[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                closest_object = detection
        
        return closest_object
