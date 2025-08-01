#Configuraciones de cam
import cv2
import numpy as np
from typing import Optional, Tuple
from .config import config

class Camera:
    def __init__(self, device_id: Optional[int] = None):
        self.device_id = device_id or config.get('camera.device_id', 0)
        self.width = config.get('camera.width', 640)
        self.height = config.get('camera.height', 480)
        self.fps = config.get('camera.fps', 30)
        
        self.cap = None
        self.is_opened = False
        
    def start(self) -> bool:
        
        try:
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                print(f"Error: No se puede abrir la cámara {self.device_id}")
                return False
            
            # Configurar propiedades de la cámara
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.is_opened = True
            print(f"Cámara iniciada: {self.width}x{self.height} @ {self.fps}fps")
            return True
            
        except Exception as e:
            print(f"Error iniciando cámara: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            print("Cámara detenida")
    
    def get_info(self) -> dict:
        #Informacion de camra
        if not self.is_opened or self.cap is None:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'device_id': self.device_id
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()