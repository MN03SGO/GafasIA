
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config/settings.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        #Carga la configuración desde el archivo YAML
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Archivo de configuración no encontrado: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error cargando configuración: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:

        return {
            'camera': {'device_id': 0, 'width': 640, 'height': 480, 'fps': 30},
            'detection': {'confidence_threshold': 0.5, 'model_path': 'models/yolov8n.pt'},
            'tts': {'rate': 150, 'volume': 0.9, 'voice_id': 0},
            'face_recognition': {'tolerance': 0.6, 'model': 'hog'},
            'ocr': {'lang': 'spa', 'config': '--psm 6'},
            'app': {'debug': True, 'log_level': 'INFO'}
        }
    
    def get(self, key: str, default=None):
    
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def camera_config(self) -> Dict[str, Any]:
        return self.config.get('camera', {})
    
    def detection_config(self) -> Dict[str, Any]:
        return self.config.get('detection', {})
    
    def tts_config(self) -> Dict[str, Any]:
        return self.config.get('tts', {})
    
    def face_recognition_config(self) -> Dict[str, Any]:
        return self.config.get('face_recognition', {})
    
    def ocr_config(self) -> Dict[str, Any]:
        return self.config.get('ocr', {})

# Instancia global de configuración
config = Config()