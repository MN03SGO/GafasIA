
import yaml
from pathlib import Path
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = "config/ajuste.yaml"):
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
        'camara': {'id_dispositivo': 0, 'ancho': 640, 'alto': 480, 'fps': 30},
        'deteccion': {'confidence_threshold': 0.5, 'model_path': 'models/yolov8n.pt'},
        'tts': {'rate': 150, 'volume': 0.9, 'voice_id': 0},
        'reconocimiento_facial': {'tolerancia': 0.6, 'model': 'hog'},
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
    #AJUS del archivo yaml
    def camara_config(self) ->Dict[str, Any]:
        return self.config.get('camara', {})
    
    def deteccion_config(self) ->Dict[str, Any]:
        return self.config.get('deteccion', {})

    def tts_config(self) -> Dict[str, Any]:
        return self.config.get('tts', {})
    
    def reconocimiento_facial_config(self) -> Dict[str,Any]:
        return self.config.get('reconocimiento_facial', {})
        
    def ocr_config(self) -> Dict[str, Any]:
        return self.config.get('ocr', {})

# Instancia global de configuración
config = Config()