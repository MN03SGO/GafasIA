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
        #Ejecucion
        self._ejecutando = False
        self._pausando = False
        self._hilo_principal =None
        self._lock = threading.Lock()
        #Subsistema
        self.camara = None
        self.detector = None
        self.tts = None
        #Deteccion
        self.ultima_deteccion = None
        self.intervalo_deteccion = self.config.get("Deteccion de intervalo por frames",30)
        self.contador_frames = 0
        #Detecciones
        self.ultima_vez_anunciando={}
        self.cooldown_objetos = self.config.get("TTS ",3.0)
        #Metricas
        self.metricas = {
            'frames_procesado':0,
            'detecciones_realizadas':0,
            'objetos_detectados': 0,
            'mensajes_tts': 0,
            'tiempo_inicio':0,
            'fps_promedio': 0
        }
        self._configurar_senales()
        self.logger.info("Gestor principal inciado")
    def cofigurar_senales(self):
        def signal_handler(signum, frame):
            self.logger.info(f"Señal recibida: {signum}")
            self.detener()
            sys.exit(0)
            