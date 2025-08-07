import yaml
import os
import platform
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import copy

try:
    from .logger import obtener_logger_modulo
    logger = obtener_logger_modulo("ConfiguradorSistema")
except ImportError:
    import logging
    logger = logging.getLogger("ConfiguradorSistema")

class DetectorHardware:

    
    @staticmethod
    def es_raspberry_pi():
        try:
            if Path('/proc/device-tree/model').exists():
                with open('/proc/device-tree/model', 'r') as f:
                    modelo = f.read().lower()
                    if 'raspberry pi' in modelo:
                        return True
                    
            if Path('/proc/cpuinfo').exists():
                with open('/proc/cpuinfo', 'r') as f:
                    contenido = f.read().lower()
                    if 'raspberry pi' in contenido or 'bcm' in contenido:
                        return True
            
            # variables de entorno
            if 'RASPBERRY_PI' in os.environ:
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error detectando Raspberry Pi: {e}")
            return False
    
    @staticmethod
    def obtener_info_sistema():
        # información detallada del sistema
        try:
            info = {
                'es_raspberry_pi': DetectorHardware.es_raspberry_pi(),
                'sistema_operativo': platform.system(),
                'version_sistema': platform.release(),
                'arquitectura': platform.machine(),
                'procesador': platform.processor(),
                'python_version': sys.version,
                'memoria_total_gb': None,
                'cpu_count': os.cpu_count()
            }
            
            # información de memoria (Linux)
            try:
                if Path('/proc/meminfo').exists():
                    with open('/proc/meminfo', 'r') as f:
                        for linea in f:
                            if 'MemTotal:' in linea:
                                # Convertir de kB a GB
                                kb = int(linea.split()[1])
                                info['memoria_total_gb'] = round(kb / (1024 * 1024), 2)
                                break
            except Exception:
                pass
            
            return info
            
        except Exception as e:
            logger.error(f"Error obteniendo información del sistema: {e}")
            return {}
    
    @staticmethod
    def gpu_disponible():
        # aceleracion de gpu
        gpu_info = {
            'nvidia_disponible': False,
            'cuda_disponible': False,
            'gpu_nombre': None
        }
        
        try:
            
            import subprocess
            try:
                resultado = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                        capture_output=True, text=True, timeout=5)
                if resultado.returncode == 0:
                    gpu_info['nvidia_disponible'] = True
                    gpu_info['gpu_nombre'] = resultado.stdout.strip()
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # CUDA con PyTorch
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_info['cuda_disponible'] = True
                    if not gpu_info['gpu_nombre']:
                        gpu_info['gpu_nombre'] = torch.cuda.get_device_name(0)
            except ImportError:
                pass
            
        except Exception as e:
            logger.debug(f"Error verificando GPU: {e}")
        
        return gpu_info

class ValidadorConfiguracion:
    # respaldo
    CONFIG_DEFECTO = {
        'proyecto': {
            'nombre': 'Gafas IA',
            'version': '1.0.0',
            'autor': 'Sigaran'
        },
        'camara': {
            'indice': 0,
            'resolucion': {'ancho': 640, 'alto': 480},
            'fps': 30
        },
        'deteccion_objetos': {
            'activa': True,
            'modelo': 'yolov8n',
            'confianza_minima': 0.5
        },
        'deteccion_personas': {
            'activa': True,
            'metodo': 'yolo'
        },
        'reconocimiento_texto': {
            'activa': False,
            'idiomas': ['spa', 'eng']
        },
        'texto_a_voz': {
            'motor': 'pyttsx3',
            'pyttsx3': {
                'velocidad': 150,
                'volumen': 0.9
            }
        },
        'controles': {
            'salir': 'q',
            'toggle_objetos': 'o',
            'toggle_personas': 'p',
            'toggle_ocr': 't',
            'ayuda': 'h'
        },
        'logging': {
            'nivel': 'INFO',
            'consola': True
        }
    }
    
    @staticmethod
    def validar_estructura(config):
        errores = []
        
        try:
            # secciones principales
            secciones_requeridas = [
                'camara', 'deteccion_objetos', 'texto_a_voz', 'controles'
            ]
            
            for seccion in secciones_requeridas:
                if seccion not in config:
                    errores.append(f"Sección faltante: {seccion}")
            
            # configuración de cámara
            if 'camara' in config:
                camara = config['camara']
                if 'resolucion' in camara:
                    res = camara['resolucion']
                    if not all(k in res for k in ['ancho', 'alto']):
                        errores.append("Configuración de resolución incompleta")
                    
                    if res.get('ancho', 0) <= 0 or res.get('alto', 0) <= 0:
                        errores.append("Resolución inválida (debe ser > 0)")
                
                if camara.get('fps', 0) <= 0:
                    errores.append("FPS inválido (debe ser > 0)")
            
            # configuración de detección
            if 'deteccion_objetos' in config:
                det_obj = config['deteccion_objetos']
                confianza = det_obj.get('confianza_minima', 0.5)
                if not 0 <= confianza <= 1:
                    errores.append("Confianza mínima debe estar entre 0 y 1")
            
            # configuración de voz
            if 'texto_a_voz' in config:
                tts = config['texto_a_voz']
                if 'pyttsx3' in tts:
                    pyttsx3_config = tts['pyttsx3']
                    velocidad = pyttsx3_config.get('velocidad', 150)
                    if not 50 <= velocidad <= 300:
                        errores.append("Velocidad de voz debe estar entre 50 y 300")
                    
                    volumen = pyttsx3_config.get('volumen', 0.9)
                    if not 0 <= volumen <= 1:
                        errores.append("Volumen debe estar entre 0 y 1")
            
        except Exception as e:
            errores.append(f"Error durante validación: {str(e)}")
        
        return len(errores) == 0, errores
    
    @staticmethod
    def completar_configuracion_faltante(config):

        config_completa = copy.deepcopy(ValidadorConfiguracion.CONFIG_DEFECTO)
        
        def fusionar_diccionarios(base, nuevo):
            for clave, valor in nuevo.items():
                if clave in base and isinstance(base[clave], dict) and isinstance(valor, dict):
                    fusionar_diccionarios(base[clave], valor)
                else:
                    base[clave] = valor
        
        fusionar_diccionarios(config_completa, config)
        return config_completa

class OptimizadorConfiguracion:
    
    @staticmethod
    def optimizar_para_hardware(config, info_hardware):
        config_optimizada = copy.deepcopy(config)

        if info_hardware.get('es_raspberry_pi', False):
            logger.info("Detectado Raspberry Pi - Aplicando optimizaciones")
            
            # Reducir resolución para mejor rendimiento
            if 'raspberry_pi' in config_optimizada.get('rendimiento', {}):
                rpi_config = config_optimizada['rendimiento']['raspberry_pi']
                if rpi_config.get('reducir_resolucion', False):
                    res_reducida = rpi_config.get('resolucion_reducida', [320, 240])
                    config_optimizada['camara']['resolucion'] = {
                        'ancho': res_reducida[0],
                        'alto': res_reducida[1]
                    }
                fps_reducido = rpi_config.get('fps_reducido', 15)
                config_optimizada['camara']['fps'] = fps_reducido
                
                # modelo liviano
                config_optimizada['deteccion_objetos']['modelo'] = 'yolov8n'
        
        # optimizaciones memoria limitada
        memoria_gb = info_hardware.get('memoria_total_gb', 8)
        if memoria_gb and memoria_gb < 4:
            logger.info(f"Memoria limitada ({memoria_gb}GB) - Aplicando optimizaciones")
            
            if 'rendimiento' in config_optimizada:
                config_optimizada['rendimiento']['buffer_frames'] = 1
            
            # usar modelo más pequeño
            config_optimizada['deteccion_objetos']['modelo'] = 'yolov8n'
            config_optimizada['deteccion_objetos']['max_detecciones'] = 5
        
        # optimizaciones para GPU
        gpu_info = DetectorHardware.gpu_disponible()
        if gpu_info['cuda_disponible']:
            logger.info(f"GPU CUDA detectada: {gpu_info['gpu_nombre']}")
            if 'rendimiento' in config_optimizada:
                if 'pc_desarrollo' in config_optimizada['rendimiento']:
                    config_optimizada['rendimiento']['pc_desarrollo']['usar_gpu'] = True
        
        return config_optimizada

class GestorConfiguracion:
    def __init__(self, archivo_config="config/ajuste.yaml"):
        
        self.archivo_config = Path(archivo_config)
        self.configuracion = {}
        self.info_hardware = DetectorHardware.obtener_info_sistema()
        
        logger.info(f"Inicializando gestor de configuración: {archivo_config}")
        
        
        self.cargar_configuracion()
    
    def cargar_configuracion(self): # ** .yaml
        try:
            if not self.archivo_config.exists():
                logger.warning(f"Archivo de configuración no encontrado: {self.archivo_config}")
                logger.info("Creando configuración por defecto...")
                self._crear_configuracion_defecto()
                return
            
            with open(self.archivo_config, 'r', encoding='utf-8') as archivo:
                self.configuracion = yaml.safe_load(archivo)
            
            logger.info(f"Configuración cargada desde: {self.archivo_config}")

            es_valida, errores = ValidadorConfiguracion.validar_estructura(self.configuracion)
            
            if not es_valida:
                logger.warning("Errores en configuracion:")
                for error in errores:
                    logger.warning(f"  - {error}")
                logger.info("Completando configuración con valores por defecto")
            
            self.configuracion = ValidadorConfiguracion.completar_configuracion_faltante(self.configuracion)
            
            # pc de la casa
            self.configuracion = OptimizadorConfiguracion.optimizar_para_hardware(
                self.configuracion, self.info_hardware
            )
            
            # 
            self._mostrar_info_sistema()
            
        except yaml.YAMLError as e:
            logger.error(f"Error de formato YAML: {e}")
            self._usar_configuracion_emergencia()
        
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            self._usar_configuracion_emergencia()
    
    def _crear_configuracion_defecto(self):
        try:
            self.archivo_config.parent.mkdir(parents=True, exist_ok=True)
            
            self.configuracion = ValidadorConfiguracion.CONFIG_DEFECTO
            
            with open(self.archivo_config, 'w', encoding='utf-8') as archivo:
                yaml.dump(
                    self.configuracion, 
                    archivo, 
                    default_flow_style=False, 
                    allow_unicode=True,
                    indent=2
                )
            
            logger.info(f"Archivo de configuración creado: {self.archivo_config}")
            
        except Exception as e:
            logger.error(f"Error creando configuración por defecto: {e}")
            self._usar_configuracion_emergencia()
    
    def _usar_configuracion_emergencia(self):
        logger.warning("Usando configuración de emergencia mínima")
        self.configuracion = ValidadorConfiguracion.CONFIG_DEFECTO
    
    def _mostrar_info_sistema(self):
        info = self.info_hardware
        
        logger.info("Información del sistema detectada:")
        logger.info(f"SO: {info.get('sistema_operativo', 'Desconocido')} {info.get('version_sistema', '')}")
        logger.info(f"Arquitectura: {info.get('arquitectura', 'Desconocida')}")
        logger.info(f"CPUs: {info.get('cpu_count', 'Desconocido')}")
        
        if info.get('memoria_total_gb'):
            logger.info(f"Memoria: {info['memoria_total_gb']} GB")
        
        if info.get('es_raspberry_pi'):
            logger.info("Raspberry Pi detectado")
        
        # Información de GPU
        gpu_info = DetectorHardware.gpu_disponible()
        if gpu_info['cuda_disponible']:
            logger.info(f"GPU CUDA: {gpu_info['gpu_nombre']}")
        elif gpu_info['nvidia_disponible']:
            logger.info(f"GPU NVIDIA: {gpu_info['gpu_nombre']} (sin CUDA)")
    
    def obtener(self, clave, valor_defecto=None):
        try:
            valor_actual = self.configuracion

            for parte in clave.split('.'):
                valor_actual = valor_actual[parte]
            
            return valor_actual
            
        except (KeyError, TypeError):
            logger.debug(f"Clave de configuración no encontrada: {clave}")
            return valor_defecto
    
    def establecer(self, clave, valor):
        try:
            partes = clave.split('.')
            valor_actual = self.configuracion
            
            for parte in partes[:-1]:
                if parte not in valor_actual:
                    valor_actual[parte] = {}
                valor_actual = valor_actual[parte]
            
            valor_actual[partes[-1]] = valor
            
            logger.debug(f"Configuración actualizada: {clave} = {valor}")
            
        except Exception as e:
            logger.error(f"Error estableciendo configuración {clave}: {e}")
    
    def guardar_configuracion(self):
        
        try:
            with open(self.archivo_config, 'w', encoding='utf-8') as archivo:
                yaml.dump(
                    self.configuracion,
                    archivo,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2
                )
            
            logger.info(f"Configuración guardada en: {self.archivo_config}")
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
    
    def obtener_toda_configuracion(self):

        return copy.deepcopy(self.configuracion)
    
    def es_raspberry_pi(self):

        return self.info_hardware.get('es_raspberry_pi', False)
    
    def obtener_info_hardware(self):
        
        return copy.deepcopy(self.info_hardware)

_gestor_config = None

def obtener_configuracion(archivo_config="config/ajuste.yaml"):
    global _gestor_config
    
    if _gestor_config is None:
        _gestor_config = GestorConfiguracion(archivo_config)
    
    return _gestor_config

if __name__ == "__main__":
    
    print("Probando sistema de configuración de Gafas IA")
    
    config = GestorConfiguracion("config/ajuste.yaml")
    print(f"Resolución cámara: {config.obtener('camara.resolucion')}")
    print(f"Modelo detección: {config.obtener('deteccion_objetos.modelo')}")
    print(f"Motor TTS: {config.obtener('texto_a_voz.motor')}")
    print(f"Tecla salir: {config.obtener('controles.salir')}")
    
    print(f"Valor inexistente: {config.obtener('no.existe', 'DEFECTO')}")
    
    # información de hardware
    hardware = config.obtener_info_hardware()
    print(f"Raspberry Pi: {config.es_raspberry_pi()}")
    print(f"CPUs: {hardware.get('cpu_count', 'Desconocido')}")
    
    print("Prueba del sistema de configuración completada")