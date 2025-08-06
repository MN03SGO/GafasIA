import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
import colorlog
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install


class ConfiguradorLogger:
    
    def __init__(self, nombre_logger="GafasIA", nivel="INFO"):
        # **Configuracion de logger
        self.nombre_logger = nombre_logger
        self.nivel = getattr(logging, nivel.upper(), logging.INFO)
        self.console = Console()
        
        self.directorio_logs = Path("logs")
        self.directorio_logs.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(nombre_logger)
        self.logger.setLevel(self.nivel)
        
        if not self.logger.handlers:
            self._configurar_handlers()
    
    def _configurar_handlers(self):
        self._configurar_handler_consola()
        
        self._configurar_handler_archivo()
        
        self._configurar_handler_errores()
    def _configurar_handler_consola(self):
        try: 
            colores_log = {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
            formato_colores = (
                "%(log_color)s%(asctime)s %(name)s [%(levelname)s]%(reset)s "
                "%(blue)s%(filename)s:%(lineno)d%(reset)s - %(message)s"
            )
            handler_consola = colorlog.StreamHandler(sys.stdout)
            handler_consola.setLevel(self.nivel)
            
            formatter_colores = colorlog.ColoredFormatter(
                formato_colores,
                datefmt='%H:%M:%S',
                log_colors=colores_log,
                secondary_log_colors={},
                style='%'
            )
            
            handler_consola.setFormatter(formatter_colores)
            self.logger.addHandler(handler_consola)
            
        except Exception as e:
            print(f"Error configurando colores en consola: {e}")
            self._configurar_handler_consola_basico()
    
    def _configurar_handler_consola_basico(self):
        #andler de consola básico sin colores (fallback)
        handler_consola = logging.StreamHandler(sys.stdout)
        handler_consola.setLevel(self.nivel)
        
        formato_basico = (
            "%(asctime)s - %(name)s [%(levelname)s] "
            "%(filename)s:%(lineno)d - %(message)s"
        )
        formatter_basico = logging.Formatter(
            formato_basico,
            datefmt='%H:%M:%S'
        )
        handler_consola.setFormatter(formatter_basico)
        self.logger.addHandler(handler_consola)
    
    def _configurar_handler_archivo(self):
        #handler de rotacion
        try:
            archivo_principal = self.directorio_logs / "gafas_ia.log"
            
            # Handler con rotación automática
            handler_archivo = logging.handlers.RotatingFileHandler(
                archivo_principal,
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5,         
                encoding='utf-8'
            )
            
            handler_archivo.setLevel(logging.DEBUG) 
            formato_archivo = (
                "%(asctime)s | %(name)s | %(levelname)s | "
                "%(filename)s:%(funcName)s:%(lineno)d | %(message)s"
            )
            formatter_archivo = logging.Formatter(
                formato_archivo,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            handler_archivo.setFormatter(formatter_archivo)
            self.logger.addHandler(handler_archivo)
            
        except Exception as e:
            self.logger.error(f"Error configurando handler de archivo: {e}")

    def _configurar_handler_errores(self):
        # criticos
        try:
            archivo_errores = self.directorio_logs / "errores.log"
            
            # errores criticos 
            handler_errores = logging.handlers.RotatingFileHandler(
                archivo_errores,
                maxBytes=5*1024*1024,   # 5 MB
                backupCount=3,       
                encoding='utf-8'
            )
            
            handler_errores.setLevel(logging.ERROR)
            
            formato_errores = (
                "%(asctime)s | ERROR CRÍTICO | %(name)s | "
                "%(filename)s:%(funcName)s:%(lineno)d | "
                "%(message)s | %(exc_info)s"
            )
            
            formatter_errores = logging.Formatter(
                formato_errores,
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            handler_errores.setFormatter(formatter_errores)
            self.logger.addHandler(handler_errores)
            
        except Exception as e:
            self.logger.error(f"Error configurando handler de errores: {e}")
    
    def obtener_logger(self):
        return self.logger
    
    def cambiar_nivel(self, nuevo_nivel):
        
        try:
            nivel_numerico = getattr(logging, nuevo_nivel.upper())
            self.logger.setLevel(nivel_numerico)
            self.nivel = nivel_numerico
            
            for handler in self.logger.handlers:
                if isinstance(handler, (logging.StreamHandler, colorlog.StreamHandler)):
                    handler.setLevel(nivel_numerico)
            
            self.logger.info(f"Nivel de logging cambiado a: {nuevo_nivel.upper()}")
            
        except AttributeError:
            self.logger.error(f"Nivel de logging inválido: {nuevo_nivel}")
    
    def obtener_estadisticas(self):

        estadisticas = {}
        
        try:
            for archivo_log in self.directorio_logs.glob("*.log"):
                if archivo_log.exists():
                    stat = archivo_log.stat()
                    estadisticas[archivo_log.name] = {
                        'tamaño_mb': round(stat.st_size / (1024*1024), 2),
                        'modificado': datetime.fromtimestamp(stat.st_mtime),
                        'lineas': self._contar_lineas(archivo_log)
                    }
        except Exception as e:
            self.logger.error(f"Error obteniendo estadísticas: {e}")
        
        return estadisticas
    
    def _contar_lineas(self, archivo):
        try:
            with open(archivo, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f)
        except Exception:
            return 0
    
    def limpiar_logs_antiguos(self, dias=30):
        try:
            import time
            tiempo_limite = time.time() - (dias * 24 * 60 * 60)
            
            archivos_eliminados = 0
            for archivo_log in self.directorio_logs.glob("*.log*"):
                if archivo_log.stat().st_mtime < tiempo_limite:
                    archivo_log.unlink()
                    archivos_eliminados += 1
            
            if archivos_eliminados > 0:
                self.logger.info(f"Eliminados {archivos_eliminados} archivos de log antiguos")
                
        except Exception as e:
            self.logger.error(f"Error limpiando logs antiguos: {e}")

class LoggerGafasIA:
    
    def __init__(self, nombre_modulo):
        
        self.nombre_modulo = nombre_modulo
        self.configurador = ConfiguradorLogger(f"GafasIA.{nombre_modulo}")
        self.logger = self.configurador.obtener_logger()
    
    def debug(self, mensaje, extra_info=None):
        
        mensaje_completo = f"{mensaje}"
        if extra_info:
            mensaje_completo += f" | Extra: {extra_info}"
        self.logger.debug(mensaje_completo)
    
    def info(self, mensaje, extra_info=None):
        """Log de información con emoji"""
        mensaje_completo = f"ℹ{mensaje}"
        if extra_info:
            mensaje_completo += f" | {extra_info}"
        self.logger.info(mensaje_completo)
    
    def warning(self, mensaje, extra_info=None):
        """Log de advertencia con emoji"""
        mensaje_completo = f" {mensaje}"
        if extra_info:
            mensaje_completo += f" | {extra_info}"
        self.logger.warning(mensaje_completo)
    
    def error(self, mensaje, excepcion=None, extra_info=None):
        """Log de error con manejo de excepciones"""
        mensaje_completo = f"{mensaje}"
        if extra_info:
            mensaje_completo += f" | {extra_info}"
        
        if excepcion:
            self.logger.error(mensaje_completo, exc_info=excepcion)
        else:
            self.logger.error(mensaje_completo)
    
    def critical(self, mensaje, excepcion=None, extra_info=None):
        """Log crítico - para errores que pueden detener el sistema"""
        mensaje_completo = f"CRITICO: {mensaje}"
        if extra_info:
            mensaje_completo += f" | {extra_info}"
        
        if excepcion:
            self.logger.critical(mensaje_completo, exc_info=excepcion)
        else:
            self.logger.critical(mensaje_completo)
    
    def funcionalidad_activada(self, nombre_funcionalidad):
        """Log específico para cuando se activa una funcionalidad"""
        self.info(f"Funcionalidad activada: {nombre_funcionalidad}")
    
    def funcionalidad_desactivada(self, nombre_funcionalidad):
        """Log específico para cuando se desactiva una funcionalidad"""
        self.info(f"Funcionalidad desactivada: {nombre_funcionalidad}")
    
    def deteccion_encontrada(self, tipo_deteccion, cantidad=1, detalles=None):
        
        mensaje = f"Detectado: {tipo_deteccion}"
        if cantidad > 1:
            mensaje += f" (x{cantidad})"
        if detalles:
            mensaje += f" - {detalles}"
        self.info(mensaje)
    
    def rendimiento(self, metrica, valor, unidad="ms"):
        
        self.debug(f"Rendimiento - {metrica}: {valor}{unidad}")
    
    def inicio_modulo(self):
        
        self.info(f"Iniciando módulo: {self.nombre_modulo}")
        
    def fin_modulo(self):
        
        self.info(f" Finalizando módulo: {self.nombre_modulo}")

def obtener_logger_modulo(nombre_modulo):

    return LoggerGafasIA(nombre_modulo)

def configurar_logging_sistema(config=None):

    if config and 'logging' in config:
        nivel = config['logging'].get('nivel', 'INFO')
        
        configurador = ConfiguradorLogger("GafasIA", nivel)
        logger = configurador.obtener_logger()
        
        logger.info("Sistema de logging configurado desde ajuste.yaml")
        
        if config['logging'].get('rotacion', False):
            configurador.limpiar_logs_antiguos()
        
        return logger
    else:
        # Configuración por defecto
        configurador = ConfiguradorLogger("GafasIA", "INFO")
        logger = configurador.obtener_logger()
        logger.warning("Usando configuración de logging por defecto")
        return logger

# Crear logger principal del sistema al importar el módulo
logger_principal = ConfiguradorLogger("GafasIA").obtener_logger()

if __name__ == "__main__":
    print("Probando sistema de logging de Gafas IA...")
    
    logger_prueba = obtener_logger_modulo("PruebaLogger")
    
    logger_prueba.inicio_modulo()
    logger_prueba.debug("Mensaje de debug para desarrollo")
    logger_prueba.info("Información general del sistema")
    logger_prueba.warning("Advertencia de algo que podría ser problemático")
    logger_prueba.error("Error que fue manejado correctamente")
    logger_prueba.funcionalidad_activada("Detección de Objetos")
    logger_prueba.deteccion_encontrada("Persona", 2, "Frente a la cámara")
    logger_prueba.rendimiento("Procesamiento frame", 45.2, "ms")
    
    try:
        resultado = 1 / 0
    except Exception as e:
        logger_prueba.critical("Error crítico simulado", excepcion=e)
    
    logger_prueba.fin_modulo()
    
    #estadísticas
    configurador = ConfiguradorLogger()
    stats = configurador.obtener_estadisticas()
    
    print("\n Estadísticas de archivos de log:")
    for archivo, info in stats.items():
        print(f"   {archivo}: {info['tamaño_mb']} MB, {info['lineas']} líneas")
    
    print(" Prueba del sistema de logging completada")