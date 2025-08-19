
import pyttsx3
import pygame
import threading
import queue
import time
import os
import tempfile
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
import logging

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Importaciones del src/de utilidad esteccion
from ..utilidades.config import ConfigManager
from ..utilidades.logger import setup_logger

class TextoAVoz:
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        
        self.config_manager = config_manager or ConfigManager()
        self.logger = setup_logger(__name__)
        
        # mensajes de TTS
        self.cola_tts = queue.Queue()
        self.reproduciendo = False
        self.hilo_tts = None
        self.detener_hilo = False
        
        # Configuración TTS
        self.configurar_tts()
        
        self.engine_pyttsx3 = None
        self.pygame_iniciado = False
        
        self._inicializar_engines()
        
        self._iniciar_hilo_tts()
        
        self.logger.info("Sistema TTS inicializado correctamente")
    
    def configurar_tts(self):
        try:
            # Configuración general
            self.habilitado = self.config_manager.get_config('audio.texto_a_voz.habilitado', True)
            self.engine_preferido = self.config_manager.get_config('audio.texto_a_voz.engine', 'pyttsx3')
            self.usar_fallback = self.config_manager.get_config('audio.texto_a_voz.usar_fallback', True)
            
            # Configuración pyttsx3
            pyttsx3_config = self.config_manager.get_config('audio.texto_a_voz.pyttsx3', {})
            self.velocidad = pyttsx3_config.get('velocidad', 180)
            self.volumen = pyttsx3_config.get('volumen', 0.9)
            self.voz_id = pyttsx3_config.get('voz', 'spanish')
            
            # Configuración gTTS
            gtts_config = self.config_manager.get_config('audio.texto_a_voz.gtts', {})
            self.idioma_gtts = gtts_config.get('idioma', 'es')
            self.tld_gtts = gtts_config.get('tld', 'com')
            self.velocidad_gtts = gtts_config.get('lenta', False)
            
            # Configuración avanzada
            avanzado = self.config_manager.get_config('audio.texto_a_voz.avanzado', {})
            self.cola_maxima = avanzado.get('cola_maxima', 10)
            self.timeout_red = avanzado.get('timeout_red', 5)
            self.limpiar_cola_nueva = avanzado.get('limpiar_cola_nueva_deteccion', True)
            self.priorizar_cerca = avanzado.get('priorizar_objetos_cercanos', True)
            
            # Filtros de texto
            filtros = self.config_manager.get_config('audio.texto_a_voz.filtros', {})
            self.longitud_minima = filtros.get('longitud_minima', 3)
            self.longitud_maxima = filtros.get('longitud_maxima', 200)
            self.palabras_ignorar = set(filtros.get('palabras_ignorar', []))
            
            self.logger.info(f"TTS configurado - Engine: {self.engine_preferido}, Velocidad: {self.velocidad}")
            
        except Exception as e:
            self.logger.error(f"Error configurando TTS: {e}")
            # Valores por defecto
            self.habilitado = True
            self.engine_preferido = 'pyttsx3'
            self.velocidad = 180
            self.volumen = 0.9
    
    def _inicializar_engines(self):
        
        self._inicializar_pyttsx3()
        
        self._inicializar_pygame()
        
        self.logger.info(f"Engines disponibles - pyttsx3: {self.engine_pyttsx3 is not None}, "
                        f"gTTS: {GTTS_AVAILABLE}, pygame: {self.pygame_iniciado}")
    
    def _inicializar_pyttsx3(self):

        try:
            self.engine_pyttsx3 = pyttsx3.init()
            
            # Configurar velocidad
            self.engine_pyttsx3.setProperty('rate', self.velocidad)
            
            # Configurar volumen
            self.engine_pyttsx3.setProperty('volume', self.volumen)
            
            # voz española 
            voces = self.engine_pyttsx3.getProperty('voices')
            voz_seleccionada = None
            
            for voz in voces:
                if 'spanish' in voz.name.lower() or 'es' in voz.id.lower():
                    voz_seleccionada = voz
                    break
            
            if voz_seleccionada:
                self.engine_pyttsx3.setProperty('voice', voz_seleccionada.id)
                self.logger.info(f"Voz española seleccionada: {voz_seleccionada.name}")
            else:
                self.logger.warning("No se encontró voz española, usando voz por defecto")
            
        except Exception as e:
            self.logger.error(f"Error inicializando pyttsx3: {e}")
            self.engine_pyttsx3 = None
    
    def _inicializar_pygame(self):
        
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
            self.pygame_iniciado = True
            self.logger.info("Pygame inicializado para gTTS")
        except Exception as e:
            self.logger.error(f"Error inicializando pygame: {e}")
            self.pygame_iniciado = False
    
    def _iniciar_hilo_tts(self):
        
        if not self.habilitado:
            return
            
        self.detener_hilo = False
        self.hilo_tts = threading.Thread(target=self._procesar_cola_tts, daemon=True)
        self.hilo_tts.start()
        self.logger.info("Hilo TTS iniciado")
    
    def _procesar_cola_tts(self):
        
        while not self.detener_hilo:
            try:
                # Esperar mensaje con timeout
                mensaje_data = self.cola_tts.get(timeout=1.0)
                
                if mensaje_data is None:
                    break
                
                self._reproducir_mensaje(mensaje_data)
                
                # Marcar tarea completada
                self.cola_tts.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error procesando cola TTS: {e}")
    
    def _reproducir_mensaje(self, mensaje_data: Dict[str, Any]):
    
        try:
            texto = mensaje_data.get('texto', '')
            prioridad = mensaje_data.get('prioridad', 1)
            
            if not self._validar_texto(texto):
                return
            
            self.reproduciendo = True
            success = False
            
            # Intentar engine preferido
            if self.engine_preferido == 'pyttsx3':
                success = self._hablar_pyttsx3(texto)
                if not success and self.usar_fallback:
                    success = self._hablar_gtts(texto)
            else:  # gTTS preferido
                success = self._hablar_gtts(texto)
                if not success and self.usar_fallback:
                    success = self._hablar_pyttsx3(texto)
            
            if success:
                self.logger.debug(f"Mensaje reproducido: '{texto[:50]}...' (prioridad: {prioridad})")
            else:
                self.logger.error(f"No se pudo reproducir mensaje: '{texto[:50]}...'")
                
        except Exception as e:
            self.logger.error(f"Error reproduciendo mensaje: {e}")
        finally:
            self.reproduciendo = False
    
    def _hablar_pyttsx3(self, texto: str) -> bool:

        if not self.engine_pyttsx3:
            return False
            
        try:
            self.engine_pyttsx3.say(texto)
            self.engine_pyttsx3.runAndWait()
            return True
        except Exception as e:
            self.logger.error(f"Error con pyttsx3: {e}")
            return False
    
    def _hablar_gtts(self, texto: str) -> bool:
        
        if not GTTS_AVAILABLE or not self.pygame_iniciado:
            return False
            
        try:
            # archivo temporal
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
            
            # Generar audio con gTTS
            tts = gTTS(
                text=texto,
                lang=self.idioma_gtts,
                tld=self.tld_gtts,
                slow=self.velocidad_gtts
            )
            
            tts.save(temp_path)
            
            # Reproducir con pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Esperar a que termine
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                if self.detener_hilo:
                    pygame.mixer.music.stop()
                    break
            
            # Limpiar archivo temporal
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error con gTTS: {e}")
            return False
    
    def _validar_texto(self, texto: str) -> bool:

        if not texto or not isinstance(texto, str):
            return False
        
        texto_limpio = texto.strip()
        
        # longitud
        if len(texto_limpio) < self.longitud_minima:
            return False
            
        if len(texto_limpio) > self.longitud_maxima:
            texto_limpio = texto_limpio[:self.longitud_maxima]
        
        #  palabras ignoradas
        palabras = texto_limpio.lower().split()
        if any(palabra in self.palabras_ignorar for palabra in palabras):
            return False
        
        return True
    
    def hablar(self, texto: str, prioridad: int = 1, limpiar_cola: bool = False):
        
        if not self.habilitado or not texto:
            return
        
        try:
        
            if limpiar_cola:
                self.limpiar_cola()
            
            #  límite de cola
            if self.cola_tts.qsize() >= self.cola_maxima:
                self.logger.warning("Cola TTS llena, descartando mensaje más antiguo")
                try:
                    self.cola_tts.get_nowait()
                except queue.Empty:
                    pass
            
        
            mensaje_data = {
                'texto': texto,
                'prioridad': prioridad,
                'timestamp': time.time()
            }
            
            self.cola_tts.put(mensaje_data)
            self.logger.debug(f"Mensaje añadido a cola TTS: '{texto[:30]}...' (prioridad: {prioridad})")
            
        except Exception as e:
            self.logger.error(f"Error añadiendo mensaje a cola TTS: {e}")
    
    def hablar_detecciones(self, detecciones: List[Dict[str, Any]]):

        if not detecciones:
            return
        
        try:

            if self.limpiar_cola_nueva:
                self.limpiar_cola()
            
            # Procesar detecciones
            mensaje = self._generar_mensaje_detecciones(detecciones)
            
            if mensaje:
                # Prioridad basada en proximidad
                prioridad = 1 if any(det.get('distancia', 100) < 2 for det in detecciones) else 2
                self.hablar(mensaje, prioridad=prioridad)
            
        except Exception as e:
            self.logger.error(f"Error procesando detecciones para TTS: {e}")
    
    def _generar_mensaje_detecciones(self, detecciones: List[Dict[str, Any]]) -> str:
        
        try:
            # Ordenar por distancia si está disponible
            if self.priorizar_cerca:
                detecciones_ordenadas = sorted(
                    detecciones, 
                    key=lambda x: x.get('distancia', 100)
                )
            else:
                detecciones_ordenadas = detecciones
            
            # Agrupar por tipo
            grupos = {}
            for det in detecciones_ordenadas[:5]:  # Máximo 5 detecciones
                tipo = det.get('tipo', 'objeto')
                if tipo not in grupos:
                    grupos[tipo] = []
                grupos[tipo].append(det)
            
            # Generar mensaje
            partes_mensaje = []
            
            for tipo, objetos in grupos.items():
                cantidad = len(objetos)
                
                if cantidad == 1:
                    obj = objetos[0]
                    distancia = obj.get('distancia')
                    if distancia and distancia < 3:
                        partes_mensaje.append(f"{tipo} cerca")
                    else:
                        partes_mensaje.append(tipo)
                else:
                    partes_mensaje.append(f"{cantidad} {tipo}s")
            
            if partes_mensaje:
                if len(partes_mensaje) == 1:
                    return f"Detectado: {partes_mensaje[0]}"
                else:
                    return f"Detectados: {', '.join(partes_mensaje[:-1])} y {partes_mensaje[-1]}"
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Error generando mensaje: {e}")
            return ""
    
    def limpiar_cola(self):
        
        try:
            while not self.cola_tts.empty():
                self.cola_tts.get_nowait()
            self.logger.debug("Cola TTS limpiada")
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error limpiando cola TTS: {e}")
    
    def detener(self):
        
        try:
            self.detener_hilo = True
            
            # Limpiar cola
            self.limpiar_cola()
            
            # Añadir señal de parada
            self.cola_tts.put(None)
            
            # Esperar a que termine el hilo
            if self.hilo_tts and self.hilo_tts.is_alive():
                self.hilo_tts.join(timeout=2)
            
            # Detener pygame
            if self.pygame_iniciado:
                try:
                    pygame.mixer.quit()
                except:
                    pass
            
            self.logger.info("Sistema TTS detenido")
            
        except Exception as e:
            self.logger.error(f"Error deteniendo TTS: {e}")
    
    def esta_reproduciendo(self) -> bool:
        
        return self.reproduciendo
    
    def get_estado(self) -> Dict[str, Any]:
        
        return {
            'habilitado': self.habilitado,
            'engine_preferido': self.engine_preferido,
            'reproduciendo': self.reproduciendo,
            'cola_pendiente': self.cola_tts.qsize(),
            'engines_disponibles': {
                'pyttsx3': self.engine_pyttsx3 is not None,
                'gtts': GTTS_AVAILABLE,
                'pygame': self.pygame_iniciado
            }
        }
    
    def cambiar_engine(self, nuevo_engine: str):

        if nuevo_engine in ['pyttsx3', 'gtts']:
            self.engine_preferido = nuevo_engine
            self.logger.info(f"Engine TTS cambiado a: {nuevo_engine}")
        else:
            self.logger.warning(f"Engine no válido: {nuevo_engine}")

    def ajustar_velocidad(self, nueva_velocidad: int):

        if 50 <= nueva_velocidad <= 300:
            self.velocidad = nueva_velocidad
            if self.engine_pyttsx3:
                try:
                    self.engine_pyttsx3.setProperty('rate', nueva_velocidad)
                    self.logger.info(f"Velocidad TTS ajustada a: {nueva_velocidad}")
                except Exception as e:
                    self.logger.error(f"Error ajustando velocidad: {e}")
        else:
            self.logger.warning(f"Velocidad no válida: {nueva_velocidad}")
    
    def __del__(self):
        
        try:
            self.detener()
        except:
            pass

# Función de utilidad para crear instancia global
_instancia_tts = None

def get_tts_instance(config_manager: Optional[ConfigManager] = None) -> TextoAVoz:
    
    global _instancia_tts
    if _instancia_tts is None:
        _instancia_tts = TextoAVoz(config_manager)
    return _instancia_tts

def limpiar_instancia_tts():

    global _instancia_tts
    if _instancia_tts:
        _instancia_tts.detener()
        _instancia_tts = None