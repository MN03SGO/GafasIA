import pyttsx3
import threading
import queue
import time
from typing import Optional, Dict, List
import os

class SintetizadorVoz:
    def __init__(self, idioma: str = 'es', velocidad: int = 180, volumen: float = 0.8):
    
        print("Inicializando sistema de síntesis de voz...")
        
        self.idioma = idioma
        self.velocidad = velocidad
        self.volumen = volumen
        

        self.cola_mensajes = queue.Queue()
        self.reproduciendo = False
        self.hilo_audio = None
        
        # motor TTS
        try:
            self.motor = pyttsx3.init()
            self._configurar_motor()
            self.disponible = True
            print("Motor de síntesis de voz inicializado correctamente")
        except Exception as e:
            print(f"Error al inicializar síntesis de voz: {e}")
            self.disponible = False
            self.motor = None
        
        self.frases_contexto = {
            'inicio': [
                "Sistema de asistencia visual activado",
                "Gafas inteligentes listas",
                "Iniciando análisis visual"
            ],
            'sin_objetos': [
                "No veo objetos cercanos en este momento",
                "El área está despejada",
                "No hay objetos detectados"
            ],
            'multiples_objetos': [
                "Veo varios objetos cerca",
                "Hay múltiples elementos en el área",
                "Detecto varios objetos"
            ],
            'persona_cerca': [
                "Hay una persona cerca",
                "Alguien se encuentra en el área",
                "Detecto a una persona"
            ],
            'error': [
                "Error en el análisis",
                "No puedo procesar la imagen en este momento",
                "Problema técnico detectado"
            ]
        }
        
        # procesamiento de audio
        self._iniciar_hilo_audio()



    # LINEA MODIFICADA jue 02 oct 2025
    #MODIFICA EN LA PC

    # =================================
    def _configurar_motor(self):
        if not self.motor:
            return
        
        try:
            self.motor.setProperty('rate', self.velocidad)
            self.motor.setProperty('volume', self.volumen)
            voces = self.motor.getProperty('voices')
            voz_espanol = self._encontrar_voz_espanol(voces)
            if voz_espanol:
                self.motor.setProperty('voice', voz_espanol)
            else:
                print("No se encontro voz")
        except Exception as e:
            print(f"error de configuracion del TTS en la LINEA 84 {e}")

        # HASTA AQUI TERMINE EL NUEVO MANEJADOR 

    def _encontra_voz_en_espanol(self, voces):
        voces_espanol = []
        for voz in voces:
            voz_lower = voz.name.lower() + "" + voz.id.lower()
            if any(lang in voz_lower for lang in ['spanish', 'español', 'es_', 'spain']):
                voces_espanol.append(voz)
        for voz in voces:
            if 'female' in voz.name.lower() or 'mujer' in voz.name.lower():
                return voz.id
    
    


    def _iniciar_hilo_audio(self):
        
        if self.disponible:
            self.hilo_audio = threading.Thread(target=self._procesar_cola_audio, daemon=True)
            self.hilo_audio.start()
            print("Hilo de audio iniciado")



    # FUNCIOINES AGREGADAS EL 02 DE OCTUBRE 
    def _procesar_cola_audio(self):
    
        while True:
            try:
                mensaje = self.cola_mensajes.get(timeout=1.0)
                if mensaje is None:
                    break
                self._reproducir_mensaje(mensaje)   # LINEA  112 MODIFICADA POR TRUE
                self.cola_mensajes.task_done()
            except queue.Empty:
                return
                
            except Exception as e:
                print(f"Error en procesamiento de audio: {e}")
                self.reproduciendo = False

    def _reproducir_mensaje(self, mensaje):
        self.reproduciendo = True
        try: 
            print(f"Reproduciendo:'{mensaje}' ")
            self.motor.say(mensaje)
            self.motor.runAndWait()
        except Exception as e:
            print(f"Error al reproducir mensaje LINEA 129: {e}")
        finally:
            self.reproduciendo  = False



    
    def decir(self, mensaje: str, prioridad: bool = False):
    
        if not self.disponible:
            print(f"TTS no disponible. Mensaje: {mensaje}")
            return
        
        if not mensaje.strip():
            return
        
        if prioridad:
            self._limpiar_cola()
        
        self.cola_mensajes.put(mensaje)
        print(f" Mensaje encolado: '{mensaje[:50]}...' (Cola: {self.cola_mensajes.qsize()})")

    def _obtener_estado(self) -> Dict:
        return {
            'disponible': self.disponible, 
            'reproduciendo': self.reproduciendo, 
            'mensaje_en_cola': self.cola_mensajes.qsize(),
            'velocidad': self.velocidad,
            'volumen': self.volumen,
            'idioma': self.idioma

        }
    def pausar (self):
        if self.motor:
            self.motor.stop()
    def obtener_voces_disponibles(self) -> List[Dict]:
    
        if not self.motor:
            return []
    
        voces = []
        for voz in self.motor.getProperty('voices'):
            voces.append({
                'id': voz.id,
                'nombre': voz.name,
                'idiomas': getattr(voz, 'languages', [])
            })
        return voces
    
    # TERMINA FUNCIIONES AGREGADAS DE PRUEBA PARA MEJOR MANEJO 







    def decir_detecciones(self, detecciones: List[Dict], incluir_detalles: bool = False):
    
        if not detecciones:
            mensaje = self._obtener_frase_aleatoria('sin_objetos')
            self.decir(mensaje)
            return
        
        # Un solo objeto
        if len(detecciones) == 1:
            det = detecciones[0]
            if incluir_detalles:
                mensaje = f"Veo {det['nombre']} {det['posicion']}, {det['distancia_relativa']}"
            else:
                mensaje = f"Hay {det['nombre']} cerca"
            
    
            if det['clase_id'] == 0:  # persona
                mensaje = self._obtener_frase_aleatoria('persona_cerca')
                if incluir_detalles:
                    mensaje += f", {det['posicion']}"
        
        # Múltiples objetos
        else:
            # Separar personas de objetos
            personas = [d for d in detecciones if d['clase_id'] == 0]
            objetos = [d for d in detecciones if d['clase_id'] != 0]
            
            partes_mensaje = []
            
            # Mencionar personas primero
            if personas:
                if len(personas) == 1:
                    partes_mensaje.append("Hay una persona cerca")
                else:
                    partes_mensaje.append(f"Hay {len(personas)} personas cerca")
            
            # Mencionar objetos
            if objetos:
                if len(objetos) == 1:
                    partes_mensaje.append(f"y veo {objetos[0]['nombre']}")
                elif len(objetos) == 2:
                    partes_mensaje.append(f"y veo {objetos[0]['nombre']} y {objetos[1]['nombre']}")
                else:
                    partes_mensaje.append(f"y veo {objetos[0]['nombre']}, {objetos[1]['nombre']} y {len(objetos)-2} objetos más")
            
            mensaje = " ".join(partes_mensaje) if partes_mensaje else self._obtener_frase_aleatoria('multiples_objetos')
        
        self.decir(mensaje)
    
    def decir_inicio(self):

        mensaje = self._obtener_frase_aleatoria('inicio')
        self.decir(mensaje, prioridad=True)
    
    def decir_error(self, tipo_error: str = "general"):
    
        mensaje = self._obtener_frase_aleatoria('error')
        self.decir(mensaje, prioridad=True)
    
    def _obtener_frase_aleatoria(self, categoria: str) -> str:
    
        import random
        frases = self.frases_contexto.get(categoria, ["Mensaje no disponible"])
        return random.choice(frases)
    
    def _limpiar_cola(self):
        while not self.cola_mensajes.empty():
            try:
                self.cola_mensajes.get_nowait()
                self.cola_mensajes.task_done()
            except queue.Empty:
                break
        print("Cola de audio limpiada")
    
    def esta_hablando(self) -> bool:

        return self.reproduciendo
    
    def esperar_finalizacion(self, timeout: float = 10.0):
    
        try:
            self.cola_mensajes.join()  # Espera a que la cola esté vacía
        except:
            print("Timeout esperando finalización de audio")
    
    def configurar(self, velocidad: Optional[int] = None, volumen: Optional[float] = None):
            # Reconfigura el motor de TTS
        if not self.disponible:
            return
        
        if velocidad is not None:
            self.velocidad = velocidad
            self.motor.setProperty('rate', velocidad)
            self.decir(f"Velocidad ajustada a {velocidad} palabras por minuto", prioridad=True)
        
        if volumen is not None:
            self.volumen = volumen
            self.motor.setProperty('volume', volumen)
            self.decir(f"Volumen ajustado al {int(volumen * 100)} por ciento", prioridad=True)
    
    def probar_voz(self):
    
        mensaje_prueba = "Hola, soy tu asistente RasVision. El sistema de síntesis de voz está funcionando correctamente."
        self.decir(mensaje_prueba, prioridad=True)
    
    def finalizar(self):
        print("Finalizando sistema de síntesis de voz...")
        self.cola_mensajes.put(None)
        
        if self.hilo_audio and self.hilo_audio.is_alive(): # Señal de terminación
            self.hilo_audio.join(timeout=2)
        
        if self.motor:
            try:
                self.motor.stop()
            except:
                pass
        
        print(" Sistema de síntesis de voz finalizado")
    
    def __del__(self):

        self.finalizar()