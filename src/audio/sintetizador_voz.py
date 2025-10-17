import pyttsx3
import threading
import queue
import time
import random
from typing import Optional, Dict, List
import os

class SintetizadorVoz:
    #└[~/Documentos/GafasIA]> date
    #vie 17 oct 2025 02:10:50 CST
    def __init__(self, idioma: str = 'es', velocidad: int = 180, volumen: float = 0.8):

        self.idioma = idioma
        self.velocidad = velocidad
        self.volumen = volumen
        self.cola_mensajes = queue.Queue()
        self.reproduciendo = False
        self.hilo_audio = None
        try:
            self.motor = pyttsx3.init()
            self._configurar_motor()
            self.disponible = True
            print("Motor voz inicializado correctamente")
        except Exception as e:
            print(f"Error al inicializar síntesis de voz: {e}")
            self.disponible = False
            self.motor = None
        self.frases_contexto = {
            'inicio': [
                "Rasvision activado",
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
        
        self._iniciar_hilo_audio()
    
    def _configurar_motor(self):
        if not self.motor:
            return
        self.motor.setProperty('rate', self.velocidad)
        self.motor.setProperty('volume', self.volumen)
        voces = self.motor.getProperty('voices')
        voz_espanol = None
        for voz in voces:
            if 'spanish' in voz.name.lower() or 'es-es' in voz.id.lower() or 'es-mx' in voz.id.lower():
                voz_espanol = voz.id
                break
        if voz_espanol:
            self.motor.setProperty('voice', voz_espanol)
            print(f"Voz en español configurada: {voz_espanol}")
        else:
            print("No se encontró voz en español, usando voz por defecto")
    
    def _iniciar_hilo_audio(self):
        if self.disponible:
            self.hilo_audio = threading.Thread(target=self._procesar_cola_audio, daemon=True)
            self.hilo_audio.start()
            print("Hilo de audio iniciado")

    def _procesar_cola_audio(self):
        while True:
            try:
                mensaje = self.cola_mensajes.get()
                if mensaje is None:
                    break
                self.reproduciendo = True
                print(f"Reproduciendo: '{mensaje}'")
                self.motor.say(mensaje)
                self.motor.runAndWait()
                self.reproduciendo = False
                self.cola_mensajes.task_done()
                time.sleep(0.5)
            except Exception as e:
                print(f"Error en procesamiento de audio: {e}")
                self.reproduciendo = False
    
    def decir(self, mensaje: str, prioridad: bool = False):
        if not self.disponible:
            print(f"TTS no disponible. Mensaje: {mensaje}")
            return
        if not mensaje.strip():
            return
        if prioridad:
            self._limpiar_cola()
        self.cola_mensajes.put(mensaje)
        print(f"Mensaje encolado: '{mensaje[:50]}...' (Cola: {self.cola_mensajes.qsize()})")
    
    def decir_detecciones(self, detecciones: List[Dict], incluir_detalles: bool = False):
        if not detecciones:
            self.decir(self._obtener_frase_aleatoria('sin_objetos'), prioridad=True)
            return

        detecciones.sort(key=lambda x: x.get('prioridad', 0), reverse=True)
        if len(detecciones) == 1:
            det = detecciones[0]
            if det['clase_id'] == 0: # Es una persona
                mensaje = self._obtener_frase_aleatoria('persona_cerca')
                if incluir_detalles:
                    mensaje += f", {det['posicion']}"
            else:
                mensaje = f"Veo {det['nombre']} {det['posicion']}"
                if 'distancia_relativa' in det:
                    mensaje += f", {det['distancia_relativa']}"
            self.decir(mensaje, prioridad=True)
            return
        
        personas = [d for d in detecciones if d['clase_id'] == 0]
        objetos = [d for d in detecciones if d['clase_id'] != 0]
        
        partes_mensaje = []
        if personas:
            num_personas = len(personas)
            partes_mensaje.append(f"{'Hay una persona' if num_personas == 1 else f'Hay {num_personas} personas'} cerca")
        
        if objetos:
            if len(objetos) == 1:
                partes_mensaje.append(f"y veo {objetos[0]['nombre']}")
            elif len(objetos) == 2:
                partes_mensaje.append(f"y veo {objetos[0]['nombre']} y {objetos[1]['nombre']}")
            else:
                partes_mensaje.append(f"y veo {objetos[0]['nombre']}, {objetos[1]['nombre']} y otros objetos")
        
        mensaje = ", ".join(partes_mensaje) if partes_mensaje else self._obtener_frase_aleatoria('multiples_objetos')
        self.decir(mensaje, prioridad=True)

    
    def decir_inicio(self):
        self.decir(self._obtener_frase_aleatoria('inicio'), prioridad=True)
    
    def decir_error(self, tipo_error: str = "general"):
        self.decir(self._obtener_frase_aleatoria('error'), prioridad=True)
    
    def _obtener_frase_aleatoria(self, categoria: str) -> str:
        return random.choice( self.frases_contexto.get(categoria, ["Mensaje no disponible"]))
    
    def _limpiar_cola(self):
        with self.cola_mensajes.mutex:
            self.cola_mensajes.queue.clear()
        print("Cola de audio limpiada.")
    
    def esta_hablando(self) -> bool:
        return self.reproduciendo or not self.cola_mensajes.empty()
    
    def esperar_finalizacion(self, timeout: float = 10.0):
        self.cola_mensajes.join()
    
    def configurar(self, velocidad: Optional[int] = None, volumen: Optional[float] = None):
        if not self.disponible: return
        
        if velocidad is not None:
            self.velocidad = velocidad
            self.motor.setProperty('rate', velocidad)
            self.decir(f"Velocidad ajustada.", prioridad=True)
        
        if volumen is not None:
            volumen_clamp = max(0.0, min(1.0, volumen))
            self.volumen = volumen_clamp
            self.motor.setProperty('volume', volumen_clamp)
            self.decir(f"Volumen ajustado.", prioridad=True)
    
    def probar_voz(self):
        mensaje_prueba = "Soy pepito. El sistema de síntesis de voz está funcionando correctamente."
        self.decir(mensaje_prueba, prioridad=True)
    
    def finalizar(self):
        print("Finalizando sistema de síntesis de voz...")
        if self.hilo_audio and self.hilo_audio.is_alive():
            self.cola_mensajes.put(None)  
            self.hilo_audio.join(timeout=2)
        
        if self.motor:
            try: self.motor.stop()
            except: pass
        print("Sistema de síntesis de voz finalizado.")

    def __del__(self):
        self.finalizar()