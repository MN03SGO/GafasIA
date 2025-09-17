# src/audio/sintetizador_voz.py
import pyttsx3
import threading
import queue
import time
from typing import Optional, Dict, List
import os

class SintetizadorVoz:
    def __init__(self, idioma: str = 'es', velocidad: int = 180, volumen: float = 0.8):
        """
        Inicializa el sistema de síntesis de voz
        
        Args:
            idioma: Código del idioma ('es' para español)
            velocidad: Velocidad de habla en palabras por minuto (150-200 recomendado)
            volumen: Volumen de 0.0 a 1.0
        """
        print("🔊 Inicializando sistema de síntesis de voz...")
        
        # Configuración inicial
        self.idioma = idioma
        self.velocidad = velocidad
        self.volumen = volumen
        
        # Cola para mensajes de audio (permite encolar múltiples mensajes)
        self.cola_mensajes = queue.Queue()
        self.reproduciendo = False
        self.hilo_audio = None
        
        # Inicializar motor TTS
        try:
            self.motor = pyttsx3.init()
            self._configurar_motor()
            self.disponible = True
            print("✅ Motor de síntesis de voz inicializado correctamente")
        except Exception as e:
            print(f"❌ Error al inicializar síntesis de voz: {e}")
            self.disponible = False
            self.motor = None
        
        # Frases pre-definidas para mayor naturalidad
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
        
        # Iniciar hilo de procesamiento de audio
        self._iniciar_hilo_audio()
    
    def _configurar_motor(self):
        """Configura el motor de TTS con las preferencias del usuario"""
        if not self.motor:
            return
        
        # Configurar velocidad
        self.motor.setProperty('rate', self.velocidad)
        
        # Configurar volumen
        self.motor.setProperty('volume', self.volumen)
        
        # Intentar configurar voz en español
        voces = self.motor.getProperty('voices')
        voz_espanol = None
        
        for voz in voces:
            # Buscar voz en español
            if 'spanish' in voz.name.lower() or 'es' in voz.id.lower():
                voz_espanol = voz.id
                break
            # En algunos sistemas, buscar por país
            elif any(pais in voz.id.lower() for pais in ['es_', 'mx_', 'ar_', 'co_']):
                voz_espanol = voz.id
                break
        
        if voz_espanol:
            self.motor.setProperty('voice', voz_espanol)
            print(f"✅ Voz en español configurada: {voz_espanol}")
        else:
            print("⚠️ No se encontró voz en español, usando voz por defecto")
            # Listar voces disponibles para debugging
            print("Voces disponibles:")
            for i, voz in enumerate(voces[:3]):  # Mostrar solo las primeras 3
                print(f"  {i}: {voz.name} ({voz.id})")
    
    def _iniciar_hilo_audio(self):
        """Inicia el hilo separado para procesamiento de audio"""
        if self.disponible:
            self.hilo_audio = threading.Thread(target=self._procesar_cola_audio, daemon=True)
            self.hilo_audio.start()
            print("✅ Hilo de audio iniciado")
    
    def _procesar_cola_audio(self):
        """Procesa los mensajes de audio en un hilo separado"""
        while True:
            try:
                # Obtener mensaje de la cola (bloquea hasta que haya uno)
                mensaje = self.cola_mensajes.get()
                
                if mensaje is None:  # Señal para terminar el hilo
                    break
                
                self.reproduciendo = True
                print(f"🔊 Reproduciendo: '{mensaje}'")
                
                # Sintetizar y reproducir
                self.motor.say(mensaje)
                self.motor.runAndWait()
                
                self.reproduciendo = False
                
                # Marcar tarea como completada
                self.cola_mensajes.task_done()
                
                # Pequeña pausa entre mensajes
                time.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error en procesamiento de audio: {e}")
                self.reproduciendo = False
    
    def decir(self, mensaje: str, prioridad: bool = False):
        """
        Envía un mensaje para síntesis de voz
        
        Args:
            mensaje: Texto a sintetizar
            prioridad: Si True, vacía la cola y reproduce inmediatamente
        """
        if not self.disponible:
            print(f"⚠️ TTS no disponible. Mensaje: {mensaje}")
            return
        
        if not mensaje.strip():
            return
        
        # Si es prioritario, limpiar cola
        if prioridad:
            self._limpiar_cola()
        
        # Añadir a la cola
        self.cola_mensajes.put(mensaje)
        print(f"📝 Mensaje encolado: '{mensaje[:50]}...' (Cola: {self.cola_mensajes.qsize()})")
    
    def decir_detecciones(self, detecciones: List[Dict], incluir_detalles: bool = False):
        """
        Convierte detecciones del detector de objetos a mensaje de voz natural
        
        Args:
            detecciones: Lista de detecciones del DetectorObjetos
            incluir_detalles: Si incluir información de posición y distancia
        """
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
            
            # Caso especial para personas
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
        """Mensaje de inicio del sistema"""
        mensaje = self._obtener_frase_aleatoria('inicio')
        self.decir(mensaje, prioridad=True)
    
    def decir_error(self, tipo_error: str = "general"):
        """Mensaje de error"""
        mensaje = self._obtener_frase_aleatoria('error')
        self.decir(mensaje, prioridad=True)
    
    def _obtener_frase_aleatoria(self, categoria: str) -> str:
        """Obtiene una frase aleatoria de una categoría"""
        import random
        frases = self.frases_contexto.get(categoria, ["Mensaje no disponible"])
        return random.choice(frases)
    
    def _limpiar_cola(self):
        """Vacía la cola de mensajes pendientes"""
        while not self.cola_mensajes.empty():
            try:
                self.cola_mensajes.get_nowait()
                self.cola_mensajes.task_done()
            except queue.Empty:
                break
        print("🗑️ Cola de audio limpiada")
    
    def esta_hablando(self) -> bool:
        """Verifica si el sistema está reproduciendo audio"""
        return self.reproduciendo
    
    def esperar_finalizacion(self, timeout: float = 10.0):
        """
        Espera a que se procesen todos los mensajes en cola
        
        Args:
            timeout: Tiempo máximo de espera en segundos
        """
        try:
            self.cola_mensajes.join()  # Espera a que la cola esté vacía
        except:
            print("⚠️ Timeout esperando finalización de audio")
    
    def configurar(self, velocidad: Optional[int] = None, volumen: Optional[float] = None):
        """
        Reconfigura el motor de TTS
        
        Args:
            velocidad: Nueva velocidad (palabras por minuto)
            volumen: Nuevo volumen (0.0 a 1.0)
        """
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
        """Prueba el sistema con un mensaje de ejemplo"""
        mensaje_prueba = "Hola, soy tu asistente visual. El sistema de síntesis de voz está funcionando correctamente."
        self.decir(mensaje_prueba, prioridad=True)
    
    def finalizar(self):
        """Finaliza el sistema de síntesis de voz"""
        print("🔄 Finalizando sistema de síntesis de voz...")
        
        # Señal para terminar el hilo
        if self.hilo_audio and self.hilo_audio.is_alive():
            self.cola_mensajes.put(None)  # Señal de terminación
            self.hilo_audio.join(timeout=2)
        
        # Finalizar motor
        if self.motor:
            try:
                self.motor.stop()
            except:
                pass
        
        print("✅ Sistema de síntesis de voz finalizado")
    
    def __del__(self):
        """Destructor para limpiar recursos"""
        self.finalizar()
