# src/ocr/lector_texto.py
import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from typing import List, Dict, Tuple, Optional
import time

class LectorTexto:
    def __init__(self, idioma: str = 'spa', confianza_minima: int = 30):
        print("Inicializando sistema OCR...")
        
        self.idioma = idioma
        self.confianza_minima = confianza_minima
        self.config_tesseract = f'--oem 3  -l {idioma}'
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract {version} detectado")
            self.disponible = True
        except Exception as e:
            print(f"Error: Tesseract no disponible - {e}")
            self.disponible = False
            return
        
        # Patrones para filtrar texto útil vs ruido
        self.patrones_utiles = {
            'direcciones': re.compile(r'\b\w+\s+\d+\b|\b\d+\s+\w+\b', re.IGNORECASE),
            'numeros_telefono': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', re.IGNORECASE),
            'horarios': re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b', re.IGNORECASE),
            'precios': re.compile(r'\$\s*\d+[.,]?\d*|\b\d+[.,]?\d*\s*(pesos?|euros?|dólares?)\b', re.IGNORECASE),
            'palabras_importantes': re.compile(r'\b(abierto|cerrado|entrada|salida|baño|emergencia|peligro|cuidado)\b', re.IGNORECASE)
        }
        self.palabras_ruido = { 'caracteres_sueltos': re.compile(r'^[a-zA-Z0-9]$') }
        
        print("Sistema OCR de RasVision iniciado...")

    def detectar_texto(self, imagen: np.ndarray, mejorar_imagen: bool = True, psm: int = 6) -> List[Dict]:
        if not self.disponible:
            print("Sistema OCR no disponible")
            return []
        
        try:
            imagen_procesada = self._mejorar_imagen_ocr(imagen) if mejorar_imagen else imagen
            
            # <-- MODIFICADO: Se puede especificar el Page Segmentation Mode (PSM)
            config_actual = f'--psm {psm} {self.config_tesseract_base}'
            
            datos_ocr = pytesseract.image_to_data(
                imagen_procesada,
                config=config_actual,
                output_type=pytesseract.Output.DICT
            )
            
            textos_individuales = self._procesar_resultados_ocr(datos_ocr, imagen.shape)
            
            # <-- NUEVO: Agrupamos las palabras individuales en líneas coherentes
            textos_agrupados = self._agrupar_texto_cercano(textos_individuales)
            
            return textos_agrupados
        except Exception as e:
            print(f"Error crítico en detección OCR: {e}")
            return []
    
    def _mejorar_imagen_ocr(self, imagen: np.ndarray) -> np.ndarray:
        try:
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris = imagen.copy()
            altura, ancho = imagen_gris.shape
            if min(altura, ancho) < 300:
                factor_escala = 300 / min(altura, ancho)
                nuevo_ancho = int(ancho * factor_escala)
                nueva_altura = int(altura * factor_escala)
                imagen_gris = cv2.resize(imagen_gris, (nuevo_ancho, nueva_altura), 
                                    interpolation=cv2.INTER_CUBIC)
            # Mejorar contraste usando CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            imagen_mejorada = clahe.apply(imagen_gris)
            # Reducir ruido con filtro Gaussiano suave
            imagen_mejorada = cv2.GaussianBlur(imagen_mejorada, (3, 3), 0)
            # Binarización adaptativa para diferentes condiciones de iluminación
            imagen_binaria = cv2.adaptiveThreshold(
                imagen_mejorada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            # Operación de apertura para limpiar ruido pequeño
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            imagen_final = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPEN, kernel)
            return imagen_final
        except Exception as e:
            print(f"Error en mejora de imagen: {e}")
            return imagen
    def _procesar_resultados_ocr(self, datos_ocr: Dict, forma_imagen: Tuple) -> List[Dict]:
        textos_detectados = []
        altura_img, ancho_img = forma_imagen[:2]
        for i in range(len(datos_ocr['text'])):
            texto = datos_ocr['text'][i].strip()
            confianza = int(datos_ocr['conf'][i]) if datos_ocr['conf'][i] != -1 else 0
    
            if confianza < self.confianza_minima:
                continue
            if len(texto) < 2:
                continue
            if self._es_ruido(texto):
                continue
            # Obtener coordenadas
            x, y, w, h = (datos_ocr['left'][i], datos_ocr['top'][i], 
                        datos_ocr['width'][i], datos_ocr['height'][i])
            # posición relativa
            centro_x = x + w // 2
            centro_y = y + h // 2
            posicion = self._calcular_posicion_texto(centro_x, centro_y, ancho_img, altura_img)
            categoria = self._categorizar_texto(texto)
            texto_limpio = self._limpiar_texto(texto)
            deteccion_texto = {
                'texto': texto_limpio,
                'texto_original': texto,
                'confianza': confianza,
                'posicion': posicion,
                'categoria': categoria,
                'coordenadas': {'x': x, 'y': y, 'ancho': w, 'alto': h},
                'centro': {'x': centro_x, 'y': centro_y},
                'prioridad': self._calcular_prioridad(texto_limpio, categoria, confianza)
            }
            
            textos_detectados.append(deteccion_texto)
        
        textos_detectados.sort(key=lambda x: x['prioridad'], reverse=True)
        return textos_detectados
    def _es_ruido(self, texto: str) -> bool:
        for patron_ruido in self.palabras_ruido['ruido_ocr']:
            if patron_ruido in texto:
                return True
        for patron in self.palabras_ruido['caracteres_sueltos']:
            if re.match(patron, texto):
                return True
        for patron in self.palabras_ruido['fragmentos']:
            if re.match(patron, texto):
                return True
        
        #  texto con simbolos 
        simbolos = sum(1 for c in texto if not c.isalnum() and c != ' ')
        if simbolos > len(texto) * 0.7:  # Más del 70% son símbolos
            return True
        
        return False
    
    def _categorizar_texto(self, texto: str) -> str:
        texto_lower = texto.lower()
        for categoria, patron in self.patrones_utiles.items():
            if patron.search(texto):
                return categoria
        if len(texto) > 50:
            return 'texto_largo'
        elif any(palabra in texto_lower for palabra in ['calle', 'avenida', 'boulevard']):
            return 'direccion'
        elif any(palabra in texto_lower for palabra in ['tienda', 'restaurante', 'farmacia']):
            return 'establecimiento'
        elif texto.isupper() and len(texto) > 3:
            return 'titulo_importante'
        else:
            return 'texto_general'
    
    def _calcular_prioridad(self, texto: str, categoria: str, confianza: int) -> int:
        prioridad = confianza  #confianza del OCR
        bonificaciones = {
            'palabras_importantes': 50,
            'direcciones': 40,
            'precios': 35,
            'horarios': 30,
            'numeros_telefono': 25,
            'titulo_importante': 20,
            'establecimiento': 15,
            'direccion': 15
        }
        
        prioridad += bonificaciones.get(categoria, 0)
        if 5 <= len(texto) <= 30:
            prioridad += 10
        if len(texto) > 100:
            prioridad -= 20
        
        return prioridad
    
    def _limpiar_texto(self, texto: str) -> str:
        texto_limpio = ' '.join(texto.split())
        #  errores comunes de OCR en español
        correcciones = {
            'rn': 'm',  # OCR confunde 'rn' con 'm'
            'n1': 'ñ',  # OCR puede confundir 'ñ'
            '0': 'o',   # En contextos de palabras, 0 es probablemente 'o'
            '1': 'l',   # En contextos de palabras, 1 es probablemente 'l'
        }
        for error, correccion in correcciones.items():
            patron = rf'(?<=[a-zA-ZáéíóúñÑ]){re.escape(error)}(?=[a-zA-ZáéíóúñÑ])'
            texto_limpio = re.sub(patron, correccion, texto_limpio)
        
        return texto_limpio
    
    def _calcular_posicion_texto(self, centro_x: int, centro_y: int, 
                            ancho: int, alto: int) -> str:
        tercio_ancho = ancho / 3
        tercio_alto = alto / 3
        if centro_x < tercio_ancho:
            horizontal = "izquierda"
        elif centro_x < 2 * tercio_ancho:
            horizontal = "centro"
        else:
            horizontal = "derecha"
        if centro_y < tercio_alto:
            vertical = "arriba"
        elif centro_y < 2 * tercio_alto:
            vertical = "medio"
        else:
            vertical = "abajo"
        if horizontal == "centro" and vertical == "medio":
            return "en el centro"
        elif horizontal == "centro":
            return f"en la parte {vertical}"
        elif vertical == "medio":
            return f"a la {horizontal}"
        else:
            return f"{vertical} a la {horizontal}"
    
    def generar_descripcion_audio(self, textos: List[Dict], 
                                modo: str = 'resumen') -> str:
        if not textos:
            return "No se detectó texto legible en la imagen"
        
        if modo == 'prioritario':
            # Solo el texto mas importante
            texto_principal = textos[0]
            return f"Texto detectado: {texto_principal['texto']}"
        
        elif modo == 'resumen':
            if len(textos) == 1:
                return f"Detecto el texto: {textos[0]['texto']}"
            elif len(textos) <= 3:
                textos_str = ", ".join([t['texto'] for t in textos])
                return f"Detecto los textos: {textos_str}"
            else:
                texto_principal = textos[0]['texto']
                return f"Detecto {len(textos)} textos. El principal es: {texto_principal}"
        else:
            descripciones = []
            for i, texto in enumerate(textos[:5], 1):  # Máximo 5 textos
                descripciones.append(f"{texto['texto']} {texto['posicion']}")
            
            if len(textos) <= 5:
                return f"Textos detectados: {'. '.join(descripciones)}"
            else:
                return f"Textos detectados: {'. '.join(descripciones)} y {len(textos)-5} más"
    
    def dibujar_texto_detectado(self, imagen: np.ndarray, 
                            textos: List[Dict]) -> np.ndarray:
        imagen_resultado = imagen.copy()
        
        for i, texto_info in enumerate(textos):
            coords = texto_info['coordenadas']
            x, y, w, h = coords['x'], coords['y'], coords['ancho'], coords['alto']
            
            # Color basado en prioridad
            if texto_info['prioridad'] > 80:
                color = (0, 255, 0)  # Verde para alta prioridad
            elif texto_info['prioridad'] > 50:
                color = (0, 255, 255)  # Amarillo para media prioridad
            else:
                color = (255, 0, 0)  # Azul para baja prioridad
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 2)
            # texto (limitado a 30 caracteres)
            texto_mostrar = texto_info['texto'][:30]
            if len(texto_info['texto']) > 30:
                texto_mostrar += "..."
            etiqueta = f"{i+1}: {texto_mostrar} ({texto_info['confianza']}%)"
            (w_texto, h_texto), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(imagen_resultado, (x, y - h_texto - 5), (x + w_texto, y), color, -1)
            
            # Texto
            cv2.putText(imagen_resultado, etiqueta, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return imagen_resultado
    def leer_texto_continuo(self, imagen: np.ndarray, 
                        velocidad_lectura: str = 'normal') -> List[str]:
        textos = self.detectar_texto(imagen, mejorar_imagen=True)
        
        if not textos:
            return ["No se detectó texto en el documento"]
        textos_ordenados = sorted(textos, key=lambda x: (x['centro']['y'], x['centro']['x']))
        fragmentos = []
        fragmento_actual = []
        
        for texto in textos_ordenados:
            fragmento_actual.append(texto['texto'])

            texto_fragmento = ' '.join(fragmento_actual)
            if len(texto_fragmento) > 100 or texto['texto'].endswith('.'):
                fragmentos.append(texto_fragmento)
                fragmento_actual = []
        if fragmento_actual:
            fragmentos.append(' '.join(fragmento_actual))
        
        return fragmentos if fragmentos else ["No se pudo procesar el texto del documento"]