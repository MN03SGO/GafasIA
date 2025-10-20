import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import re
from typing import List, Dict, Tuple, Optional
import time

class LectorTexto:
    #└[~/Documentos/GafasIA]> date
    #vie 17 oct 2025 02:10:50 CST

    _UMBRAL_Y_AGRUPACION = 10 
    _FACTOR_X_AGRUPACION = 1.5
    _BONIFICACIONES_PRIORIDAD = {
        'palabras_importantes': 50, 'direcciones': 40, 'precios': 35, 'horarios': 30, 'numeros_telefonos': 25, 'titulo_importante': 20,'establecimiento': 15, 'direccion': 15
    }
    _CORRECCIONES_TEXTO = {'rn': 'm', 'n1': 'ñ', '0': 'o', '1': 'l'}
    _PATRONES_UTILES = {
        'direcciones': re.compile(r'\b\w+\s+\d+\b|\b\d+\s+\w+\b', re.IGNORECASE),
        'numeros_telefono': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', re.IGNORECASE),
        'horarios': re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b', re.IGNORECASE),
        'precios': re.compile(r'\$\s*\d+[.,]?\d*|\b\d+[.,]?\d*\s*(pesos?|euros?|dólares?)\b', re.IGNORECASE),
        'palabras_importantes': re.compile(r'\b(abierto|cerrado|entrada|salida|baño|emergencia|peligro|cuidado)\b', re.IGNORECASE)
    }
    _PATRON_RUIDO = {
        'caracteres_sueltos': re.compile(r'^[a-zA-Z0-9]$')
    }

    def __init__(self, idioma: str = 'es', confianza_minima: int = 30, motor: str = 'easyocr', usar_gpu: bool = True): # Cambiarlo a falso cuando lo pase a la raspi
        print("Inicializando sistema OCR...")
        self.idioma = idioma
        self.confianza_minima = confianza_minima
        self.motor = motor 
        self.disponible = False # 'tesseract'/'easyocr'

        if motor == 'tesseract':
            try:
                version = pytesseract.get_tesseract_version()
                print(f"Tesseract {version} detectado")
                self.disponible = True
            except Exception as e:
                print(f"Error: Tesseract no disponible - {e}")
                self.disponible = False
        elif motor == 'easyocr':
            try:
                self.reader = easyocr.Reader([idioma], gpu=usar_gpu)
                
                self.disponible = True
                print("EasyOCR inicializado")
            except Exception as e:
                print(f"Error: EasyOCR no disponible - {e}")
                self.disponible = False
        else:
            print("Motor OCR no reconocido")
            self.disponible = False
        print("Sistema OCR de RasVision iniciado...")

    def detectar_texto(self, imagen: np.ndarray, mejorar_imagen: bool = True, psm: int = 6) -> List[Dict]:
        if not self.disponible:
            print("Sistema OCR no disponible")
            return []

        imagen_procesada = self._mejorar_imagen_ocr(imagen) if mejorar_imagen else imagen
        textos_individuales = []

        if self.motor == 'tesseract':
            config_actual = f'--psm {psm} -l {self.idioma}'
            datos_ocr = pytesseract.image_to_data(
                imagen_procesada,
                config=config_actual,
                output_type=pytesseract.Output.DICT
            )
            textos_individuales = self._procesar_resultados_ocr(datos_ocr, imagen.shape)

        elif self.motor == 'easyocr':
            resultados = self.reader.readtext(imagen_procesada)
            altura_img, ancho_img = imagen.shape[:2]
            for bbox, texto, confianza in resultados:
                if confianza * 100 < self.confianza_minima:
                    continue
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x = int(top_left[0])
                y = int(top_left[1])
                w = int(top_right[0] - top_left[0])
                h = int(bottom_left[1] - top_left[1])
                centro_x = x + w // 2
                centro_y = y + h // 2
                categoria = self._categorizar_texto(texto)
                textos_individuales.append({
                    'texto': self._limpiar_texto(texto),
                    'texto_original': texto,
                    'confianza': int(confianza * 100),
                    'posicion': self._calcular_posicion_texto(centro_x, centro_y, ancho_img, altura_img),
                    'categoria': categoria,
                    'coordenadas': {'x': x, 'y': y, 'ancho': w, 'alto': h},
                    'centro': {'x': centro_x, 'y': centro_y},
                    'prioridad': self._calcular_prioridad(texto, categoria, int(confianza * 100))
                })

        textos_agrupados = self._agrupar_texto_cercano(textos_individuales, imagen.shape)
        return textos_agrupados

    def _mejorar_imagen_ocr(self, imagen: np.ndarray) -> np.ndarray:
        try:
            if len(imagen.shape) == 3:
                imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                imagen_gris = imagen.copy()
            altura, ancho = imagen_gris.shape
            if min(altura, ancho) < 300:
                factor_escala = 300 / min(altura, ancho)
                nuevo_ancho = int(ancho * factor_escala)
                nueva_altura = int(altura * factor_escala)
                imagen_gris = cv2.resize(imagen_gris, (nuevo_ancho, nueva_altura), interpolation=cv2.INTER_CUBIC)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            imagen_mejorada = clahe.apply(imagen_gris)
            imagen_mejorada = cv2.GaussianBlur(imagen_mejorada, (3, 3), 0)
            imagen_binaria = cv2.adaptiveThreshold(
                imagen_mejorada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
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
            x, y, w, h = datos_ocr['left'][i], datos_ocr['top'][i], datos_ocr['width'][i], datos_ocr['height'][i]
            centro_x = x + w // 2
            centro_y = y + h // 2
            categoria = self._categorizar_texto(texto)
            textos_detectados.append({
                'texto': self._limpiar_texto(texto),
                'texto_original': texto,
                'confianza': confianza,
                'posicion': self._calcular_posicion_texto(centro_x, centro_y, ancho_img, altura_img),
                'categoria': categoria,
                'coordenadas': {'x': x, 'y': y, 'ancho': w, 'alto': h},
                'centro': {'x': centro_x, 'y': centro_y},
                'prioridad': self._calcular_prioridad(texto, categoria, confianza)
            })
        textos_detectados.sort(key=lambda x: x['prioridad'], reverse=True)
        return textos_detectados

    def _agrupar_texto_cercano(self, textos: List[Dict], forma_imagen: Tuple) -> List[Dict]:
        if not textos: return []
        textos.sort(key=lambda t: (t['coordenadas']['y'], t['coordenadas']['x']))
        grupos = []
        if not textos: return []
        grupo_actual = textos[0]

        for i in range(1, len(textos)):
            prev_coords = textos[i-1]['coordenadas']
            curr_coords = textos[i]['coordenadas']
            misma_linea = abs(curr_coords['y'] - prev_coords['y']) < self._UMBRAL_Y_AGRUPACION
            espacio_esperado = prev_coords['ancho'] * self._FACTOR_X_AGRUPACION
            es_contiguo = (curr_coords['x'] - (prev_coords['x'] + prev_coords['ancho'])) < espacio_esperado
            if misma_linea and es_contiguo:
                grupo_actual['texto'] += " " + textos[i]['texto']
                grupo_actual['confianza'] = (grupo_actual['confianza'] + textos[i]['confianza']) / 2
                grupo_actual['coordenadas']['ancho'] = (curr_coords['x'] + curr_coords['ancho']) - grupo_actual['coordenadas']['x']
            else:
                grupos.append(grupo_actual)
                grupo_actual = textos[i]
        grupos.append(grupo_actual)

        # prioridad
        for grupo in grupos:
            altura_img, ancho_img = 480, 640
            coords = grupo['coordenadas']
            centro_x = coords['x'] + coords['ancho'] / 2
            centro_y = coords['y'] + coords['alto'] / 2
            grupo['posicion'] = self._calcular_posicion_texto(centro_x, centro_y, ancho_img, altura_img)
            grupo['categoria'] = self._categorizar_texto(grupo['texto'])
            grupo['prioridad'] = self._calcular_prioridad(grupo['texto'], grupo['categoria'], int(grupo['confianza']))
        grupos.sort(key=lambda x: x['prioridad'], reverse=True)
        return grupos

    def _es_ruido(self, texto: str) -> bool:
        if self._PATRON_RUIDO['caracteres_sueltos'].match(texto):
            return True
        simbolos  = sum(1 for c in texto if not c.isalnum() and c !='')
        return simbolos > len(textos)* 0.7

    def _categorizar_texto(self, texto: str) -> str:
        texto_lower = texto.lower()
        for categoria, patron in self._PATRONES_UTILES.items():
            if patron.search(texto_lower):
                return categoria
        if len(texto) > 50: return 'texto_largo'
        if any(palabra in texto_lower for palabra in ['calle', 'avenida', 'diagonal']): return 'direccion'
        if any(palabra in texto_lower for palabra in ['tienda', 'restaurante', 'farmacia']): return 'establecimiento'
        if texto.isupper() and len(texto) > 3: return 'titulo_importante'
        return 'texto_general'

    def _calcular_prioridad(self, texto: str, categoria: str, confianza: int) -> int: 
        prioridad = confianza + self._BONIFICACIONES_PRIORIDAD.get(categoria, 0)
        if 5 <= len(texto) <= 30:
            prioridad += 10
        if len(texto) > 100:
            prioridad -= 20
        return prioridad

    def _limpiar_texto(self, texto: str) -> str:
        texto_limpio = ' '.join(texto.split())
        for error, correc in self._CORRECCIONES_TEXTO.items():
            patron = rf'(?<=[a-zA-ZáéíóúñÑ]){re.escape(error)}(?=[a-zA-ZáéíóúñÑ])'
            texto_limpio = re.sub(patron, correc, texto_limpio)
        return texto_limpio

    def _calcular_posicion_texto(self, centro_x: float, centro_y: float, ancho: int, alto: int) -> str:
        tercio_ancho = ancho / 3
        tercio_alto = alto / 3

        if centro_x < tercio_ancho: horizontal = "izquierda"
        elif centro_x < 2 * tercio_ancho: horizontal = "centro"
        else: horizontal = "derecha"
        
        if centro_y < tercio_alto: vertical = "arriba"
        elif centro_y < 2 * tercio_alto: vertical = "medio"
        else: vertical = "abajo"
        
        if horizontal == "centro" and vertical == "medio": return "en el centro"
        if horizontal == "centro": return f"en la parte {vertical}"
        if vertical == "medio": return f"a la {horizontal}"
        return f"{vertical} a la {horizontal}"

    def generar_descripcion_audio(self, textos: List[Dict], modo: str = 'resumen') -> str:
        if not textos: return "No se detectó texto legible en la imagen"
        
        if modo == 'prioritario':
            return f"Texto detectado: {textos[0]['texto']}"
        elif modo == 'resumen':
            num_textos = len(textos)
            if num_textos == 1: return f"Detecto el texto: {textos[0]['texto']}"
            if num_textos <= 3:
                textos_str = ", ".join([t['texto'] for t in textos])
                return f"Detecto los textos: {textos_str}"
            return f"Detecto {num_textos} textos. El principal es: {textos[0]['texto']}"
        else:
            descripciones = [f"{i+1}: {t['texto']} {t['posicion']}" for i, t in enumerate(textos[:5])]
            base_str = f"Textos detectados: {'. '.join(descripciones)}"
            if len(textos) > 5:
                base_str += f" y {len(textos) - 5} más"
            return base_str

    def dibujar_texto_detectado(self, imagen: np.ndarray, textos: List[Dict]) -> np.ndarray:
        imagen_resultado = imagen.copy()
        for texto_info in textos:
            coords = texto_info['coordenadas']
            x, y, w, h = coords['x'], coords['y'], coords['ancho'], coords['alto']
            
            prioridad = texto_info['prioridad']
            if prioridad > 80: color = (0, 255, 0) # Verde
            elif prioridad > 50: color = (0, 255, 255) # Amarillo
            else: color = (255, 0, 0) # Azul
            
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 2)
            
            texto_mostrar = texto_info['texto'][:30]
            if len(texto_info['texto']) > 30: texto_mostrar += "..."
            (w_texto, h_texto), _ = cv2.getTextSize(texto_mostrar, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(imagen_resultado, (x, y - h_texto - 5), (x + w_texto, y), color, -1)
            cv2.putText(imagen_resultado, texto_mostrar, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return imagen_resultado