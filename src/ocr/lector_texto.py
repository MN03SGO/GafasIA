import cv2
import numoy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
from typing import List, Dict, Tuple, Optional
import time 

class LectorTexto:
    def __init__(self, idioma: str = 'spa', confianza_minima: int  = 30): # Idioma chaveliado
        print("Inicializando sistema RasVision")
        self.idioma = idioma
        self.confianza_minima = confianza_minima

        self.config_tesseract = f'--oem 3 --psm 6 -L{idioma}'

        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract {version} detectado")
            self.disponible = True
        except Exception as  e:
            print(f"tesseract, no funciona - {e}")
            self.disponible = False
            return
        
        self.patrones_utiles = { #filtrar text, importante del malo
            'direcciones': re.compile(r'\b\w+\s+\d+\b|\b\d+\s+\w+\b', re.IGNORECASE),
            'numeros_telefono': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', re.IGNORECASE),
            'horarios': re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b', re.IGNORECASE),
            'precios': re.compile(r'\$\s*\d+[.,]?\d*|\b\d+[.,]?\d*\s*(pesos?|euros?|dólares?)\b', re.IGNORECASE),
            'palabras_importantes': re.compile(r'\b(abierto|cerrado|entrada|salida|baño|emergencia|peligro|cuidado)\b', re.IGNORECASE)
        }

        self.palabras_ruido = {
            'ruido_ocr': ['|||', '___', '...', '---', '==='],
            'caracteres_sueltos': [r'^\w$', r'^\d$'],  # Una sola letra o número
            'fragmentos': [r'^.{1,2}$']  # Texto  corto (1-2 caracteres)
        }

        print("Sistema OCR de RasVision iniciado...")

        def detectar_texto(self, imagen: np.ndarray, mejorar_imagen: bool = True) -> Lis[Dict]:
            if not self.disponibles:
                print("Sistema OCR, no disponible")
                return[]
            try:
                print("Iniciando deteccion de texto")
                inicio_tiempo = time.time()

                imagen_procesada = self._mejorar_imagen_ocr(imagen) if  mejorar_imagen else imagen
                if len(imagen_procesada.shape) == 3:
                    imagen_pil = Image.fromarray(cv2.cvtColor(imagen_procesada, cv2))
                else:
                    imagen_pil = Image.fromarray(cv2.cvtColor(imagen_procesada))
                    #Obtener datos ocr
                datos_ocr = pytesseract.image_to_data(
                    imagen_pil, 
                    config=self.config_teserract,
                    output_type=pytesseract.Output.DICT
                )
                textos_detectados = self._procesar_resultados_ocr(datos_ocr, imagen.shape)

                tiempo_total = time.time() - inicio_tiempo
                print("OCR, complego en {tiempo_total:.2f}s- {len(textos_detectados)} textos detectados")

                return textos_detectados 
            except Exception as e: 
                print(f"Error en la deteccion de OCR: {e}")
                return[]
            
        def _mejorar_imagen_ocr(self, imagen: np.ndarray) -> np.ndarray:
            try:
                if len(imagen.shape) == 3: 
                    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
                else:
                    imagen_gris = imagen.copy()
                
                    altura, ancho = imagen_gris.shape 
                if min(altura, ancho) < 300: 
                    factor_escala = 300 / min(altura, ancho)
                    nuevo_ancho = int(ancho*factor_escala)
                    nueva_altura = int(altura * factor_escala)
                    imagen_gris = cv2.resize(imagen_gris, (nuevo_ancho, nueva_altura),
                                    interpolation=cv2.INTER_CUBIC)
                    
                    clahe = cv2.createCLAHE(clipLimit= 2.0, tilderGridSize=(8,8))
                    imagen_mejorada = clahe.apply(imagen_gris)

                    imagen_mejorada = cv2.GaussingBlur(imagen_mejorada, (3,3), 0)
                    imagen_binaria = cv2.adaptiveThreshold(
                    imagen_mejorada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11,2
                    )

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                    imagen_final = cv2.morphologyEx(imagen_binaria, cv2.MORPH_OPENING,kernel)
                    return imagen_final
            
            except Exception as e:
                print(f"Error de la imagen: {e}")
                return imagen
            
        def _procesar_resultados_ocr(self, datos_ocr: Dict, forma_imagen: Tuple) -> List[Dict]:
            textos_detectados = []
            altura_img, ancho_img = forma_imagen[:2]
            
            for i in range(len(datos_ocr['text'])):
                textos = datos_ocr['text'][i].strip()
                confianza = int(datos_ocr['conf'][i]) if datos_ocr['conf'][i] != -1 else 0
                
                





                   









        





