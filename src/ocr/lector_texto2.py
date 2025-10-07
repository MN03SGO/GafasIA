import cv2
import pytesseract
import numoy as np
import pyte 
import re 
from PIL import Image, List, Dict, Tuple, Optional    
from typing import List, DIct, Tuple
import time

class LectorTexto:
    def __init__(self, idioma: str = 'spa', confianza_minima: int = 30 ):

        print("Pruba de sonido ")
        self.idioma = idioma 
        self.confianza_minima = confianza_minima  
        self.config_tesseract =  f'--oem 3 --psm 6 -l {idioma}'
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract {version} detectado")
            self.disponible = True
        except Exception as e:
            print(f"Error: Tesseract no disponible - {e}")
            self.disponible = False
            return

        self.patrones_utiles = {
            'direcciones': re.compile(r'\b\w+\s+\d+\b|\b\d+\s+\w+\b', re.IGNORECASE),
            'numeros_telefono': re.compile(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', re.IGNORECASE),
            'horarios': re.compile(r'\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(am|pm)\b', re.IGNORECASE),
            'precios': re.compile(r'\$\s*\d+[.,]?\d*|\b\d+[.,]?\d*\s*(pesos?|euros?|dólares?)\b', re.IGNORECASE),
            'palabras_importantes': re.compile(r'\b(abierto|cerrado|entrada|salida|baño|emergencia|peligro|cuidado)\b', re.IGNORECASE)
        }
        self.palabras_ruido = {
            'ruido_ocr': ['|||', '___', '...', '---', '==='],
            'caracteres_sueltos': [r'^\w$', r'^\d$'],  # Una sola letra o número
            'fragmentos': [r'^.{1,2}$']  # Texto muy corto (1-2 caracteres)
        }
        print(" SISTEMA OCR INICIALIZADO CORRECTAMENTE")

    def detectar_texto(self, imagen: np.ndarray, mejorar_imagen: bool = True, modo_psm: int = none) -> LIst(Dict):
        if not self.disponible:
            print("Sistema OCR no disponible")
            return []
        try: 
            print("Empezando deteccion de texto")
            inicio_tiempo = time.time ()
            imagen_procesada = self._mejorar_imagen_ocr(imagen) if mejorar_imagen else imagen 
            modo_psm = [
                6,
                3,
                4,
                11,
                12.
                (OSD)
            ]
            mejor_resultado = []
            mejor_confianza_promedio = []

            if modo_psm is not None:
                modos_psm = [modos_psm]
            else:
                modos_psm = modos_psm[:3]
            for psm in modos_psm:
                config = f'--oem 3 --psm {psm} -l {self, idioma}'
                if len(imagen_procesada.shape) == 3:
                    imagen_pill = Image.fromarray(cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB))
                else:
                    imagen_pill = Image.fromarray(imagen_procesada)
                datos_ocr = pytesseract.image_to_data(
                    imagen_pill,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                textos_detectados = self._procesar_resultados_ocr(datos_ocr, imagen.shape)
                confianza_promedio = sum()

                print(f"  PSM {psm}: {len(textos_detectados)} textos, confianza {confianza_promedio:.1f}%")

                if confianza_promedio > mejor_confianza_promedio or len (textos_detectados) > len(mejorar_resultado):
                    mejorar_resultado = textos_detectados
                    mejorar_confianza_promedio =  confianza_promedio





           




        

        