import cv2
from cv2.ocl import KernelArg_PTR_ONLY
import pytesseract
import numpy as np 
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

    def detectar_texto(self, imagen: np.ndarray, mejorar_imagen: bool = True, modo_psm: int = None) -> List[Dict]:
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
                
            ]
            mejor_resultado = []
            mejor_confianza_promedio = 0

            if modo_psm is not None:
                modos_psm = [modos_psm]
            else:
                modos_psm = modos_psm[:3]
            for psm in modos_psm:
                config = f'--oem 3 --psm {psm} -l {self.idi}'
                if len(imagen_procesada.shape) == 3:
                    imagen_pill = Image.fromarray(cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2RGB))
                else:
                    imagen_pill = Image.fromarray(imagen_procesada)
                try:
                    datos_ocr = pytesseract.image_to_data(
                        imagen_pill,
                        config=config,
                        output_type=pytesseract.Output.DICT
                    )
                    textos_detectados = self._procesar
                    textos_detectados = self._procesar_resultados_ocr(datos_ocr, imagen.shape)
                    if textos_detectados: 
                        confianza_promedio = sum(t['confianza'] for t in textos_detectados) / len(textos_detectados)
                        print(f"  PSM {psm}: {len(textos_detectados)} textos, confianza {confianza_promedio:.1f}%")
                        if confianza_promedio > mejor_confianza_promedio or len(textos_detectados) > len(mejor_resultado):
                            mejor_resultado = textos_detectados
                            mejor_confianza_promedio = confianza_promedio
                            print(f"Resultado actualizado con PSM {psm}")
                except Exception as e:
                    print(f"Error del PSM{psm}: {e}")
                    continue

            tiempo_total = time.time() - inicio_tiempo
            print(f"OCR completo en {tiempo_total:.2f}s - {len(mejor_resultado)} textos detectados")
            print(f"Confianza promedio: {mejor_confianza_promedio:.1f}%")
            return mejor_resultado
        
        except Exception as e: 
            print(f"Erro en deteccio ocr: {e}")
            return []
        
    def _mejorar_imagen_ocr(self, imagen: np.ndarray) -> np.ndarray:
        try :
            if len (imagen.shape) == 3:
                imagen_gris = cv2.cvtcolor(imagen, ccv2.COLOR_BGR2GRAY)
            else: 
                imagen_gris = imagen.copy()
            if  min(alutra, ancho) < 300:
                factor_escala - 800 / min(altura, ancho)
                nuevo_ancho = int(ancho * factor_escala)
                nueva_altura = int(altura * factor_escala)
                imagen_gris  = cv2.resize(iamgen_gris, (nuevo_ancho, nueva_altura),
                interpolation = cv2.INTER_CUBIC

                ) 
                print(f"Redimenciones de la imagen: {ancho}x{altura} -> {nuevo_ancho}x{nueva_altura}")

                imagen_denoised = cv2.fastNlMeansDenoising(imagen_gris, Non, h=10, templateWindowSize=7, searchWindowSize=21)
                imagen_rotada = self._corregir_rotacion(imagen_denoised)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8.8))
                imagen_mejora  = clahe.aply(imagen_rotada)

                kernel_sharpening = np.ndarray([-1, -1, -1], [-1, 9, -1], [-1, -1, -1])