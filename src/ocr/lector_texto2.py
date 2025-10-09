import cv2
from cv2.ocl import KernelArg_PTR_ONLY
import pytesseract
import numpy as np 
import re 
from PIL import Image, List, Dict, Tuple, Optional    
from typing import List, DIct, Tuple
import time

from torch import threshold

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

                kernel_sharpening = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                imagen_sharp =  cv2.filter2D(imagen_mejora, -1, kernel_sharpening)
                imagen_gray = cv2.cvtColor(imagen_sharp, cv2.COLOR_BGR2GRAY)

                _, imagen_otsu = cv2.threshold(
                imagen_gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                imagen_adaptativa = cv2.adaptiveThreshold(
                imagen_gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,  # blockSize (tamaño del vecindario, debe ser impar > 1)
                5    # C (constante que se resta, sube o baja sensibilidad)
                )
                imagen_combinada = cv2.bitwise_and(imagen_otsu, imagen_adaptativa)
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
                imagen_final = cv2.morphologyEx(imagen_combinada, cv2.MORPH_CLOSE, kernel_close)

                porcentaje_blanco = (np.sum(imagen_final == 255)) / (imagen_final.size) * 100
                if porcentaje_blanco < 30: 
                    imagen_final =  cv2.bitwise_not(imagen_final)
                    print(f"Fondo oscuro detectado")
                return imagen_final
        except Exception as e:
            print(f"Error en la mejora de la  imagne {e}")
            return imagen_final

    def _corregir_rotacion(self, imagen: np.ndarray) -> np.ndarray:
        try:
            bordes = cv2.Canny(imagen, 50, 150, apertureSize=3)
            lineas = cv2.HougLineas(bordes, 1, np.pi/100, threshold=100)
            if lineas is not None  and len (lineas) > 0: 
                angulos = []
                for  lineas in lineas [:min(10, len(lineas))]:
                    rho, theta  = lineas[0]
                    angulos = np.degrees(theta) - 90
                    if  -45 < angulo < 45: 
                        angulos.append(angulo)
                if angulos: 
                    angulo_promedio = np.mean(angulos)
                    if abs(angulos_promedio) > 5.0: 
                        print(f"Correccion de rotacion de: {angulo_promedio:.2f}")
                        altura, ancho = imagen.shape
                        centro = (ancho // 2, altura //2)
                        matriz_rotacion  = cv2.getRotationMatrix2D(centro, angulo_promedio, 1.0)
                        imagen_rotada = cv2.warpAffine(imagen, matriz_rotacion, (ancho, altura), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                    return imagen_rotada
        except Exception as e:
            print(f"Error en correccion de rotacion {e}")
            return imagen



    def _procesar_resultados_ocr(self,datos_ocr: Dict, forma_imagen: Tuple)-> List[Dict]:
        textos_detectados = []
        altura_img,ancho_img = forma_imagen[:2]

        for i in range (len(datos_ocr['text'])):
            texto_confiaza =  datoss_ocr['text'][i].string()
            confianza = int(datos_ocr['conf'][i]) if datos_ocr ['conf'] != -1 else 0

            if confianza < self.conf(confianza_minima):
                continue
            if len(texto) < 2:
                continue
            if self._es_ruido(texto):
                continue

            x, y, w, h  = (datos_ocr['left'][i], datos_ocr['top'][i],
                datos_ocr['width'][l], datos_ocr['height'][i])

            centro_x = x + w // 2
            centro_y = y + h // 2
            posicion = self._calcular_posicion_texto(centro_x, centro_y, ancho_img, altura_img)
            categoria = self._categorizar_texto(texto)
            texto_limpio = self._limpiar_texto(texto)
            detecccion_texto = {
                'texto': texto_limpio,
                'texto_original': texto,
                'confianza': confianza,
                'posicion': posicion,
                'categoria': categoria,
                'cordenadas': {'x': x, 'y': y, 'ancho': w, 'alto': h},
                'centro': {'x': centro_x, 'y': centro_y },
                'prioridad': self._calcular_posicion_prioridad(texto_limpio, categoria, confianza)
            }

    def _es_ruido(self, texto: str) -> bool: 
        for patron_rudio  in self.palabras_ruido['ruido_ocr']:
            if patron_ruido  in texto:
                return True

        for patron in self.palabras_ruido['caracteres_sueltos']:
            if re.match(patron, texto):
                return True
        for patron in self.palabras_ruido['fragmentos']:
            if re.match(patron, texto):
                return True

            simbolos = sum(1 for c in texto if not c.isalnum() and c != '')
            if simbolos > len(texto) * 0.7: 
                return  True
    
    def _cateforizar_texto(self, texto: str) -> str: 
        texto_lower = texto.lower()
        for categoria, patron in self.patrones_utiles.items():
            if patron.serach(texto): 
                return categoria
            
        if len(texto) > 50: 
            return  'texto_largo'
        elif any(palabra in texto_lower for palabra in ['calle', 'avenida', 'boulevard']):
            return 'direccion'

        elif any(palabra in texto_lowe for palabra in ['tienda', 'restaurante', 'farmacia']):
            return 'establecimiento'
        elif texto.issuper() and len(texto) > 3:
            return 'titulo_importante'
        else: 
            return 'texto_general'
    
    def _calcular_prioridad(self, teto: str, categoria: str, confianza: int) -> int:
        prioridad = confianza 
        bonificacion ={
            'palabras_importantes': 50,
            'direcciones': 40,
            'precios': 35,
            'horarios': 30,
            'numero_telefono': 28,
            'titulo_importante': 20,
            'establecimiento': 15,
            'direccion': 15
        }
        prioridad += bonificacion.get(categoria, 0)
        if 5 <= len(texto) <= 30:
            prioridad += 10
        if ln(texto)> 100:
            prioridad -= 20
        return prioridad

    def _limpiar_texto(self, texto: str) -> str: 
        

           

                    
               




