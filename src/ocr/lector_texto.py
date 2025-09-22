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

        def detectar_texto(self, imagen: np.ndarray, mejorar_imagen: bool = True) -> List[Dict]:
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
                texto = datos_ocr['text'][i].strip()
                confianza = int(datos_ocr['conf'][i]) if datos_ocr['conf'][i] != -1 else 0
                if confianza < self.confianza_minima:
                    continue
                if len(texto) < 2:
                    if self.es_ruido(texto):
                        continue
                    x,y,w,h  = (datos_ocr['left'][i],datos_ocr['top'][i],
                    datos_ocr['width'][i], datos_ocr['height'][i])
                    #posicion relativa 
                    centro_x =  x + y // 2
                    centro_y = y + h //2
                    posicion = self.calcular_posicion_texto(centro_x, centro_y, ancho_img, altura_img)
                    categoria = self._categorizar_texto(texto)
                    texto_limpio = self._limpiar_texto(texto)
                    detecteccion_texto = {
                        'texto': texto_limpio, 
                        'texto_original': texto, 
                        'confianza':confianza,
                        'categotia':categoria,
                        'cordenadas': centro_x, 'y': centro_y,
                        'prioridad': self._calcular_prioridad(texto_limpio, categoria, confianza)
                    }
                    textos_detectados.append(detecteccion_texto)
                    textos_detectados.sort(key=lambda x: x['prioridad'], reverse=True)
                    return textos_detectados
                
                def _es_ruido(self, texto: str)-> bool: 
                    for patron_ruido in self.palabras_ruido['ruido_ocr']:
                        if patron_ruido in texto:
                            return True
                    for patron  in self.palabras_ruido['caracteres_sueltos']:
                        if re.match(patron, texto):
                            return True
                    for patron in self.palabras_ruido['fragmentos']:
                        if re.match(patron, texto):
                            return True
                    simbolos = sum(1 for c in texto if not c.lsalnum()and c != '') #Detecta texto con simbolos
                    if simbolos > len(texto)*0.7: # 70% de deteccion de simbolos
                        return True   

                    return False   
                
                def _categorizar_texto(self, texto: str)->str:#Divide el texto detectado por categoria 21/09/2025
                    texto_lower = texto.lower()
                    for categoria, patron in self.patrones_utiles.items():
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
                    

                def _calcular_prioridad(self, texto: str, categoria: str, confianza) -> int: 
                    prioridad = confianza 
                    bonificaciones = {
                        'palabras_importantes':50,
                        'direcciones':40,
                        'precios':35,
                        'horario':30,
                        'numero_telefono':25,
                        'titulo_importante':20,
                        'establecimiento':15,
                        'direccion':15
                    }
                    prioridad += bonificaciones.get(categoria, 0)
                    if 5 <=  len(texto) <= 30:
                        prioridad +=10
                    if len(texto) > 100:
                        prioridad -=20 
                    if len (texto)>100:
                        prioridad -=20
                    return prioridad
                
                def _limpiar_texto(self, texto: str) -> str:
                    corecciones = {
                        'rn': 'm',
                        'n1': 'ñ',
                        '0': 'o',
                        '1': 'l'
                    }
                    for error, correcion in corecciones.items():
                        patron = rf'(?<=[a-zA-ZáéíóúñÑ]){re.escape(error)}(?=[a-zA-ZáéíóúñÑ])' 
                        texto_limpio = re.sub(patron, correcion, texto_limpio)
                        return texto_limpio
                
                def _calcular_posicion_texto(self, centro_x: int, centro_y: int, ancho:int, alto: int) -> str:
                    tercio_ancho  = ancho / 3
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
                        return f"en la parte{vertical}"
                    elif vertical == "medio":
                        return f"a la {horizontal}"
                    else:
                        return f"{vertical} a la {horizontal}"
                    
                def generar_descripcion_audio(self, textos: List[Dict], modo:str ='resumen') -> str:
                    if not textos:
                        return  "No se detecta texto legible en la imagen"
                    if modo == 'prioritario':
                        texto_principal = texto[0]
                        return f"Textos detectados: {texto_principal['texto']}"
                    elif modo == 'resumen':
                        if len(textos) == 1:
                            return f"Detecto el texto: {textos[0]['textos']}"
                        elif len (textos) <= 3:
                            textos_str = ", ".join([t ['textos']for t in textos] )
                            return f"Detecto los textos:{textos_str}"
                        else:
                            texto_principal = textos[0]['texto']
                            return f"Detecto {len(textos)} textos. El  principal es: {texto_principal}"
                        
                    else: 
                        descripciones = []
                        for i, texto in enumerate(textos[:5],1): 
                            descripciones.append(f"{texto['texto']}{texto['posicion']}")
                            
                            if len(texto) <=5: 
                                return f"Textos detectados: {'.'.join(descripciones)}"
                            else:
                                return f"Textos detectados: {'.'.join(descripciones)} y {len(textos)-5} mas"
                            
                def dibujar_texto_detectado(self, imagen: np.ndarray, textos: List[Dict]) -> np.ndarray: # ARREGLO DEL DIBUJADO EN RECTAGULO
                    imagen_resultado = imagen.copy()
                    for i, textos_info  in enumerate(textos):
                        coords = textos_info['coordenadas']
                        x, y, w, h = coords['x'], coords['y'], coords['ancho'], coords['alto']
                        # Color basado en prioridad
                        if textos_info['prioridad'] > 80:
                            color = (0,255,0) # color verde para alta prioridad
                        elif textos_info['prioridad'] > 50:
                            color = (255,238,0) # color amarillo para medio propiedad
                        else:
                            color = (255,0,0) # rojo para baja prioridad
                        cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), color, 2)

                        texto_mostrar = textos_info['texto'][:30] 
                        if len (textos_info['texto']) > 30:
                            texto_mostrar += "..."
                        etiqueta = f"{i+1}: {texto_mostrar}'({textos_info['confianza']}%)"
                        (w_texto, h_texto), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5,1)
                        cv2.rectangle(imagen_resultado, (x,y - h_texto - 5 ), (x + w_texto, y), color, -1)
                        #texto
                        cv2.putText(imagen_resultado, etiqueta, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return imagen_resultado
                
                def leer_texto_continuo(self, imagen: np.ndarray, velocidad_lectura: str = 'normal') ->List[str]:
                    textos = self.detectar_textos(imagen, mejorar_imagen = True)
                    if not textos: 
                        return ["No se detecto texto en el documento"]
                    
                    textos_ordenados = sorted(textos, key=lambda x: (x['centro']['y'], x['centro']['x']))

                    fragmentos = []
                    fragmento_actual = []

                    for textos in textos_ordenados:
                        fragmento_actual.append(texto['texto'])

                        texto_fragmento = ' '.join(fragmento_actual)
                        if len(texto_fragmento) > 100 or texto['texto'].endswith('.'):
                            fragmentos.append(texto_fragmento)
                            fragmento_actual = []
                    if fragmento_actual:
                        fragmentos.append(' '.join(fragmento_actual))

                    return fragmentos if fragmentos else ["No se pudo procesar el texto del documento"] 





                            




                        



                    


                    













                    








                        








                   









        





