import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict
from collections import Counter
import textwrap
import re 

class AnalizadorEscena:
    def __init__(self, modelo_det_path: str = 'yolov8n.pt', modelo_seg_path: str = 'yolov8n-seg.pt',
                confianza_minima: float = 0.4, modelo_custom_path: str = None):

        print("Inicializando Analizador de Escena...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {self.device}")

        ruta_modelo_det = modelo_custom_path or modelo_det_path
        print(f"Cargando modelo de detección desde: {ruta_modelo_det}")
        try:
            self.modelo_deteccion = YOLO(ruta_modelo_det) = self.modelo_deteccion.model.names
        except Exception as e:
            print(f"ERROR CRITICO al cargar el modelo de detección desde '{ruta_modelo_det}': {e} !!!")
            print("Asegúrate de que la ruta es correcta y el archivo del modelo no está corrupto.")
            raise ValueError(f"No se pudo cargar el modelo de detección: {ruta_modelo_det}") from e

        try:
            self.modelo_segmentacion = YOLO(modelo_seg_path)
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo cargar el modelo de segmentación desde '{modelo_seg_path}': {e}. El análisis de contexto estará desactivado.")
            self.modelo_segmentacion = None

        self.confianza_minima_real = confianza_minima
        self.confianza_minima_depuracion = 0.1
        print(f"!!! MODO DEPURACIÓN AVANZADO ACTIVADO (umbral bajo: {self.confianza_minima_depuracion}, umbral real: {self.confianza_minima_real}) !!!")

        self.ETIQUETAS_INGLES_A_ESPANOL = {
            # Clases 91
            # mié 22 oct 2025 
            'Dime': 'moneda de diez centavos',
            'Fifty': 'billete de cincuenta dólares',
            'Five': 'billete de cinco dólares',
            'Hundred': 'billete de cien dólares',
            'Nickel': 'moneda de cinco centavos',
            'One': 'billete de un dólar',
            'Penny': 'moneda de un centavo',
            'Quarter': 'moneda de veinticinco centavos',
            'Ten': 'billete de diez dólares',
            'Twenty': 'billete de veinte dólares',
            'Two': 'billete de dos dólares',
            'aeroplane': 'avión', # Nombre del modelo base COCO
            'ascending': 'escaleras que suben',
            'backpack': 'mochila',
            'banana': 'plátano',
            'baseball bat': 'bate de béisbol',
            'baseball glove': 'guante de béisbol',
            'bear': 'oso',
            'bed': 'cama',
            'bench': 'banco',
            'bicycle': 'bicicleta',
            'bird': 'pájaro',
            'boat': 'barco',
            'book': 'libro',
            'bottle': 'botella',
            'bowl': 'tazón',
            'broccoli': 'brócoli',
            'bus': 'autobús',
            'cake': 'pastel',
            'car': 'automóvil',
            'carrot': 'zanahoria',
            'cat': 'gato',
            'cell phone': 'celular',
            'chair': 'silla',
            'clock': 'reloj',
            'cup': 'taza',
            'descending': 'escaleras que bajan',
            'diningtable': 'mesa de comedor',
            'dog': 'perro',
            'donut': 'dona',
            'elephant': 'elefante',
            'fifty': 'billete de cincuenta dólares', # Minúscula por si acaso
            'five': 'billete de cinco dólares',   # Minúscula por si acaso
            'fork': 'tenedor',
            'frisbee': 'frisbee',
            'giraffe': 'jirafa',
            'handbag': 'bolso',
            'horse': 'caballo',
            'hot dog': 'perro caliente',
            'hundred': 'billete de cien dólares', # Minúscula por si acaso
            'kite': 'cometa',
            'knife': 'cuchillo',
            'laptop': 'computadora portátil',
            'microwave': 'microondas',
            'motorbike': 'motocicleta', # Nombre del modelo base COCO
            'mouse': 'ratón',
            'one': 'billete de un dólar',       # Minúscula por si acaso
            'orange': 'naranja',
            'oven': 'horno',
            'person': 'persona',
            'pizza': 'pizza',
            'pottedplant': 'planta en maceta', # Nombre del modelo base COCO
            'refrigerator': 'refrigerador',
            'remote': 'control remoto',
            'sandwich': 'sándwich',
            'scissors': 'tijeras',
            'sink': 'fregadero',
            'skateboard': 'patineta',
            'skis': 'esquís',
            'snowboard': 'tabla de snowboard',
            'sofa': 'sofá',
            'spoon': 'cuchara',
            'sports ball': 'pelota deportiva',
            'stop sign': 'señal de alto',
            'suitcase': 'maleta',
            'teddy bear': 'osito de peluche',
            'ten': 'billete de diez dólares',     # Minúscula por si acaso
            'tennis racket': 'raqueta de tenis',
            'tie': 'corbata',
            'toilet': 'inodoro',
            'toothbrush': 'cepillo de dientes',
            'traffic light': 'semáforo',
            'train': 'tren',
            'truck': 'camión',
            'tvmonitor': 'televisor', # Nombre del modelo base COCO
            'twenty': 'billete de veinte dólares',# Minúscula por si acaso
            'umbrella': 'paraguas',
            'vase': 'florero',
            'walls': 'pared',
            'wine glass': 'copa de vino',
            'zebra': 'cebra'
        }

        self.etiquetas_es = {}
        self.mapa_ids_modelo_a_sistema = {}
        nombres_del_modelo_en_ingles = {}
        try:
            nombres_del_modelo_en_ingles = self.modelo_deteccion.model.names
            if not nombres_del_modelo_en_ingles:
                raise ValueError("El modelo cargado no tiene nombres de clases (names).")
            print(f"Clases detectadas en el archivo del modelo ({len(nombres_del_modelo_en_ingles)}): {list(nombres_del_modelo_en_ingles.values())}")
        except AttributeError:
            print("!!! ERROR: El modelo cargado no tiene el atributo 'names'. No se puede mapear clases. !!!")
        except ValueError as e:
            print(f"!!! ERROR: {e} !!!")

        id_sistema_actual = 0
        ids_usados = set()
        if nombres_del_modelo_en_ingles:
            for id_modelo, nombre_clase_en_ingles in nombres_del_modelo_en_ingles.items():
                nombre_clase_es = self.ETIQUETAS_INGLES_A_ESPANOL.get(nombre_clase_en_ingles)
                if not nombre_clase_es:
                    print(f"  -> ADVERTENCIA: La clase '{nombre_clase_en_ingles}' (ID Mod:{id_modelo}) del modelo no tiene traducción a español definida. Se ignorará.")
                    continue

                # Asignar un ID único del sistema, reutilizando si es posible, o creando uno nuevo
                id_existente = next((id_sis for id_sis, nombre in self.etiquetas_es.items() if nombre == nombre_clase_es), None)
                if id_existente is None:
                    # Buscar el siguiente ID disponible
                    while id_sistema_actual in ids_usados:
                        id_sistema_actual += 1
                    id_sistema_final = id_sistema_actual
                    self.etiquetas_es[id_sistema_final] = nombre_clase_es
                    ids_usados.add(id_sistema_final)
                    print(f"  -> Clase '{nombre_clase_es}' registrada en el sistema con ID: {id_sistema_final}")
                    id_sistema_actual += 1
                else:
                    id_sistema_final = id_existente

                self.mapa_ids_modelo_a_sistema[id_modelo] = id_sistema_final
        else:
            print("!!! ERROR: No se pudieron leer las clases del modelo. El mapeo estará vacío. !!!")


        # Actualizar artículos y objetos prioritarios basados en las etiquetas finales
        # (Asegúrate de que los nombres aquí coincidan EXACTAMENTE con las traducciones finales)
        self.articulos_es = { 'persona': 'una', 'bicicleta': 'una', 'automóvil': 'un', 'motocicleta': 'una', 'avión': 'un', 'autobús': 'un', 'tren': 'un', 'camión': 'un', 'barco': 'un', 'semáforo': 'un', 'boca de incendios': 'una', 'señal de alto': 'una', 'parquímetro': 'un', 'banco': 'un', 'pájaro': 'un', 'gato': 'un', 'perro': 'un', 'caballo': 'un', 'oveja': 'una', 'vaca': 'una', 'elefante': 'un', 'oso': 'un', 'cebra': 'una', 'jirafa': 'una', 'mochila': 'una', 'paraguas': 'un', 'bolso': 'un', 'corbata': 'una', 'maleta': 'una', 'frisbee': 'un', 'esquís': 'unos', 'tabla de snowboard': 'una', 'pelota deportiva': 'una', 'cometa': 'una', 'bate de béisbol': 'un', 'guante de béisbol': 'un', 'patineta': 'una', 'tabla de surf': 'una', 'raqueta de tenis': 'una', 'botella': 'una', 'copa de vino': 'una', 'taza': 'una', 'tenedor': 'un', 'cuchillo': 'un', 'cuchara': 'una', 'tazón': 'un', 'plátano': 'un', 'manzana': 'una', 'sándwich': 'un', 'naranja': 'una', 'brócoli': 'un', 'zanahoria': 'una', 'perro caliente': 'un', 'pizza': 'una', 'dona': 'una', 'pastel': 'un', 'silla': 'una', 'sofá': 'un', 'planta en maceta': 'una', 'cama': 'una', 'mesa de comedor': 'una', 'inodoro': 'un', 'televisor': 'un', 'computadora portátil': 'una', 'ratón': 'un', 'control remoto': 'un', 'teclado': 'un', 'celular': 'un', 'microondas': 'un', 'horno': 'un', 'tostadora': 'una', 'fregadero': 'un', 'refrigerador': 'un', 'libro': 'un', 'reloj': 'un', 'florero': 'un', 'tijeras': 'unas', 'osito de peluche': 'un', 'secador de cabello': 'un', 'cepillo de dientes': 'un',
                        'escaleras': 'unas', 'escaleras que suben': 'unas', 'escaleras que bajan': 'unas', 'pared': 'una',
                        'billete de un dólar': 'un', 'billete de cinco dólares': 'un', 'billete de diez dólares': 'un', 'billete de veinte dólares': 'un', 'billete de cincuenta dólares': 'un', 'billete de cien dólares': 'un', 'billete de dos dólares':'un',
                        'moneda de un centavo': 'una', 'moneda de cinco centavos': 'una', 'moneda de diez centavos': 'una', 'moneda de veinticinco centavos': 'una' }

        nombres_prioritarios = { 'persona', 'automóvil', 'motocicleta', 'autobús', 'mochila', 'botella', 'taza', 'silla', 'sofá', 'cama', 'mesa de comedor', 'inodoro', 'televisor', 'computadora portátil', 'control remoto', 'celular', 'libro', 'escaleras', 'escaleras que suben', 'escaleras que bajan', 'pared', 'billete de un dólar', 'billete de cinco dólares', 'billete de diez dólares', 'billete de veinte dólares', 'billete de cincuenta dólares', 'billete de cien dólares', 'billete de dos dólares', 'moneda de un centavo', 'moneda de cinco centavos', 'moneda de diez centavos', 'moneda de veinticinco centavos'}
        self.objetos_prioritarios = {id for id, nombre in self.etiquetas_es.items() if nombre in nombres_prioritarios}

        print(f"IDs de sistema mapeados ({len(self.mapa_ids_modelo_a_sistema)}): {self.mapa_ids_modelo_a_sistema}")
        print(f"Etiquetas finales en español ({len(self.etiquetas_es)}): {self.etiquetas_es}")
        print("Analizador inicializado correctamente con mapeo de clases dinámico.")

    def _detectar_objetos_principales(self, imagen: np.ndarray, solo_prioritarios: bool) -> List[Dict]:
        detecciones = []
        if not hasattr(self.modelo_deteccion, 'model') or not hasattr(self.modelo_deteccion.model, 'names'):
            print("ERROR: El modelo de detección no se cargó correctamente o no tiene nombres de clases.")
            return detecciones
            
        resultados = self.modelo_deteccion(imagen, conf=self.confianza_minima_depuracion, verbose=False)
        altura_img, ancho_img = imagen.shape[:2]

        print("\n--- INICIO DEPURACIÓN DE FRAME ---")
        if not resultados or not hasattr(resultados[0], 'boxes') or resultados[0].boxes is None:
            print("El modelo no detectó NADA en este frame.")
        else:
            nombres_modelo = self.modelo_deteccion.model.names
            detecciones_validas_para_descripcion = 0 # Contador para el mensaje final
            for res in resultados:
                if res.boxes is None: continue
                for box in res.boxes:
                    confianza = round(float(box.conf[0]), 2)
                    clase_id_original = int(box.cls[0])
                    nombre_clase_en_ingles = nombres_modelo.get(clase_id_original, f'ID_{clase_id_original}_EN')

                    clase_id_sistema = self.mapa_ids_modelo_a_sistema.get(clase_id_original)
                    # --- CAMBIO: Usar get con default 'desconocido_es' ---
                    nombre_clase_es = self.etiquetas_es.get(clase_id_sistema, 'desconocido_es')

                    print(f"[DEBUG] Objeto Visto: '{nombre_clase_en_ingles}' (ID Mod:{clase_id_original}) -> '{nombre_clase_es}' (ID Sis:{clase_id_sistema}) (Conf: {confianza})")

                    # Aplicar filtros DESPUÉS de imprimir el debug
                    if confianza < self.confianza_minima_real:
                        print(f"Ignorado por voz: Confianza ({confianza}) < Umbral ({self.confianza_minima_real})")
                        continue
                    if clase_id_sistema is None or nombre_clase_es == 'desconocido_es':
                        print(f"Ignorado por voz: Clase desconocida o sin mapeo.")
                        continue
                    if solo_prioritarios and clase_id_sistema not in self.objetos_prioritarios:
                         print(f"Ignorado por voz: '{nombre_clase_es}' no es prioritario.")
                         continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 >= x2 or y1 >= y2:
                        print(f"Ignorado: Coordenadas inválidas x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                        continue
                    #Lista final 
                    deteccion = {
                        'nombre': nombre_clase_es,
                        'confianza': confianza,
                        'posicion': self._calcular_posicion((x1 + x2) / 2, (y1 + y2) / 2, ancho_img, altura_img),
                        'distancia_relativa': self._estimar_distancia((x2 - x1) * (y2 - y1), ancho_img * altura_img),
                        'coordenadas': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    }
                    detecciones.append(deteccion)
                    detecciones_validas_para_descripcion += 1 # Incrementar contador

        print(f" FIN DEPURACIÓN DE FRAME ({detecciones_validas_para_descripcion} detecciones pasarán a descripción) ---\n")
        return detecciones

    def _generar_descripcion_completa(self, objetos: List[Dict], contexto: List[str]) -> str:
        objetos_validos = [obj for obj in objetos if obj['nombre'] != 'desconocido_es']

        if not objetos_validos and not contexto:
            return "No se detectan elementos claros en la escena."

        personas = [obj for obj in objetos_validos if obj['nombre'] == 'persona']
        otros_objetos_validos = [obj for obj in objetos_validos if obj['nombre'] != 'persona']
        contador_objetos = Counter(obj['nombre'] for obj in otros_objetos_validos)
        partes = []

        if personas:
            if len(personas) == 1:
                p = personas[0]
                partes.append(f"hay una persona {p['distancia_relativa']} {p['posicion']}")
            else:
                partes.append(f"hay {len(personas)} personas")


        objetos_descritos = []
        if contador_objetos:
            for nombre, cantidad in contador_objetos.most_common():
                articulo_singular = self.articulos_es.get(nombre, 'un' if not nombre.endswith('a') and not nombre.endswith('s') else 'una')

                if cantidad == 1:
                    obj = next(o for o in otros_objetos_validos if o['nombre'] == nombre)
                    objetos_descritos.append(f"{articulo_singular} {nombre} {obj['posicion']}")
                else:
                    plural = nombre
                    if nombre.endswith('z'): plural = nombre[:-1] + 'ces'
                    elif nombre.endswith(('s', 'x')): pass
                    elif nombre.endswith(('ión', 'án', 'én')): plural = nombre[:-2] + nombre[-1] + 'nes'
                    elif nombre.endswith(('á', 'é', 'í', 'ó', 'ú')): plural += 's'
                    elif nombre.endswith(('a', 'e', 'i', 'o', 'u')): plural += 's'
                    else: plural += 'es'

                    if cantidad == 2: num_str = 'dos'
                    elif cantidad == 3: num_str = 'tres'
                    else: num_str = str(cantidad)

                    objetos_descritos.append(f"{num_str} {plural}")

        if objetos_descritos:
            if partes: partes.append("además de")
            else: partes.append("veo")
            if len(objetos_descritos) > 1:
                objetos_str = ", ".join(objetos_descritos[:-1]) + " y " + objetos_descritos[-1]
            else:
                objetos_str = objetos_descritos[0]
            partes.append(objetos_str)


        if contexto:
            contexto_str = f"en un entorno con {', '.join(contexto)}"
            if partes:
                if partes[-1].endswith('.'): partes[-1] = partes[-1][:-1]
                partes.append(contexto_str)
            else:
                return f"El entorno parece tener {', '.join(contexto)}.".capitalize()

        if not partes: return "No se detectan objetos de interés."

        descripcion_final = " ".join(partes).capitalize()
        descripcion_final = re.sub(r'\s*\.+\s*$', '', descripcion_final) + '.'
        descripcion_final = re.sub(r'\s{2,}', ' ', descripcion_final)

        return descripcion_final

    def analizar(self, imagen: np.ndarray, solo_prioritarios: bool = True) -> Dict:
        try:
            objetos_detectados = self._detectar_objetos_principales(imagen, solo_prioritarios)
            contexto_escena = self._analizar_contexto_escena(imagen)
            descripcion_completa = self._generar_descripcion_completa(objetos_detectados, contexto_escena)

            #(filtrados) en el resultado 
            return {
                'objetos': objetos_detectados,
                'contexto': contexto_escena,
                'descripcion': descripcion_completa
            }
        except Exception as e:
            import traceback
            print(f"Error FATAL en el análisis de escena: {e}")
            traceback.print_exc()
            return {'objetos': [], 'contexto': [], 'descripcion': 'Error grave en el análisis.'}


    def _analizar_contexto_escena(self, imagen: np.ndarray) -> List[str]:
        if self.modelo_segmentacion is None:
            return []

        clases_contexto_map = {
            'road': 'carretera', 'sidewalk': 'acera', 'building': 'edificios',
            'wall': 'muro', 'sky': 'cielo', 'grass': 'césped', 'tree': 'árboles'
        }
        contexto_detectado = set()
        try:
            resultados = self.modelo_segmentacion(imagen, conf=0.3, verbose=False, device=self.device)

            if resultados and hasattr(resultados[0], 'masks') and resultados[0].masks is not None:
                nombres_modelo_seg = resultados[0].names
                num_masks = len(resultados[0].masks)
                num_boxes = len(resultados[0].boxes) if hasattr(resultados[0], 'boxes') and resultados[0].boxes is not None else 0

                for i in range(min(num_masks, num_boxes)):
                    if resultados[0].boxes[i].cls is None or not resultados[0].boxes[i].cls.nelement():
                        continue
                    clase_id = int(resultados[0].boxes[i].cls[0])
                    nombre_clase_en = nombres_modelo_seg.get(clase_id)
                    if nombre_clase_en and nombre_clase_en in clases_contexto_map:
                        contexto_detectado.add(clases_contexto_map[nombre_clase_en])

        except Exception as e:
            print(f"Error analizando contexto: {e}")

        return sorted(list(contexto_detectado))


    def _calcular_posicion(self, centro_x: float, centro_y: float, ancho: int, alto: int) -> str:
        if ancho <= 0 or alto <= 0: return "en posición desconocida"
        if centro_x < ancho / 3: horizontal = "a la izquierda"
        elif centro_x < 2 * ancho / 3: horizontal = "en el centro"
        else: horizontal = "a la derecha"
        if centro_y < alto / 3: vertical = "arriba"
        elif centro_y < 2 * alto / 3: vertical = ""
        else: vertical = "abajo"
        posicion = f"{vertical} {horizontal}".strip()
        return posicion if posicion else "en el centro"


    def _estimar_distancia(self, area_objeto: float, area_total: float) -> str:
        if area_total <= 0: return "a distancia desconocida"
        porcentaje_area = (area_objeto / area_total) * 100
        if porcentaje_area > 20: return "muy cerca"
        elif porcentaje_area > 8: return "cerca"
        elif porcentaje_area > 3: return "a media distancia"
        else: return "lejos"

    def dibujar_analisis(self, imagen: np.ndarray, analisis: Dict) -> np.ndarray:
        frame_resultado = imagen.copy()
        for det in analisis.get('objetos', []):
            if 'coordenadas' not in det: continue
            coords = det['coordenadas']
            x1, y1, x2, y2 = coords.get('x1',0), coords.get('y1',0), coords.get('x2',0), coords.get('y2',0)
            if x2 > x1 and y2 > y1:
                color = (34, 139, 34)
                cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), color, 2)
                label = f"{det.get('nombre', '???')} ({det.get('confianza', 0):.0%})"
                (w_text, h_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                overlay = frame_resultado.copy()
                bg_y1_text = max(0, y1 - h_text - baseline - 2)
                cv2.rectangle(overlay, (x1, bg_y1_text), (x1 + w_text, y1), (0, 0, 0), -1)
                alpha = 0.6
                cv2.addWeighted(overlay, alpha, frame_resultado, 1 - alpha, 0, frame_resultado)
                text_y = max(h_text + baseline, y1 - baseline)
                cv2.putText(frame_resultado, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        descripcion = analisis.get('descripcion', '')
        y = 30
        line_height = 0
        try:
            wrapped_text = textwrap.wrap(descripcion, width=50)
            for i, line in enumerate(wrapped_text):
                (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                line_height = h + baseline
                bg_y1 = y - line_height - 2
                bg_y2 = y + baseline + 2
                if bg_y1 < 0:
                    bg_y1 = 0
                y = line_height + 2


                overlay_desc = frame_resultado.copy()
                cv2.rectangle(overlay_desc, (5, bg_y1), (15 + w, bg_y2), (0, 0, 0), -1)
                alpha_desc = 0.7
                cv2.addWeighted(overlay_desc, alpha_desc, frame_resultado, 1 - alpha_desc, 0, frame_resultado)
                cv2.putText(frame_resultado, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                y += line_height + 5
        except Exception as e:
            print(f"Error al dibujar texto de descripción: {e}")


        return frame_resultado