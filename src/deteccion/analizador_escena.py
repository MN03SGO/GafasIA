import cv2
import numpy as np
# --- CAMBIO CRÍTICO: CORREGIDO EL ERROR DE ESCRITURA EN EL IMPORT ---
from ultralytics import YOLO
import torch
from typing import List, Dict
from collections import Counter
import textwrap
import re
import traceback

class AnalizadorEscena:
    def __init__(self, modelo_det_path: str = 'yolov8n.pt', modelo_seg_path: str = 'yolov8n-seg.pt',
                 confianza_minima: float = 0.4, modelo_custom_path: str = None):

        print("Inicializando Analizador de Escena...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {self.device}")

        ruta_modelo_det = modelo_custom_path or modelo_det_path
        print(f"Cargando modelo de detección desde: {ruta_modelo_det}")
        try:

            self.modelo_deteccion = YOLO(ruta_modelo_det)
            _ = self.modelo_deteccion.model.names
        except Exception as e:
            print(f"!!! ERROR CRÍTICO al cargar el modelo de detección '{ruta_modelo_det}': {e} !!!")
            traceback.print_exc()
            raise ValueError(f"No se pudo cargar el modelo de detección: {ruta_modelo_det}") from e

        try:
            self.modelo_segmentacion = YOLO(modelo_seg_path)
        except Exception as e:
             print(f"ADVERTENCIA: No se pudo cargar modelo de segmentación '{modelo_seg_path}': {e}. Contexto desactivado.")
             self.modelo_segmentacion = None

        self.confianza_minima_real = confianza_minima
        print(f"Umbral de confianza para descripción: {self.confianza_minima_real}")


        # diccionario en espanol
        self.ETIQUETAS_INGLES_A_ESPANOL = {
             'Dime': 'moneda de diez centavos','Fifty': 'billete de cincuenta dólares','Five': 'billete de cinco dólares','Hundred': 'billete de cien dólares','Nickel': 'moneda de cinco centavos','One': 'billete de un dólar','Penny': 'moneda de un centavo','Quarter': 'moneda de veinticinco centavos','Ten': 'billete de diez dólares','Twenty': 'billete de veinte dólares','Two': 'billete de dos dólares','aeroplane': 'avión','ascending': 'escaleras que suben','backpack': 'mochila','banana': 'plátano','baseball bat': 'bate de béisbol','baseball glove': 'guante de béisbol','bear': 'oso','bed': 'cama','bench': 'banco','bicycle': 'bicicleta','bird': 'pájaro','boat': 'barco','book': 'libro','bottle': 'botella','bowl': 'tazón','broccoli': 'brócoli','bus': 'autobús','cake': 'pastel','car': 'automóvil','carrot': 'zanahoria','cat': 'gato','cell phone': 'celular','chair': 'silla','clock': 'reloj','cup': 'taza','descending': 'escaleras que bajan','diningtable': 'mesa de comedor','dog': 'perro','donut': 'dona','elephant': 'elefante','fifty': 'billete de cincuenta dólares','five': 'billete de cinco dólares','fork': 'tenedor','frisbee': 'frisbee','giraffe': 'jirafa','handbag': 'bolso','horse': 'caballo','hot dog': 'perro caliente','hundred': 'billete de cien dólares','kite': 'cometa','knife': 'cuchillo','laptop': 'computadora portátil','microwave': 'microondas','motorbike': 'motocicleta','mouse': 'ratón','one': 'billete de un dólar','orange': 'naranja','oven': 'horno','person': 'persona','pizza': 'pizza','pottedplant': 'planta en maceta','refrigerator': 'refrigerador','remote': 'control remoto','sandwich': 'sándwich','scissors': 'tijeras','sink': 'fregadero','skateboard': 'patineta','skis': 'esquís','snowboard': 'tabla de snowboard','sofa': 'sofá','spoon': 'cuchara','sports ball': 'pelota deportiva','stop sign': 'señal de alto','suitcase': 'maleta','teddy bear': 'osito de peluche','ten': 'billete de diez dólares','tennis racket': 'raqueta de tenis','tie': 'corbata','toilet': 'inodoro','toothbrush': 'cepillo de dientes','traffic light': 'semáforo','train': 'tren','truck': 'camión','tvmonitor': 'televisor','twenty': 'billete de veinte dólares','umbrella': 'paraguas','vase': 'florero','walls': 'pared','wine glass': 'copa de vino','zebra': 'cebra'
        }

        self.etiquetas_es = {}
        self.mapa_ids_modelo_a_sistema = {}
        nombres_del_modelo_en_ingles = {}
        try:
            nombres_del_modelo_en_ingles = self.modelo_deteccion.model.names
            if not nombres_del_modelo_en_ingles: raise ValueError("Modelo sin nombres de clases.")
            print(f"Clases detectadas en modelo ({len(nombres_del_modelo_en_ingles)}): {list(nombres_del_modelo_en_ingles.values())}")
        except Exception as e:
             print(f"!!! ERROR LEYENDO CLASES DEL MODELO: {e}. Mapeo vacío. !!!")

        id_sistema_actual = 0
        ids_usados = set()
        if nombres_del_modelo_en_ingles:
            for id_modelo, nombre_clase_en_ingles in nombres_del_modelo_en_ingles.items():
                nombre_clase_es = self.ETIQUETAS_INGLES_A_ESPANOL.get(nombre_clase_en_ingles)
                if not nombre_clase_es:

                    continue
                id_existente = next((id_sis for id_sis, nombre in self.etiquetas_es.items() if nombre == nombre_clase_es), None)
                if id_existente is None:
                    while id_sistema_actual in ids_usados: id_sistema_actua
                    self.etiquetas_es[id_sistema_final] = nombre_clase_es
                    ids_usados.add(id_sistema_final)
                    id_sistema_actual += 1
                else:
                    id_sistema_final = id_existente
                self.mapa_ids_modelo_a_sistema[id_modelo] = id_sistema_final
        else:
             print("!!! ERROR CRÍTICO: No se pudo mapear ninguna clase del modelo. !!!")

        # lista de articulos artículos
        self.articulos_es = { 'persona': 'una', 'bicicleta': 'una', 'automóvil': 'un', 'motocicleta': 'una', 'avión': 'un', 'autobús': 'un', 'tren': 'un', 'camión': 'un', 'barco': 'un', 'semáforo': 'un', 'boca de incendios': 'una', 'señal de alto': 'una', 'parquímetro': 'un', 'banco': 'un', 'pájaro': 'un', 'gato': 'un', 'perro': 'un', 'caballo': 'un', 'oveja': 'una', 'vaca': 'una', 'elefante': 'un', 'oso': 'un', 'cebra': 'una', 'jirafa': 'una', 'mochila': 'una', 'paraguas': 'un', 'bolso': 'un', 'corbata': 'una', 'maleta': 'una', 'frisbee': 'un', 'esquís': 'unos', 'tabla de snowboard': 'una', 'pelota deportiva': 'una', 'cometa': 'una', 'bate de béisbol': 'un', 'guante de béisbol': 'un', 'patineta': 'una', 'tabla de surf': 'una', 'raqueta de tenis': 'una', 'botella': 'una', 'copa de vino': 'una', 'taza': 'una', 'tenedor': 'un', 'cuchillo': 'un', 'cuchara': 'una', 'tazón': 'un', 'plátano': 'un', 'manzana': 'una', 'sándwich': 'un', 'naranja': 'una', 'brócoli': 'un', 'zanahoria': 'una', 'perro caliente': 'un', 'pizza': 'una', 'dona': 'una', 'pastel': 'un', 'silla': 'una', 'sofá': 'un', 'planta en maceta': 'una', 'cama': 'una', 'mesa de comedor': 'una', 'inodoro': 'un', 'televisor': 'un', 'computadora portátil': 'una', 'ratón': 'un', 'control remoto': 'un', 'teclado': 'un', 'celular': 'un', 'microondas': 'un', 'horno': 'un', 'tostadora': 'una', 'fregadero': 'un', 'refrigerador': 'un', 'libro': 'un', 'reloj': 'un', 'florero': 'un', 'tijeras': 'unas', 'osito de peluche': 'un', 'secador de cabello': 'un', 'cepillo de dientes': 'un',
                           'escaleras': 'unas', 'escaleras que suben': 'unas', 'escaleras que bajan': 'unas', 'pared': 'una',
                           'billete de un dólar': 'un', 'billete de cinco dólares': 'un', 'billete de diez dólares': 'un', 'billete de veinte dólares': 'un', 'billete de cincuenta dólares': 'un', 'billete de cien dólares': 'un', 'billete de dos dólares':'un',
                           'moneda de un centavo': 'una', 'moneda de cinco centavos': 'una', 'moneda de diez centavos': 'una', 'moneda de veinticinco centavos': 'una' }

        # Objetos prioritarios
        nombres_prioritarios = { 'persona', 'automóvil', 'motocicleta', 'autobús', 'mochila', 'botella', 'taza', 'silla', 'sofá', 'cama', 'mesa de comedor', 'inodoro', 'televisor', 'computadora portátil', 'control remoto', 'celular', 'libro', 'escaleras', 'escaleras que suben', 'escaleras que bajan', 'pared', 'billete de un dólar', 'billete de cinco dólares', 'billete de diez dólares', 'billete de veinte dólares', 'billete de cincuenta dólares', 'billete de cien dólares', 'billete de dos dólares', 'moneda de un centavo', 'moneda de cinco centavos', 'moneda de diez centavos', 'moneda de veinticinco centavos'}
        self.objetos_prioritarios = {id for id, nombre in self.etiquetas_es.items() if nombre in nombres_prioritarios}

        print(f"IDs de sistema mapeados ({len(self.mapa_ids_modelo_a_sistema)}): OK")
        print(f"Etiquetas finales en español ({len(self.etiquetas_es)}): OK")
        print("Analizador inicializado correctamente.")

    def _detectar_objetos_principales(self, imagen: np.ndarray, solo_prioritarios: bool) -> List[Dict]:
        detecciones = []
        if not hasattr(self.modelo_deteccion, 'model') or not hasattr(self.modelo_deteccion.model, 'names'):
             print("ERROR: Modelo de detección no cargado/inválido.")
             return detecciones

        resultados = self.modelo_deteccion(imagen, conf=self.confianza_minima_real, verbose=False, device=self.device)
        altura_img, ancho_img = imagen.shape[:2]

        if resultados and hasattr(resultados[0], 'boxes') and resultados[0].boxes is not None:
            nombres_modelo = self.modelo_deteccion.model.names
            for res in resultados:
                if res.boxes is None: continue
                for box in res.boxes:
                    confianza = round(float(box.conf[0]), 2)
                    clase_id_original = int(box.cls[0])
                    clase_id_sistema = self.mapa_ids_modelo_a_sistema.get(clase_id_original)
                    if clase_id_sistema is None: continue
                    nombre_clase_es = self.etiquetas_es.get(clase_id_sistema)
                    if not nombre_clase_es or nombre_clase_es == 'desconocido_es': continue
                    if solo_prioritarios and clase_id_sistema not in self.objetos_prioritarios: continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x1 >= x2 or y1 >= y2: continue

                    deteccion = {
                        'nombre': nombre_clase_es,
                        'confianza': confianza,
                        'posicion': self._calcular_posicion((x1 + x2) / 2, (y1 + y2) / 2, ancho_img, altura_img),
                        'distancia_relativa': self._estimar_distancia((x2 - x1) * (y2 - y1), ancho_img * altura_img),
                        'coordenadas': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    }
                    detecciones.append(deteccion)
        return detecciones

    def _generar_descripcion_completa(self, objetos: List[Dict], contexto: List[str]) -> str:
        objetos_validos = [obj for obj in objetos if obj.get('nombre') and obj['nombre'] != 'desconocido_es']

        if not objetos_validos and not contexto:
            return "No se detectan elementos claros en la escena."
        personas = [obj for obj in objetos_validos if obj['nombre'] == 'persona']
        otros_objetos_validos = [obj for obj in objetos_validos if obj['nombre'] != 'persona']
        contador_objetos = Counter(obj['nombre'] for obj in otros_objetos_validos)
        partes = []

        if personas:
            if len(personas) == 1:
                p = personas[0]
                partes.append(f"hay una persona {p.get('distancia_relativa','')} {p.get('posicion','')}".strip())
            else:
                partes.append(f"hay {len(personas)} personas")

        objetos_descritos = []
        if contador_objetos:
            items_ordenados = contador_objetos.most_common()
            for nombre, cantidad in items_ordenados:
                articulo_singular = self.articulos_es.get(nombre, 'un' if not nombre.endswith(('a','s')) else 'una')

                if cantidad == 1:
                    obj = next(o for o in otros_objetos_validos if o['nombre'] == nombre)
                    objetos_descritos.append(f"{articulo_singular} {nombre} {obj.get('posicion',' en posición desconocida')}")
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

        if not partes: return "No se detectan objetos prioritarios en la escena."

        descripcion_final = " ".join(partes).capitalize()
        descripcion_final = re.sub(r'\s*\.+\s*$', '', descripcion_final) + '.'
        descripcion_final = re.sub(r'\s{2,}', ' ', descripcion_final).strip()

        if descripcion_final == "Hay una persona." and personas: # Asegurarse de que 'personas' no esté vacío
             p = personas[0]
             return f"Hay una persona {p.get('distancia_relativa','')} {p.get('posicion','')}".strip().capitalize() + "."


        return descripcion_final

    def analizar(self, imagen: np.ndarray, solo_prioritarios: bool = True) -> Dict:

            objetos_detectados = self._detectar_objetos_principales(imagen, solo_prioritarios)
            contexto_escena = self._analizar_contexto_escena(imagen)
            descripcion_completa = self._generar_descripcion_completa(objetos_detectados, contexto_escena)

            return {
                'objetos': objetos_detectados,
                'contexto': contexto_escena,
                'descripcion': descripcion_completa
            }

    def _analizar_contexto_escena(self, imagen: np.ndarray) -> List[str]:
        if self.modelo_segmentacion is None: return []

        clases_contexto_map = { 'road': 'carretera', 'sidewalk': 'acera', 'building': 'edificios', 'wall': 'muro', 'sky': 'cielo', 'grass': 'césped', 'tree': 'árboles' }
        contexto_detectado = set()
        try:
            resultados = self.modelo_segmentacion(imagen, conf=0.3, verbose=False, device=self.device)

            if resultados and hasattr(resultados[0], 'masks') and resultados[0].masks is not None:
                nombres_modelo_seg = resultados[0].names
                num_masks = len(resultados[0].masks)
                num_boxes = len(resultados[0].boxes) if hasattr(resultados[0], 'boxes') and resultados[0].boxes is not None else 0

                for i in range(min(num_masks, num_boxes)):
                    if resultados[0].boxes[i].cls is None or not resultados[0].boxes[i].cls.nelement(): continue
                    clase_id = int(resultados[0].boxes[i].cls[0])
                    nombre_clase_en = nombres_modelo_seg.get(clase_id)
                    if nombre_clase_en and nombre_clase_en in clases_contexto_map:
                        contexto_detectado.add(clases_contexto_map[nombre_clase_en])
        except Exception as e:
            print(f"Error analizando contexto: {e}")

        return sorted(list(contexto_detectado))


    def _calcular_posicion(self, centro_x: float, centro_y: float, ancho: int, alto: int) -> str:
        if ancho <= 0 or alto <= 0: return "en posición desconocida"
        lim_izq, lim_der = ancho * 0.33, ancho * 0.66
        lim_sup, lim_inf = alto * 0.33, alto * 0.66

        if centro_x < lim_izq: horizontal = "a la izquierda"
        elif centro_x <= lim_der: horizontal = "en el centro"
        else: horizontal = "a la derecha"

        if centro_y < lim_sup: vertical = "arriba"
        elif centro_y <= lim_inf: vertical = ""
        else: vertical = "abajo"

        posicion = f"{vertical} {horizontal}".strip()
        return posicion if posicion else "en el centro"


    def _estimar_distancia(self, area_objeto: float, area_total: float) -> str:
        if area_total <= 0: return "a distancia desconocida"
        porcentaje_area = (area_objeto / area_total) * 100
        if porcentaje_area > 25: return "muy cerca"
        elif porcentaje_area > 10: return "cerca"
        elif porcentaje_area > 4: return "a media distancia"
        else: return "lejos"

    def dibujar_analisis(self, imagen: np.ndarray, analisis: Dict) -> np.ndarray:
        frame_resultado = imagen.copy()
        for det in analisis.get('objetos', []):
            if 'coordenadas' not in det: continue
            coords = det['coordenadas']
            x1, y1, x2, y2 = coords.get('x1',0), coords.get('y1',0), coords.get('x2',0), coords.get('y2',0)
            if x2 > x1 and y2 > y1:
                color = (50, 205, 50)
                grosor = 2
                cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), color, grosor)
                label = f"{det.get('nombre', '???')} ({det.get('confianza', 0):.0%})"
                (w_text, h_text), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                bg_y1_text = max(0, y1 - h_text - 5)
                bg_y2_text = y1

                overlay = frame_resultado.copy()
                cv2.rectangle(overlay, (x1, bg_y1_text), (x1 + w_text + 4, bg_y2_text), (0, 0, 0), -1)
                alpha = 0.65
                cv2.addWeighted(overlay, alpha, frame_resultado, 1 - alpha, 0, frame_resultado)

                text_y = max(h_text + 3, y1 - 3)
                cv2.putText(frame_resultado, label, (x1 + 2, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


        descripcion = analisis.get('descripcion', '')
        y = frame_resultado.shape[0] - 10
        line_height = 0
        try:
            wrapped_text = textwrap.wrap(descripcion, width=60)
            for i, line in enumerate(reversed(wrapped_text)):
                (w, h), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                line_height = h + baseline + 5

                current_y = y - (i * line_height)
                bg_y1 = max(0, current_y - h - baseline - 2)
                bg_y2 = current_y + 3

                overlay_desc = frame_resultado.copy()
                cv2.rectangle(overlay_desc, (5, bg_y1), (15 + w, bg_y2), (0, 0, 0), -1)
                alpha_desc = 0.75
                cv2.addWeighted(overlay_desc, alpha_desc, frame_resultado, 1 - alpha_desc, 0, frame_resultado)
                text_y_desc = max(h + baseline, current_y)
                cv2.putText(frame_resultado, line, (10, text_y_desc), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        except Exception as e:
             print(f"Error al dibujar texto de descripción: {e}")


        return frame_resultado

