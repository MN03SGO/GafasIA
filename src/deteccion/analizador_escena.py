import cv2
import numpy as np
from ultralytics import YOLO
import torch
from typing import List, Dict
from collections import Counter

class AnalizadorEscena:

    def __init__(self, modelo_det_path: str = 'yolov8n.pt', modelo_seg_path: str = 'yolov8n-seg.pt', 
                    confianza_minima: float = 0.5, modelo_custom_path: str = None):


        print("Inicializando Analizador de Escena...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {self.device}")

        if modelo_custom_path:
            print(f"Cargando modelo de detección PERSONALIZADO desde: {modelo_custom_path}")
            self.modelo_deteccion = YOLO(modelo_custom_path)
        else:
            print(f"Cargando modelo de detección ESTÁNDAR desde: {modelo_det_path}")
            self.modelo_deteccion = YOLO(modelo_det_path)

        self.modelo_segmentacion = YOLO(modelo_seg_path)
        self.confianza_minima = confianza_minima

        self.etiquetas_es = {
            0: 'persona', 1: 'bicicleta', 2: 'automóvil', 3: 'motocicleta', 4: 'avión', 5: 'autobús', 6: 'tren', 7: 'camión', 8: 'barco', 9: 'semáforo', 10: 'boca de incendios', 11: 'señal de alto', 12: 'parquímetro', 13: 'banco', 14: 'pájaro', 15: 'gato', 16: 'perro', 17: 'caballo', 18: 'oveja', 19: 'vaca', 20: 'elefante', 21: 'oso', 22: 'cebra', 23: 'jirafa', 24: 'mochila', 25: 'paraguas', 26: 'bolso', 27: 'corbata', 28: 'maleta', 29: 'frisbee', 30: 'esquís', 31: 'tabla de snowboard', 32: 'pelota deportiva', 33: 'cometa', 34: 'bate de béisbol', 35: 'guante de béisbol', 36: 'patineta', 37: 'tabla de surf', 38: 'raqueta de tenis', 39: 'botella', 40: 'copa de vino', 41: 'taza', 42: 'tenedor', 43: 'cuchillo', 44: 'cuchara', 45: 'tazón', 46: 'plátano', 47: 'manzana', 48: 'sándwich', 49: 'naranja', 50: 'brócoli', 51: 'zanoria', 52: 'perro caliente', 53: 'pizza', 54: 'dona', 55: 'pastel', 56: 'silla', 57: 'sofá', 58: 'planta en maceta', 59: 'cama', 60: 'mesa de comedor', 61: 'inodoro', 62: 'televisor', 63: 'computadora portátil', 64: 'ratón', 65: 'control remoto', 66: 'teclado', 67: 'celular', 68: 'microondas', 69: 'horno', 70: 'tostadora', 71: 'fregadero', 72: 'refrigerador', 73: 'libro', 74: 'reloj', 75: 'florero', 76: 'tijeras', 77: 'osito de peluche', 78: 'secador de cabello', 79: 'cepillo de dientes'
        }

        # Nuevo set escaleras 
        nuevas_etiquetas = {
            80: 'escaleras',
            # 81: 'puerta',
        }
        self.etiquetas_es.update(nuevas_etiquetas)

        self.objetos_prioritarios = { 0, 2, 3, 5, 24, 39, 41, 56, 57, 58, 59, 60, 61, 62, 63, 65, 67, 73, 80}
        self.objetos_prioritarios.update(nuevas_etiquetas.keys())

        print("Analizador inicializado correctamente.")

    def analizar(self, imagen: np.ndarray, solo_prioritarios: bool = True) -> Dict:
        try:
            objetos_detectados = self._detectar_objetos_principales(imagen, solo_prioritarios)
            contexto_escena = self._analizar_contexto_escena(imagen)
            descripcion_completa = self._generar_descripcion_completa(objetos_detectados, contexto_escena)
            
            return {
                'objetos': objetos_detectados,
                'contexto': contexto_escena,
                'descripcion': descripcion_completa
            }
        except Exception as e:
            print(f"Error en el análisis de escena: {e}")
            return {'objetos': [], 'contexto': [], 'descripcion': 'Error en el análisis.'}

    def _detectar_objetos_principales(self, imagen: np.ndarray, solo_prioritarios: bool) -> List[Dict]:
        resultados = self.modelo_deteccion(imagen, conf=self.confianza_minima, verbose=False)
        detecciones = []
        altura_img, ancho_img = imagen.shape[:2]

        for res in resultados:
            for box in res.boxes:
                clase_id = int(box.cls[0])
                if solo_prioritarios and clase_id not in self.objetos_prioritarios:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                centro_x, centro_y = (x1 + x2) / 2, (y1 + y2) / 2
                deteccion = {
                    'nombre': self.etiquetas_es.get(clase_id, f'objeto_{clase_id}'),
                    'confianza': round(float(box.conf[0]), 2),
                    'posicion': self._calcular_posicion(centro_x, centro_y, ancho_img, altura_img),
                    'distancia_relativa': self._estimar_distancia(area, ancho_img * altura_img),
                    'coordenadas': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                }
                detecciones.append(deteccion)
        return detecciones

    def _analizar_contexto_escena(self, imagen: np.ndarray) -> List[str]:
        clases_contexto = { 'road': 'una carretera', 'sidewalk': 'una acera', 'building': 'edificios', 'wall': 'un muro', 'sky': 'el cielo', 'grass': 'césped', 'tree': 'árboles'}
        resultados = self.modelo_segmentacion(imagen, conf=0.3, verbose=False)
        contexto_detectado = set()
        if resultados[0].masks is not None:
            nombres_modelo = resultados[0].names
            for i in range(len(resultados[0].masks)):
                clase_id = int(resultados[0].boxes[i].cls[0])
                nombre_clase_en = nombres_modelo[clase_id]
                if nombre_clase_en in clases_contexto:
                    contexto_detectado.add(clases_contexto[nombre_clase_en])
        return sorted(list(contexto_detectado))

    def _generar_descripcion_completa(self, objetos: List[Dict], contexto: List[str]) -> str:
        if not objetos and not contexto:
            return "No se detectan elementos claros en la escena."
        personas = [obj for obj in objetos if obj['nombre'] == 'persona']
        otros_objetos = [obj for obj in objetos if obj['nombre'] != 'persona']
        contador_objetos = Counter(obj['nombre'] for obj in otros_objetos)
        partes_descripcion = []

        if personas:
            if len(personas) == 1:
                det_persona = personas[0]
                partes_descripcion.append(f"hay una persona {det_persona['distancia_relativa']} {det_persona['posicion']}")
            else:
                partes_descripcion.append(f"hay {len(personas)} personas")
        
        objetos_descritos = []
        if contador_objetos:
            for nombre, cantidad in contador_objetos.most_common():
                if cantidad == 1:
                    obj = next(o for o in otros_objetos if o['nombre'] == nombre)
                    objetos_descritos.append(f"un {nombre} {obj['posicion']}")
                else:
                    plural = nombre + ('es' if nombre.endswith(('n', 'l', 'r', 's', 'd', 'z', 'j')) else 's')
                    objetos_descritos.append(f"{cantidad} {plural}")
        
        if objetos_descritos:
            if partes_descripcion: partes_descripcion.append("además de")
            else: partes_descripcion.append("veo")
            partes_descripcion.append(", ".join(objetos_descritos))

        if contexto:
            contexto_str = f"El entorno parece tener {', '.join(contexto)}"
            if partes_descripcion: partes_descripcion[-1] += "."
            partes_descripcion.append(contexto_str)

        if not partes_descripcion:
            return f"El entorno parece tener {', '.join(contexto)}."
            
        descripcion_final = " ".join(partes_descripcion).capitalize()
        if not descripcion_final.endswith('.'): descripcion_final += '.'
        return descripcion_final

    def _calcular_posicion(self, centro_x: float, centro_y: float, ancho: int, alto: int) -> str:
        tercio_ancho, tercio_alto = ancho / 3, alto / 3
        if centro_x < tercio_ancho: horizontal = "a la izquierda"
        elif centro_x < 2 * tercio_ancho: horizontal = "en el centro"
        else: horizontal = "a la derecha"
        if centro_y < tercio_alto: vertical = "arriba"
        elif centro_y < 2 * tercio_alto: vertical = "en el medio"
        else: vertical = "abajo"
        if horizontal == "en el centro" and vertical == "en el medio": return "en el centro"
        return f"{vertical} {horizontal}"

    def _estimar_distancia(self, area_objeto: float, area_total: float) -> str:
        porcentaje_area = (area_objeto / area_total) * 100
        if porcentaje_area > 20: return "muy cerca"
        elif porcentaje_area > 8: return "cerca"
        elif porcentaje_area > 3: return "a media distancia"
        else: return "lejos"

    def dibujar_analisis(self, imagen: np.ndarray, analisis: Dict) -> np.ndarray:
        frame_resultado = imagen.copy()
        for det in analisis.get('objetos', []):
            coords = det['coordenadas']
            x1, y1, x2, y2 = coords['x1'], coords['y1'], coords['x2'], coords['y2']
            cv2.rectangle(frame_resultado, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det['nombre']} {det['confianza']:.2f}"
            cv2.putText(frame_resultado, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        descripcion = analisis.get('descripcion', '')
        (w, h), _ = cv2.getTextSize(descripcion, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame_resultado, (5, 5), (15 + w, 35 + h), (0,0,0), -1)
        cv2.putText(frame_resultado, descripcion, (10, 30 + h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        return frame_resultado

