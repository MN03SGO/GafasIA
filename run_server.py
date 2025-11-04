import cv2
import socket
import struct
import pickle
import threading
import sys
import os
import time
from flask import Flask, render_template, Response
# --- CAMBIO: Importamos tu clase desde 'src.audio' ---
from src.audio.sintetizador_voz import SintetizadorVoz


# --- FIX: Agregar 'src' al path de Python ---
# (Asegurarse de que 'src' esté en el path para encontrar 'audio')
sys.path.insert(0, os.path.join(os.path.dirname(__file__))) 


try:
    from src.ocr.lector_texto import LectorTexto
    from src.deteccion.analizador_escena import AnalizadorEscena
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    # --- CAMBIO: Mensaje de error corregido ---
    print("Asegúrate de que 'src' contenga 'ocr', 'deteccion' y 'audio'.")
    sys.exit(1)


HOST_IP = '0.0.0.0'
HOST_PORT = 9999
FLASK_PORT = 5000


latest_processed_frame = None
frame_lock = threading.Lock()
objetos_detectados_anteriormente = set()

print("Ravision")
analizador_ia = AnalizadorEscena(
    modelo_custom_path='models/detecciones/Modelo_V4.pt',
    modelo_seg_path='yolov8n-seg.pt',
    confianza_minima=0.5
)
lector_ocr_ia = LectorTexto(
    idioma='es',
    confianza_minima=40,
    usar_gpu=True
)
print("¡IA inicializada!")

# --- CAMBIO: Inicialización de tu Narrador ---
try:
    print("Inicializando Narrador (SintetizadorVoz)...")
    # Ajusta la velocidad aquí si quieres (ej. 140, 160, 170)
    narrador = SintetizadorVoz(idioma='es', velocidad=140) 
    if narrador.disponible:
        narrador.decir_inicio()
    else:
        print("El narrador no pudo iniciarse.")

except Exception as e:
    print(f"Error al inicializar SintetizadorVoz: {e}. El narrador no funcionará.")
    narrador = None
# --- FIN DEL CAMBIO ---


def process_video_stream(conn):
    global latest_processed_frame
    global objetos_detectados_anteriormente
    
    data_unpacker = struct.Struct('L')
    data = b""
    payload_size = data_unpacker.size

    try:
        while True:
            # (El código de recepción de frames sigue igual...)
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionAbortedError("Pi desconectada.")
                data += packet
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = data_unpacker.unpack(packed_msg_size)[0]

            while len(data) < msg_size:
                data += conn.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame_encoded = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame_bgr = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                continue

            # Procesamiento de yolo
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            analisis_escena = analizador_ia.analizar(frame_rgb, solo_prioritarios=True)
            textos = lector_ocr_ia.detectar_texto(frame_rgb, mejorar_imagen=False)

            # --- LÓGICA DEL NARRADOR (Adaptada a tu clase) ---
            if narrador and narrador.disponible:
                try:
                    # ¡Tu función robusta!
                    def obtener_nombre_objeto(obj):
                        posibles_claves = ['etiqueta_es', 'label_es', 'clase_es', 'nombre', 'etiqueta', 'label', 'clase', 'name']
                        for clave in posibles_claves:
                            if clave in obj:
                                return obj[clave]
                        print(f"[DEBUG] Estructura de objeto no reconocida: {obj}")
                        return None

                    objetos_lista = analisis_escena.get('objetos', [])
                    nombres_objetos_actuales = set()
                    for obj in objetos_lista:
                        nombre = obtener_nombre_objeto(obj)
                        if nombre:
                            nombres_objetos_actuales.add(nombre)

                    
                    nuevos_objetos = nombres_objetos_actuales - objetos_detectados_anteriormente
                    
                    # ¡Usamos tu clase!
                    # Solo habla si hay objetos nuevos Y el narrador no está ya hablando
                    if nuevos_objetos and not narrador.esta_hablando():
                        texto_a_decir = "Veo " + ", ".join(nuevos_objetos)
                        objetos_detectados_anteriormente = nombres_objetos_actuales
                        # Tu clase ya maneja el hilo, solo llamamos a 'decir'
                        narrador.decir(texto_a_decir, prioridad=True) 
                    
                    # Si la escena está vacía Y el narrador no está ocupado, resetea la memoria
                    elif not nombres_objetos_actuales and objetos_detectados_anteriormente and not narrador.esta_hablando():
                        print("[DEBUG Narrador] Escena vacía y narrador libre, reseteando memoria.")
                        objetos_detectados_anteriormente.clear()
                    
                    # Si hay objetos nuevos PERO el narrador está ocupado, lo imprimimos
                    elif nuevos_objetos and narrador.esta_hablando():
                        print(f"[NARRADOR OCUPADO]: Saltando: 'Veo {', '.join(nuevos_objetos)}'")

                except Exception as e:
                    print(f"Error en la lógica del narrador: {e}")
            # --- FIN DE LÓGICA DEL NARRADOR ---

            # (El código de dibujar y codificar sigue igual...)
            frame_a_mostrar = frame_bgr
            if analisis_escena.get('objetos'):
                frame_a_mostrar = analizador_ia.dibujar_analisis(frame_a_mostrar, analisis_escena)
            if textos:
                frame_a_mostrar = lector_ocr_ia.dibujar_texto_detectado(frame_a_mostrar, textos)

            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
            (flag, encodedImage) = cv2.imencode(".jpg", frame_a_mostrar, encode_param)

            if not flag:
                continue

            with frame_lock:
                latest_processed_frame = encodedImage.tobytes()

    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError) as e:
        print(f"Se perdió la conexión con la Pi: {e}")
    except Exception as e:
        print(f"Error en el hilo de procesamiento: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cerrando conexión con la Pi.")
        conn.close()


def start_pi_listener():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((HOST_IP, HOST_PORT))
    server_socket.listen(1)
    print(f"Servidor escuchando a la Pi en {HOST_IP}:{HOST_PORT}")

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"¡Pi conectada desde {addr}!")
            processing_thread = threading.Thread(
                target=process_video_stream,
                args=(conn,)
            )
            processing_thread.daemon = True
            processing_thread.start()
        except Exception as e:
            print(f"Error al aceptar conexión de la Pi: {e}")
        except KeyboardInterrupt:
            print("\nCerrando servidor (Ctrl+C).")
            break
    server_socket.close()


# Flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def flask_frame_generator():
    while True:
        with frame_lock:
            if latest_processed_frame is None:
                time.sleep(0.1)
                continue
            frame_bytes = latest_processed_frame
        
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03) # Limitar a ~30 FPS

@app.route('/video_feed')
def video_feed():
    return Response(
        flask_frame_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    pi_listener_thread = threading.Thread(target=start_pi_listener)
    pi_listener_thread.daemon = True
    pi_listener_thread.start()

    print(f"Iniciando servidor Flask. Abre http://127.0.0.1:{FLASK_PORT} en tu navegador.")

    try:
        app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nDeteniendo servidor Flask (Ctrl+C).")
    finally:
        # --- CAMBIO: Asegurarse de finalizar el narrador ---
        if narrador and narrador.disponible:
            narrador.finalizar()

    print("Servidor detenido.")

