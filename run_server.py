import cv2
import socket
import struct
import pickle
import threading
import sys
import os
import time
from flask import Flask, render_template, Response
import pyttsx3

# --- FIX: Agregar 'src' al path de Python ---
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# --- IMPORTANTE ---
# Este script espera que las carpetas 'src', 'models', 'templates' y 'static'
# estén en el mismo directorio.

try:
    from ocr.lector_texto import LectorTexto
    from deteccion.analizador_escena import AnalizadorEscena
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de que 'src' contenga las carpetas 'ocr' y 'deteccion'.")
    sys.exit(1)

# --- Configuración del Servidor ---
HOST_IP = '0.0.0.0'
HOST_PORT = 9999
FLASK_PORT = 5000

# --- Variables Globales ---
latest_processed_frame = None
frame_lock = threading.Lock()
objetos_detectados_anteriormente = set()
tts_lock = threading.Lock()

# --- Inicialización de la IA ---
print("Inicializando IA (YOLO + OCR)... Esto puede tardar.")
analizador_ia = AnalizadorEscena(
    modelo_custom_path='models/detecciones/Modelo_V4.pt',
    modelo_seg_path='yolov8n-seg.pt',
    confianza_minima=0.3
)
lector_ocr_ia = LectorTexto(
    idioma='es',
    confianza_minima=40,
    usar_gpu=True
)
print("¡IA inicializada!")

# --- Inicialización del Narrador (TTS) ---
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 160)
    print("Narrador (pyttsx3) inicializado.")
    tts_engine.say("Sistema de narración iniciado.")
    tts_engine.runAndWait()
except Exception as e:
    print(f"Error al inicializar pyttsx3: {e}. El narrador no funcionará.")
    tts_engine = None


# --- Función para narrar en un hilo ---
def narrar_en_hilo(texto):
    """
    Usa pyttsx3 en un hilo separado para no bloquear el stream de video.
    Usa un lock para evitar que hable sobre sí mismo (tartamudeo).
    """
    if not tts_engine:
        return

    if tts_lock.acquire(blocking=False):
        print(f"[NARRADOR DICE]: {texto}")
        try:
            tts_engine.say(texto)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"Error en el hilo del narrador: {e}")
        finally:
            tts_lock.release()
    else:
        print(f"[NARRADOR OCUPADO]: Saltando: '{texto}'")


def obtener_nombre_objeto(obj):
    """
    Extrae el nombre del objeto detectado probando diferentes claves.
    """
    # Probar diferentes posibles nombres de clave
    posibles_claves = ['etiqueta_es', 'label_es', 'clase_es', 'nombre', 'etiqueta', 'label', 'clase', 'name']
    
    for clave in posibles_claves:
        if clave in obj:
            return obj[clave]
    
    # Si no encuentra ninguna clave conocida, intenta convertir el objeto a string para debug
    print(f"[DEBUG] Estructura de objeto no reconocida: {obj}")
    return None


def process_video_stream(conn):
    global latest_processed_frame
    global objetos_detectados_anteriormente
    
    data_unpacker = struct.Struct('L')
    data = b""
    payload_size = data_unpacker.size

    try:
        while True:
            # Recibir el frame de la Pi
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

            # Procesamiento de IA
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            analisis_escena = analizador_ia.analizar(frame_rgb, solo_prioritarios=True)
            textos = lector_ocr_ia.detectar_texto(frame_rgb, mejorar_imagen=False)
            
            # --- Lógica del Narrador ---
            try:
                objetos_lista = analisis_escena.get('objetos', [])
                
                # Obtener nombres de objetos actuales
                nombres_objetos_actuales = set()
                for obj in objetos_lista:
                    nombre = obtener_nombre_objeto(obj)
                    if nombre:
                        nombres_objetos_actuales.add(nombre)
                
                # Comparamos con los que ya habíamos visto
                nuevos_objetos = nombres_objetos_actuales - objetos_detectados_anteriormente
                
                if nuevos_objetos:
                    # Si hay objetos nuevos, los narramos
                    texto_a_decir = "Veo " + ", ".join(nuevos_objetos)
                    
                    # Iniciar el narrador en un hilo para no bloquear el video
                    threading.Thread(target=narrar_en_hilo, args=(texto_a_decir,), daemon=True).start()
                    
                    # Actualizar el estado para no repetir
                    objetos_detectados_anteriormente = nombres_objetos_actuales
                
                # Si no se ve nada, reseteamos el set para que la próxima vez lo vuelva a decir
                if not nombres_objetos_actuales and objetos_detectados_anteriormente:
                    objetos_detectados_anteriormente.clear()

            except Exception as e:
                print(f"Error en la lógica del narrador: {e}")
                import traceback
                traceback.print_exc()

            # Dibujo de resultados en el frame
            frame_a_mostrar = frame_bgr
            if analisis_escena.get('objetos'):
                frame_a_mostrar = analizador_ia.dibujar_analisis(frame_a_mostrar, analisis_escena)
            if textos:
                frame_a_mostrar = lector_ocr_ia.dibujar_texto_detectado(frame_a_mostrar, textos)

            # Codificación JPEG
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


# --- Aplicación Flask ---
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
        
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(
        flask_frame_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    # Iniciar el hilo que escucha a la Raspberry Pi
    pi_listener_thread = threading.Thread(target=start_pi_listener)
    pi_listener_thread.daemon = True
    pi_listener_thread.start()

    print(f"Iniciando servidor Flask. Abre http://127.0.0.1:{FLASK_PORT} en tu navegador.")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)

    print("Servidor detenido.")
