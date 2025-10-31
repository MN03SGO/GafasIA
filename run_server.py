import cv2
import socket
import struct
import pickle
import threading
import sys
import os
import time
from flask import Flask, render_template, Response

# --- IMPORTANTE ---
# Este script espera que las carpetas 'src', 'models', 'templates' y 'static'
# estén en el mismo directorio.

try:
    from src.ocr.lector_texto import LectorTexto
    from src.deteccion.analizador_escena import AnalizadorEscena
except ImportError:
    print("Error: No se encontraron las carpetas 'src' o 'models'.")
    print("Asegúrate de copiar 'src' y 'models' de tu Pi a esta carpeta.")
    sys.exit(1)

# --- Configuración del Servidor ---
HOST_IP = '0.0.0.0' # Escuchar en todas las IPs
HOST_PORT = 9999    # Puerto para recibir de la Pi
FLASK_PORT = 5000   # Puerto para ver en el navegador

# --- Variables Globales (para compartir entre hilos) ---
# Aquí guardaremos el último frame procesado para enviarlo al navegador
latest_processed_frame = None
frame_lock = threading.Lock() # Para evitar que dos hilos escriban a la vez

# --- Inicialización de la IA (¡en la Laptop!) ---
print("Inicializando IA (YOLO + OCR)... Esto puede tardar.")
analizador_ia = AnalizadorEscena(
    modelo_custom_path='models/detecciones/Modelo_V4.pt',
    modelo_seg_path='yolov8n-seg.pt',
    confianza_minima=0.3
)
lector_ocr_ia = LectorTexto(
    idioma='es',
    confianza_minima=40,
    usar_gpu=True # ¡Pon True si tu laptop tiene GPU NVIDIA! (o False si no)
)
print("¡IA inicializada!")
# --- Fin de la IA ---


def process_video_stream(conn):
    """
    Función que corre en un hilo. Recibe frames de la Pi, los procesa
    con IA y los guarda en la variable global.
    """
    global latest_processed_frame
    
    data_unpacker = struct.Struct('L')
    data = b""
    payload_size = data_unpacker.size

    try:
        while True:
            # 1. Leer el tamaño del frame que viene
            while len(data) < payload_size:
                packet = conn.recv(4096)
                if not packet:
                    raise ConnectionAbortedError("Pi desconectada.")
                data += packet
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = data_unpacker.unpack(packed_msg_size)[0]

            # 2. Leer el frame completo
            while len(data) < msg_size:
                data += conn.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            # 3. Deserializar y decodificar
            frame_encoded = pickle.loads(frame_data, fix_imports=True, encoding="bytes")
            frame_bgr = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)

            if frame_bgr is None:
                continue

            # 4. ¡Procesar con IA! (El trabajo pesado)
            # (Convertimos a RGB para YOLO y OCR)
            # --- ¡CORRECCIÓN AQUÍ! ---
            # Había un error de tipeo: decía 'cvV2' en lugar de 'cv2'
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Correr análisis de escena
            analisis_escena = analizador_ia.analizar(frame_rgb, solo_prioritarios=True)
            
            # Correr OCR
            textos = lector_ocr_ia.detectar_texto(frame_rgb, mejorar_imagen=False)
            
            # 5. Dibujar los resultados (sobre el frame BGR)
            frame_a_mostrar = frame_bgr
            if analisis_escena.get('objetos'):
                frame_a_mostrar = analizador_ia.dibujar_analisis(frame_a_mostrar, analisis_escena)
            if textos:
                frame_a_mostrar = lector_ocr_ia.dibujar_texto_detectado(frame_a_mostrar, textos)

            # 6. Codificar a JPEG para el navegador
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90] # 90% de calidad
            (flag, encodedImage) = cv2.imencode(".jpg", frame_a_mostrar, encode_param)

            if not flag:
                continue

            # 7. Guardar en la variable global
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
    """
    Función que corre en un hilo. Espera la conexión de la Pi.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, HOST_PORT))
    server_socket.listen(1)
    print(f"Servidor escuchando a la Pi en {HOST_IP}:{HOST_PORT}")

    while True:
        try:
            conn, addr = server_socket.accept()
            print(f"¡Pi conectada desde {addr}!")
            # Iniciar un hilo dedicado para procesar el video de esta Pi
            processing_thread = threading.Thread(
                target=process_video_stream,
                args=(conn,)
            )
            processing_thread.daemon = True # Morir si el programa principal muere
            processing_thread.start()
        except Exception as e:
            print(f"Error al aceptar conexión de la Pi: {e}")
    server_socket.close()


# --- Configuración del Servidor Web (Flask) ---
# (Esto es tu 'app.py' ahora integrado aquí)
app = Flask(__name__) # Flask buscará 'templates' y 'static' aquí

@app.route('/')
def index():
    """Página principal que muestra el video."""
    return render_template('index.html')

def flask_frame_generator():
    """Generador que envía los frames procesados al navegador."""
    while True:
        with frame_lock:
            if latest_processed_frame is None:
                # Esperar si aún no hay frames
                time.sleep(0.1)
                continue
            frame_bytes = latest_processed_frame
        
        # Enviar el frame
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Dormir un poco para no saturar
        time.sleep(0.03) # Limitar a ~30 FPS

@app.route('/video_feed')
def video_feed():
    """Ruta que envía el stream de video."""
    return Response(
        flask_frame_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
# --- Fin de Flask ---


if __name__ == '__main__':
    # --- Paso 1: Iniciar el hilo que escucha a la Pi ---
    pi_listener_thread = threading.Thread(target=start_pi_listener)
    pi_listener_thread.daemon = True
    pi_listener_thread.start()

    # --- Paso 2: Iniciar el servidor Flask (en el hilo principal) ---
    print(f"Iniciando servidor Flask. Abre http://127.0.0.1:{FLASK_PORT} en tu navegador.")
    # 'threaded=True' es importante para que Flask maneje múltiples peticiones
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)

    print("Servidor detenido.")
