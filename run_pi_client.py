import cv2
import socket
import struct
import pickle
import time
import sys

LAPTOP_IP = 'IP DE LA RED EN LA QUE TE CONECTAS'  
LAPTOP_PORT = 9999

WIDTH = 640
HEIGHT = 480
FPS = 30
JPEG_QUALITY = 90

try:
    from picamera2 import Picamera2
except ImportError:
    print("No se encontró la librería 'picamera2'.")
    print("Asegúrate de instalarla con: pip install picamera2")
    sys.exit(1)

def main():
    try:
        print("Inicializando Picamera2...")
        picam2 = Picamera2()
        video_config = picam2.create_video_configuration(
            main={"size": (WIDTH, HEIGHT), "format": "RGB888"},
            controls={"FrameRate": FPS}
        )
        picam2.configure(video_config)
        picam2.start()
        print(f"Picamera2 iniciada ({WIDTH}x{HEIGHT} @ {FPS}fps). (CTRL+C para salir)")
    except Exception as e:
        print(f"Error fatal al iniciar la cámara: {e}")
        print("Asegúrate de que la cámara esté bien conectada y habilitada en 'raspi-config'.")
        sys.exit(1)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    data_packer = struct.Struct('L')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]

    while True:
        try:
            print(f"Intentando conectar a {LAPTOP_IP}:{LAPTOP_PORT}...")
            client_socket.connect((LAPTOP_IP, LAPTOP_PORT))
            print(f"¡Conectado a la PC ({LAPTOP_IP})!")
            break
        except ConnectionRefusedError:
            print("Conexión rechazada. ¿Está el servidor (run_server.py) corriendo en la PC?")
            print("Reintentando en 5 segundos...")
            time.sleep(5)
        except Exception as e:
            print(f"Error al conectar: {e}")
            time.sleep(5)

    try:
        while True:
           
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            (flag, frame_encoded) = cv2.imencode(".jpg", frame_bgr, encode_param)
            if not flag:
                continue
            data = pickle.dumps(frame_encoded, 0)
            msg_size = len(data)
            client_socket.sendall(data_packer.pack(msg_size))
            client_socket.sendall(data)

    except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError) as e:
        print(f"Se perdió la conexión con la PC: {e}")
    except KeyboardInterrupt:
        print("\nCerrando.")
    except Exception as e:
        print(f"Error inesperado: {e}")
    finally:
        print("Cerrando conexión y cámara.")
        client_socket.close()
        picam2.stop()
if __name__ == '__main__':
    main()

