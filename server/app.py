from flask import Flask, Response, render_template, stream_with_context
import time
import atexit
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.append(root_dir)

from ejemplo_ocr_integrado import GafasIACompleto
print("flask")
template_folder_path = os.path.join(root_dir, 'templates')
app = Flask(__name__, template_folder=template_folder_path)

print("instancia global")
gafas = GafasIACompleto()
gafas.ejecutando = True 

@stream_with_context 
def generador_web():
    try:
        if not gafas.picam2 or not gafas.picam2.started:
            if not gafas.iniciar_camara():
                print("No se pudo abrir camara en la pagina")
                return
        for frame_bytes in gafas.generar_frames_flask():
            yield frame_bytes
            
    except Exception as e:
        print(f"Error en el generador web de Flask: {e}")
    finally:
        print("El stream de video se ha detenido.")
        

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/video_feed')
def video_feed():
    print("Solicitud recibida para /video_feed")
    return Response(
        generador_web(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def limpiar_al_salir():
    print("Detectado cierre del servidor Flask. Limpiando recursos...")
    gafas.ejecutando = False
    gafas._limpiar_recursos()

atexit.register(limpiar_al_salir)

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
