
import cv2
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent))

from utils.camera import Camera
from utils.config import config

class GafasIA:
    def __init__(self):
        self.camera = Camera()
        self.running = False
        print(" Gafas IA inicializadas")
        print(f" Configuración cargada desde: {config.config_path}")
    
    def test_camera(self):
        #Cmara basica
        print("\n🎥 Iniciando prueba de cámara...")
        print(" Controles:")
        print("   - Presiona 'q' para salir")
        print("   - Presiona 'i' para información de cámara")
        print("   - Presiona 'c' para capturar imagen")
        
        if not self.camera.start():
            print(" Error: No se pudo iniciar la cámara")
            return
        
        # Mostrar información de la cámara
        info = self.camera.get_info()
        print(f" Cámara: {info['width']}x{info['height']} @ {info['fps']}fps")
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                
                if not ret:
                    print(" Error leyendo frame")
                    break
                
                frame_count += 1
                
                # Añadir información al frame
                cv2.putText(frame, f"Gafas IA - Frame: {frame_count}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "Presiona 'q' para salir", 
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.imshow('Gafas IA - Test Camera', frame)
                
                # Manejo de teclas
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(" Saliendo...")
                    break
                elif key == ord('i'):
                    print(f"Info: {info}")
                elif key == ord('c'):
                    filename = f"capture_{frame_count}.jpg"
                    cv2.imwrite(f"assets/images/{filename}", frame)
                    print(f" Imagen capturada: {filename}")
        
        except KeyboardInterrupt:
            print("\nDetenido por usuario")
        
        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            print(" Prueba de cámara completada")
    
    def run(self):
        
        print("\nIniciando Gafas IA...")
        
        #Lanzamiento de camara laptop
        #--Cambiar el sabado a esp cam
        self.test_camera()
    
    def stop(self):
        
        self.running = False

def main():
    print("=" * 50)
    print(" Asistencia visual")
    print("   Proyecto de bachillerato by sigaran")
    print("=" * 50)

    print(f" Configuración:")
    print(f"   - Cámara: ID {config.get('camera.device_id')}")
    print(f"   - Resolución: {config.get('camera.width')}x{config.get('camera.height')}")
    print(f"   - Debug: {config.get('app.debug')}")
    
    try:
        gafas = GafasIA()
        gafas.run()
    except Exception as e:
        print(f" Error crítico: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()