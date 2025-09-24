# diagnostico_gafas_ia.py - Script para diagnosticar problemas específicos
import cv2
import time
from src.deteccion.detector_objetos import DetectorObjetos
from src.ocr.lector_texto import LectorTexto
from src.audio.sintetizador_voz import SintetizadorVoz

class DiagnosticoGafasIA:
    def __init__(self):
        """Inicializa el sistema de diagnóstico"""
        print("🔧 Iniciando diagnóstico del sistema Gafas IA...")
        
        self.detector = DetectorObjetos(modelo_path='yolov8n.pt', confianza_minima=0.3)  # Confianza más baja
        self.lector_ocr = LectorTexto(idioma='spa', confianza_minima=30)  # Confianza más baja
        self.sintetizador = SintetizadorVoz(idioma='es', velocidad=180, volumen=0.8)
        
        print("✅ Componentes cargados para diagnóstico")
    
    def diagnosticar_deteccion_objetos(self):
        """Diagnostica específicamente la detección de objetos"""
        print("\n" + "="*50)
        print("🔍 DIAGNÓSTICO: DETECCIÓN DE OBJETOS")
        print("="*50)
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("📷 Cámara inicializada. Presiona:")
        print("  'd' → Debug completo (todos los objetos)")
        print("  'p' → Solo objetos prioritarios")
        print("  'c' → Cambiar confianza mínima")
        print("  'q' → Salir")
        
        confianza_actual = 0.3
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mostrar configuración actual
            texto_config = f"Confianza: {confianza_actual:.1f}"
            cv2.putText(frame, texto_config, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Diagnostico - Objetos', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                print(f"\n🔍 DEBUG COMPLETO (confianza: {confianza_actual}):")
                self.detector.confianza_minima = confianza_actual
                detecciones = self.detector.detectar(frame, solo_prioritarios=False, debug=True)
                
                if detecciones:
                    frame_debug = self.detector.dibujar_detecciones(frame, detecciones)
                    cv2.imshow('Diagnostico - Objetos', frame_debug)
                    
                    # Mostrar estadísticas
                    print(f"📊 ESTADÍSTICAS:")
                    print(f"  Total detectado: {len(detecciones)}")
                    clases_detectadas = {}
                    for det in detecciones:
                        clase = det['nombre']
                        if clase in clases_detectadas:
                            clases_detectadas[clase] += 1
                        else:
                            clases_detectadas[clase] = 1
                    
                    for clase, cantidad in clases_detectadas.items():
                        print(f"  {clase}: {cantidad}")
                else:
                    print("❌ NO SE DETECTARON OBJETOS")
                    print("💡 Prueba:")
                    print("   - Acercar objetos a la cámara")
                    print("   - Mejorar iluminación")
                    print("   - Presionar 'c' para reducir confianza")
            
            elif key == ord('p'):
                print(f"\n🎯 OBJETOS PRIORITARIOS (confianza: {confianza_actual}):")
                self.detector.confianza_minima = confianza_actual
                detecciones = self.detector.detectar(frame, solo_prioritarios=True, debug=True)
                
                if detecciones:
                    frame_prioritarios = self.detector.dibujar_detecciones(frame, detecciones)
                    cv2.imshow('Diagnostico - Objetos', frame_prioritarios)
                    print(f"✅ {len(detecciones)} objetos prioritarios detectados")
                else:
                    print("❌ No se detectaron objetos prioritarios")
            
            elif key == ord('c'):
                # Cambiar confianza
                confianzas = [0.1, 0.3, 0.5, 0.7]
                idx_actual = confianzas.index(confianza_actual) if confianza_actual in confianzas else 1
                confianza_actual = confianzas[(idx_actual + 1) % len(confianzas)]
                print(f"🎚️ Confianza cambiada a: {confianza_actual}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def diagnosticar_ocr(self):
        """Diagnostica específicamente el sistema OCR"""
        print("\n" + "="*50)
        print("📖 DIAGNÓSTICO: SISTEMA OCR")
        print("="*50)
        
        # Inicializar cámara
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("📷 Cámara inicializada. Presiona:")
        print("  't' → Detectar texto (debug completo)")
        print("  'n' → Sin mejoras de imagen")
        print("  'm' → Con mejoras de imagen")
        print("  'v' → Ver imagen procesada")
        print("  'q' → Salir")
        
        mostrar_procesada = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_mostrar = frame.copy()
            
            # Indicador de modo
            texto_modo = "Procesada" if mostrar_procesada else "Original"
            cv2.putText(frame_mostrar, texto_modo, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Diagnostico - OCR', frame_mostrar)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                print(f"\n📖 ANÁLISIS OCR COMPLETO:")
                textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True, debug=True)
                
                if textos:
                    frame_ocr = self.lector_ocr.dibujar_texto_detectado(frame, textos)
                    cv2.imshow('Diagnostico - OCR', frame_ocr)
                    
                    print(f"✅ {len(textos)} textos detectados:")
                    for i, texto in enumerate(textos[:5], 1):
                        print(f"  {i}. '{texto['texto']}' ({texto['confianza']}%)")
                        print(f"     Categoría: {texto['categoria']}")
                        print(f"     Posición: {texto['posicion']}")
                        print(f"     Coords: {texto['coordenadas']}")
                    
                    # Probar audio
                    descripcion = self.lector_ocr.generar_descripcion_audio(textos, modo='completo')
                    print(f"🔊 Audio: '{descripcion}'")
                    self.sintetizador.decir(descripcion)
                else:
                    print("❌ NO SE DETECTÓ TEXTO")
                    print("💡 Sugerencias:")
                    print("   - Apuntar a texto claro y grande")
                    print("   - Mejorar iluminación")
                    print("   - Evitar reflejos en el texto")
            
            elif key == ord('n'):
                print(f"\n📖 OCR SIN MEJORAS:")
                textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=False, debug=True)
                print(f"Resultado: {len(textos)} textos detectados")
            
            elif key == ord('m'):
                print(f"\n📖 OCR CON MEJORAS:")
                textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True, debug=True)
                print(f"Resultado: {len(textos)} textos detectados")
            
            elif key == ord('v'):
                mostrar_procesada = not mostrar_procesada
                if mostrar_procesada:
                    # Mostrar imagen procesada
                    imagen_procesada = self.lector_ocr._mejorar_imagen_ocr(frame)
                    if len(imagen_procesada.shape) == 2:
                        imagen_procesada = cv2.cvtColor(imagen_procesada, cv2.COLOR_GRAY2BGR)
                    cv2.imshow('Diagnostico - OCR', imagen_procesada)
                print(f"👁️ Mostrando imagen {'procesada' if mostrar_procesada else 'original'}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def test_modo_combinado(self):
        """Prueba específica para el modo combinado"""
        print("\n" + "="*50)
        print("🔄 DIAGNÓSTICO: MODO COMBINADO")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("📷 Prueba del modo combinado. Presiona:")
        print("  'ESPACIO' → Análisis combinado")
        print("  'q' → Salir")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            cv2.imshow('Test Modo Combinado', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Espacio
                print(f"\n🔄 PRUEBA MODO COMBINADO:")
                
                # Cronometrar cada paso
                inicio_total = time.time()
                
                try:
                    # Paso 1: Objetos
                    print("  🔍 Paso 1/3: Detectando objetos...")
                    inicio_objetos = time.time()
                    detecciones = self.detector.detectar(frame, solo_prioritarios=True, debug=False)
                    tiempo_objetos = time.time() - inicio_objetos
                    print(f"    ✅ {len(detecciones)} objetos en {tiempo_objetos:.2f}s")
                    
                    # Paso 2: Texto (solo si no hay muchos objetos)
                    textos = []
                    if len(detecciones) < 5:
                        print("  📖 Paso 2/3: Detectando texto...")
                        inicio_texto = time.time()
                        textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True, debug=False)
                        tiempo_texto = time.time() - inicio_texto
                        print(f"    ✅ {len(textos)} textos en {tiempo_texto:.2f}s")
                    else:
                        print("  ⏭️ Paso 2/3: OCR omitido (muchos objetos)")
                    
                    # Paso 3: Visualizar
                    print("  🎨 Paso 3/3: Dibujando resultados...")
                    frame_resultado = frame.copy()
                    
                    if detecciones:
                        frame_resultado = self.detector.dibujar_detecciones(frame_resultado, detecciones)
                    
                    if textos:
                        frame_resultado = self.lector_ocr.dibujar_texto_detectado(frame_resultado, textos)
                    
                    cv2.imshow('Test Modo Combinado', frame_resultado)
                    
                    tiempo_total = time.time() - inicio_total
                    print(f"  ⏱️ TOTAL: {tiempo_total:.2f}s")
                    
                    # Mensaje de audio combinado
                    if detecciones or textos:
                        if detecciones and textos:
                            msg = f"Veo {len(detecciones)} objetos y {len(textos)} textos"
                        elif detecciones:
                            msg = f"Veo {len(detecciones)} objetos"
                        else:
                            msg = f"Veo {len(textos)} textos"
                        
                        print(f"  🔊 '{msg}'")
                        self.sintetizador.decir(msg)
                    
                except Exception as e:
                    print(f"  ❌ ERROR en modo combinado: {e}")
                    import traceback
                    traceback.print_exc()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def test_rendimiento(self):
        """Test de rendimiento del sistema"""
        print("\n" + "="*50)
        print("⚡ DIAGNÓSTICO: RENDIMIENTO")
        print("="*50)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara")
            return
        
        # Capturar 10 frames para el test
        frames = []
        print("📸 Capturando frames para test...")
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            time.sleep(0.1)
        
        cap.release()
        
        if not frames:
            print("❌ No se pudieron capturar frames")
            return
        
        print(f"✅ {len(frames)} frames capturados\n")
        
        # Test 1: Solo objetos
        print("🔍 TEST 1: Solo detección de objetos")
        tiempos_objetos = []
        for i, frame in enumerate(frames):
            inicio = time.time()
            detecciones = self.detector.detectar(frame, solo_prioritarios=True, debug=False)
            tiempo = time.time() - inicio
            tiempos_objetos.append(tiempo)
            print(f"  Frame {i+1}: {len(detecciones)} objetos en {tiempo:.3f}s")
        
        promedio_objetos = sum(tiempos_objetos) / len(tiempos_objetos)
        print(f"  📊 Promedio: {promedio_objetos:.3f}s por frame")
        
        # Test 2: Solo OCR
        print(f"\n📖 TEST 2: Solo OCR")
        tiempos_ocr = []
        for i, frame in enumerate(frames):
            inicio = time.time()
            textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True, debug=False)
            tiempo = time.time() - inicio
            tiempos_ocr.append(tiempo)
            print(f"  Frame {i+1}: {len(textos)} textos en {tiempo:.3f}s")
        
        promedio_ocr = sum(tiempos_ocr) / len(tiempos_ocr)
        print(f"  📊 Promedio: {promedio_ocr:.3f}s por frame")
        
        # Test 3: Combinado
        print(f"\n🔄 TEST 3: Modo combinado")
        tiempos_combinado = []
        for i, frame in enumerate(frames):
            inicio = time.time()
            detecciones = self.detector.detectar(frame, solo_prioritarios=True, debug=False)
            if len(detecciones) < 5:
                textos = self.lector_ocr.detectar_texto(frame, mejorar_imagen=True, debug=False)
            else:
                textos = []
            tiempo = time.time() - inicio
            tiempos_combinado.append(tiempo)
            print(f"  Frame {i+1}: {len(detecciones)} obj + {len(textos)} txt en {tiempo:.3f}s")
        
        promedio_combinado = sum(tiempos_combinado) / len(tiempos_combinado)
        print(f"  📊 Promedio: {promedio_combinado:.3f}s por frame")
        
        # Resumen
        print(f"\n📈 RESUMEN DE RENDIMIENTO:")
        print(f"  Solo objetos: {promedio_objetos:.3f}s ({1/promedio_objetos:.1f} FPS)")
        print(f"  Solo OCR: {promedio_ocr:.3f}s ({1/promedio_ocr:.1f} FPS)")  
        print(f"  Combinado: {promedio_combinado:.3f}s ({1/promedio_combinado:.1f} FPS)")
        
        # Recomendaciones
        print(f"\n💡 RECOMENDACIONES:")
        if promedio_objetos > 0.5:
            print("  - Detección de objetos lenta. Considerar modelo más ligero")
        if promedio_ocr > 2.0:
            print("  - OCR muy lento. Reducir resolución o desactivar mejoras")
        if promedio_combinado > 3.0:
            print("  - Modo combinado demasiado lento para tiempo real")


def main():
    """Función principal de diagnóstico"""
    print("🔧 SISTEMA DE DIAGNÓSTICO - GAFAS IA")
    print("="*50)
    
    diagnostico = DiagnosticoGafasIA()
    
    while True:
        print("\nSelecciona diagnóstico:")
        print("1. 🔍 Problemas de detección de objetos")
        print("2. 📖 Problemas de OCR/texto")
        print("3. 🔄 Problemas modo combinado") 
        print("4. ⚡ Test de rendimiento")
        print("5. 👋 Salir")
        
        opcion = input("\nOpción (1-5): ").strip()
        
        if opcion == '1':
            diagnostico.diagnosticar_deteccion_objetos()
        elif opcion == '2':
            diagnostico.diagnosticar_ocr()
        elif opcion == '3':
            diagnostico.test_modo_combinado()
        elif opcion == '4':
            diagnostico.test_rendimiento()
        elif opcion == '5':
            print("👋 ¡Diagnóstico completado!")
            break
        else:
            print("❌ Opción no válida")


if __name__ == "__main__":
    main()