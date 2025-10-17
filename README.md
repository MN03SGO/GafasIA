
#  RasVision

Este proyecto nace con la idea de un compañero con el propósito de desarrollar una solución accesible y de ayuda a largo plazo a personas con discapacidad y es donde retomo esa idea para la ayuda de mi abuela y desarrollo de proyecto de expotecnia del Insituto Nacional Tecnico Industrial, ya que la idea es que sea funcional y de bajo costo que pueda asistirle a ella y otras personas con discapacidad visual en la realización de sus actividades cotidianas. La idea central es aprovechar herramientas tecnológicas disponibles al entorno del ciudadano salvadoreño permitiendo a los usuarios interactuar con su entorno de forma más segura, autónoma y eficiente.

El enfoque es casero y adaptable, pensado para que cualquier persona con conocimientos básicos de programación y electrónica pueda replicarlo, modificarlo o mejorarlo. Esto lo convierte en una alternativa viable frente a soluciones comerciales que suelen ser costosas o inaccesibles para muchas familias.

Además de su funcionalidad técnica, el proyecto busca fomentar la inclusión digital, la empatía social y el diseño centrado en el usuario, poniendo en primer plano las necesidades reales de quienes enfrentan barreras visuales en su día a día.

## Estructura
Propenso a cambios 17/10/2025

```javascript
[Oct 17 02:19]  .
├── [Oct 10 22:00]  camaras.py
├── [Oct 17 02:14]  ejemplo_ocr_integrado.py
├── [Oct  4 01:11]  gafas_ia_integrado.py
├── [Oct 17 02:19]  libs
│   ├── [Oct 16 22:20]  requirements_laptop.txt
│   └── [Oct  7 19:30]  requirements.txt
├── [Sep 29 22:25]  LICENSE
├── [Oct 16 23:52]  models
│   ├── [Oct 16 23:55]  deteccion
│   │   └── [Sep 29 23:13]  yolov8n.pt
│   └── [Oct  3 18:45]  tts
│       ├── [Oct  2 23:04]  es_ES-davefx-medium.onnx
│       └── [Oct  2 23:04]  es_ES-davefx-medium.onnx.json
├── [Oct 15 09:54]  README.md
├── [Oct 16 00:35]  src
│   ├── [Oct 17 02:08]  audio
│   │   ├── [Oct 17 02:08]  __pycache__
│   │   └── [Oct 17 02:16]  sintetizador_voz.py
│   ├── [Oct 17 02:08]  deteccion
│   │   ├── [Oct 17 01:41]  detector_objetos.py
│   │   └── [Oct 17 02:08]  __pycache__
│   ├── [Oct 16 00:57]  logica
│   └── [Oct 17 02:08]  ocr
│       ├── [Oct 17 02:10]  lector_texto.py
│       └── [Oct 17 02:08]  __pycache__
└── [Oct  3 23:41]  temp_audio.wav

13 directories, 14 files


```
![15a4b14c-da9a-4dea-b511-cd47723c10b9](https://github.com/user-attachments/assets/149c5e11-3454-4af0-8de7-ebbb1bd4c242)

## Herramientas
 - [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
 - [Modulo de pi cam](https://www.amazon.com/Raspberry-Pi-Camera-Module-Megapixel/dp/B01ER2SKFS)
## Librerias principales usadas 
Propenso a cambiar

 - [OpenCV]()
 - [Pytesseract]()
 - [Tesseract]()
- [pyttsx3 (TTS)]()
 - [pyttsx3 (TTS)]()
 - [Torch (torchvision)]()
- [YOLOv8n]()

## Documentation
En proceso