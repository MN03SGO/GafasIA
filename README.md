
#  RasVision

Este proyecto nace con la idea de un compañero con el propósito de desarrollar una solución accesible y de ayuda a largo plazo a personas con discapacidad y es donde retomo esa idea para la ayuda de mi abuela y desarrollo de proyecto de expotecnia del Insituto Nacional Tecnico Industrial, ya que la idea es que sea funcional y de bajo costo que pueda asistirle a ella y otras personas con discapacidad visual en la realización de sus actividades cotidianas. La idea central es aprovechar herramientas tecnológicas disponibles al entorno del ciudadano salvadoreño permitiendo a los usuarios interactuar con su entorno de forma más segura, autónoma y eficiente.

El enfoque es casero y adaptable, pensado para que cualquier persona con conocimientos básicos de programación y electrónica pueda replicarlo, modificarlo o mejorarlo. Esto lo convierte en una alternativa viable frente a soluciones comerciales que suelen ser costosas o inaccesibles para muchas familias.

Además de su funcionalidad técnica, el proyecto busca fomentar la inclusión digital, la empatía social y el diseño centrado en el usuario, poniendo en primer plano las necesidades reales de quienes enfrentan barreras visuales en su día a día.

## [Estructura](https://github.com/MN03SGO)
```javascript
[Oct 23 21:15]  .
├── [Oct 10 22:00]  camaras.py
├── [Oct 20 20:43]  docs
│   └── [Oct 20 01:24]  TRABAJAR_LUNES.TXT
├── [Oct 26 22:18]  ejemplo_ocr_integrado.py
├── [Oct 17 02:19]  libs
│   ├── [Oct 16 22:20]  requirements_laptop.txt
│   └── [Oct  7 19:30]  requirements.txt
├── [Sep 29 22:25]  LICENSE
├── [Oct 22 23:25]  models
│   ├── [Oct 26 21:46]  detecciones
│   │   ├── [Oct 26 14:47]  Modelo_v3.pt
│   │   ├── [Oct 26 21:46]  Modelo_V4.pt
│   │   └── [Oct 22 21:46]  rasvision_final_v2.pt
│   └── [Oct  3 18:45]  tts
│       ├── [Oct  2 23:04]  es_ES-davefx-medium.onnx
│       └── [Oct  2 23:04]  es_ES-davefx-medium.onnx.json
├── [Oct 23 21:15]  rasvision_final_v2.pt
├── [Oct 23 21:15]  README.md
├── [Oct 22 21:59]  src
│   ├── [Oct 26 21:50]  audio
│   │   ├── [Oct 26 21:50]  __pycache__
│   │   └── [Oct 17 02:16]  sintetizador_voz.py
│   ├── [Oct 26 21:50]  deteccion
│   │   ├── [Oct 26 15:01]  analizador_escena.py
│   │   └── [Oct 26 21:50]  __pycache__
│   └── [Oct 26 21:50]  ocr
│       ├── [Oct 20 01:24]  lector_texto.py
│       └── [Oct 26 21:50]  __pycache__
├── [Oct  3 23:41]  temp_audio.wav
├── [Oct 20 00:10]  yolov8n.pt
└── [Oct 20 00:10]  yolov8n-seg.pt

13 directories, 19 files


```
![15a4b14c-da9a-4dea-b511-cd47723c10b9](https://github.com/user-attachments/assets/149c5e11-3454-4af0-8de7-ebbb1bd4c242)

## [Herramientas](https://github.com/MN03SGO)
 - [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
 - [Modulo de pi cam](https://www.amazon.com/Raspberry-Pi-Camera-Module-Megapixel/dp/B01ER2SKFS)
## [Librerias principales usadas](https://github.com/MN03SGO)
Propenso a cambiar

 - [OpenCV]()
 - [Pytesseract]()
 - [Tesseract]()
 - [pyttsx3 (TTS)]()
 - [pyttsx3 (TTS)]()
 - [Torch (torchvision)]()
 - [YOLOv8n]()


## [Agradacimientos](https://github.com/MN03SGO)
RasVision no habría sido posible sin las siguientes servicios de infraestructura:
 - [Google Colab](https://colab.google/) 
 - [Roboflow (y Roboflow Universe)](https://app.roboflow.com/proyecto-rkmc5)
   
## [Software y Librerías de Código Abierto](https://github.com/MN03SGO)
RasVision no habría sido posible sin las siguientes plataformas:
 - [Ultralytics (YOLOv8)](https://github.com/ultralytics/ultralytics) 
 - [PyTorch](https://pytorch.org/)
 - [Debian (y la comunidad GNU/Linux):](https://www.debian.org/)
 - [Python](https://www.python.org/downloads/release/python-3100/)
 - [OpenCV](https://opencv.org/)
 - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
 - [pyttsx3](https://pypi.org/project/pyttsx3/)
 - [Numpy](https://numpy.org/)
   
## [Datos y Modelos Pre-entrenados](https://github.com/MN03SGO)
RasVision no habría sido posible sin los siguientes recursos:
 - [Consorcio COCO:](https://cocodataset.org/) Por el dataset Common Objects in Context (COCO), sobre el cual se pre-entrenó el modelo base yolov8n.pt. Este conocimiento previo fue la base sobre la que construi toda la personalización.
 - [La Comunidad de Creadores de Roboflow Universe:](https://roboflow.com/universe) Un agradecimiento especial a todos los individuos y equipos anónimos que dedicaron tiempo a crear, anotar y compartir públicamente los datasets de alta calidad (paredes, escaleras, dinero, frutas, papel higiénico, etc.) que sirvieron como "ingredientes" para el super-dataset final.

Instituto Nacional Tecnico Industrial 2025

## [Autor]

- [@MN03SGO](https://github.com/MN03SGO)

 
