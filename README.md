
# Gafas de asistencia visual 

Este proyecto nace con la idea de un compañero con el propósito de desarrollar una solución accesible y de ayuda a largo plazo a personas con discapacidad y es donde retomo esa idea para la ayuda de mi abuela, ya que la idea es que sea funcional y de bajo costo que pueda asistirle a ella y otras personas con discapacidad visual en la realización de sus actividades cotidianas. La idea central es aprovechar herramientas tecnológicas disponibles al entorno del ciudadano salvadoreño permitiendo a los usuarios interactuar con su entorno de forma más segura, autónoma y eficiente.

El enfoque es casero y adaptable, pensado para que cualquier persona con conocimientos básicos de programación y electrónica pueda replicarlo, modificarlo o mejorarlo. Esto lo convierte en una alternativa viable frente a soluciones comerciales que suelen ser costosas o inaccesibles para muchas familias.

Además de su funcionalidad técnica, el proyecto busca fomentar la inclusión digital, la empatía social y el diseño centrado en el usuario, poniendo en primer plano las necesidades reales de quienes enfrentan barreras visuales en su día a día.


## Estructura del proyecto 
18/08/2025  Propenso a modificaciones

```javascript
[Aug 19 12:23]  .
├── [Aug 17 15:24]  config
│   └── [Aug 17 15:24]  ajuste.yaml
├── [Aug 19 12:23]  README.md
├── [Aug 17 15:24]  requirements.txt
├── [Aug 18 18:17]  src
│   ├── [Aug 19 12:29]  audio
│   │   ├── [Aug 19 12:29]  __init__.py
│   │   └── [Aug 19 12:37]  texto_a_voz.py
│   ├── [Aug 19 12:46]  deteccion
│   │   ├── [Aug 18 00:31]  detector_objetos.py
│   │   ├── [Aug 19 16:14]  detector_personas.py
│   │   └── [Aug 18 16:47]  __init__.py
│   ├── [Aug 18 16:48]  __init__.py
│   ├── [Aug 17 15:24]  main.py
│   └── [Aug 17 15:24]  utilidades
│       ├── [Aug 17 23:57]  camara.py
│       ├── [Aug 17 15:24]  config.py
│       └── [Aug 17 15:24]  logger.py
└── [Aug 17 19:59]  venv
    ├── [Aug 17 23:10]  bin
    │   ├── [Aug 17 15:26]  activate
    │   ├── [Aug 17 15:26]  activate.csh
    │   ├── [Aug 17 15:26]  activate.fish
    │   ├── [Aug 17 15:26]  Activate.ps1
    │   ├── [Aug 17 23:10]  black
    │   ├── [Aug 17 23:10]  blackd
    │   ├── [Aug 17 21:03]  ccmake
    │   ├── [Aug 17 21:03]  cmake
    │   ├── [Aug 17 23:10]  coverage
    │   ├── [Aug 17 23:10]  coverage3
    │   ├── [Aug 17 23:10]  coverage-3.11
    │   ├── [Aug 17 21:03]  cpack
    │   ├── [Aug 17 19:59]  cpuinfo
    │   ├── [Aug 17 21:03]  ctest
    │   ├── [Aug 17 20:37]  f2py
    │   ├── [Aug 17 20:37]  f2py3
    │   ├── [Aug 17 20:37]  f2py3.11
    │   ├── [Aug 17 18:10]  face_detection
    │   ├── [Aug 17 18:10]  face_recognition
    │   ├── [Aug 17 23:10]  flake8
    │   ├── [Aug 17 19:59]  fonttools
    │   ├── [Aug 17 17:59]  gtts-cli
    │   ├── [Aug 17 19:59]  isympy
    │   ├── [Aug 17 20:17]  jsonschema
    │   ├── [Aug 17 21:03]  lit
    │   ├── [Aug 17 18:03]  markdown-it
    │   ├── [Aug 17 19:59]  normalizer
    │   ├── [Aug 17 15:29]  pip
    │   ├── [Aug 17 15:29]  pip3
    │   ├── [Aug 17 15:29]  pip3.11
    │   ├── [Aug 17 22:16]  proton
    │   ├── [Aug 17 22:16]  proton-viewer
    │   ├── [Aug 17 20:17]  pyav
    │   ├── [Aug 17 23:10]  pycodestyle
    │   ├── [Aug 17 23:10]  pyflakes
    │   ├── [Aug 17 19:59]  pyftmerge
    │   ├── [Aug 17 19:59]  pyftsubset
    │   ├── [Aug 17 18:03]  pygmentize
    │   ├── [Aug 17 23:10]  py.test
    │   ├── [Aug 17 23:10]  pytest
    │   ├── [Aug 17 15:26]  python -> /home/sigaran/.pyenv/versions/3.11.9/bin/python
    │   ├── [Aug 17 15:26]  python3 -> python
    │   ├── [Aug 17 15:26]  python3.11 -> python
    │   ├── [Aug 17 22:29]  torchfrtrace
    │   ├── [Aug 17 22:29]  torchrun
    │   ├── [Aug 17 19:59]  tqdm
    │   ├── [Aug 17 19:59]  ttx
    │   ├── [Aug 17 20:00]  ultralytics
    │   ├── [Aug 17 21:03]  wheel
    │   └── [Aug 17 20:00]  yolo
    ├── [Aug 17 18:00]  include
    │   ├── [Aug 17 15:26]  python3.11
    │   └── [Aug 17 18:00]  site
    ├── [Aug 17 15:26]  lib
    │   └── [Aug 17 15:26]  python3.11
    ├── [Aug 17 15:26]  lib64 -> lib
    ├── [Aug 17 15:26]  pyvenv.cfg
    └── [Aug 17 19:59]  share
        └── [Aug 17 19:59]  man

16 directories, 64 files


```


## Herramientas
Propenso a cambiar

 - [Raspberry Pi 4](https://www.raspberrypi.com/products/raspberry-pi-4-model-b/)
 - [Modulo de pi cam](https://www.amazon.com/Raspberry-Pi-Camera-Module-Megapixel/dp/B01ER2SKFS)
 - [Posiblemente 3 pulsadores fisicos](https://articulo.mercadolibre.com.mx/MLM-680773293-100pz-push-button-boton-12x12x-4-pines-microswitch-negro-_JM?searchVariation=36450427677#polycard_client=search-nordic&searchVariation=36450427677&position=14&search_layout=grid&type=item&tracking_id=c9b90ca6-fdc7-4155-88a7-50bc910c6adb)
- [Impresion stl de lentes con compartimiemto de Pi cam Y Raspberry]()


## LIbrerias usadas 
Propenso a cambiar

 - [OpenCV]()
 - [Pytesseract]()
 - [Tesseract]()
- [pyttsx3 (TTS)]()
 - [pyttsx3 (TTS)]()
 - [Torch (torchvision)]()
- [YOLOv5]()


## Documentation
En proceso


[Documentation](https://linktodocumentation)

