
#  RasVision

La discapacidad visual limita la autonomía y participación social de millones de personas, y en El Salvador el acceso a tecnologías asistivas es reducido por su alto costo y escasa disponibilidad. Para responder a esta necesidad, este anteproyecto propone el desarrollo de un asistente visual portátil de bajo costo, diseñado con herramientas accesibles en el entorno salvadoreño.

COMPONENTES USADOS (POSIBLES CAMBIOS)

OS

Debian GNU/Linux 13 (Trixie) en PC, laptop y Raspberry Pi 4

Hardware

PC de escritorio: H510M-HDV/M.2 SE, Nvidia RTX 3050 (6 GB), i5 10th, 16 GB RAM, 500 GB M.2, HDD 2 TB + 500 GB

Laptop: Lenovo ThinkBook 14s Yoga ITL, i5 11th, 16 GB RAM, 256 GB M.2, gráficos integrados

Microcontrolador: Raspberry Pi 4 Model B Rev 1.5 (4 GB RAM, BCM2711, 28 GB SD)

Cámara: Arducam 5MP OV5647, 1080P HD

Otros: Fuente CA, impresora 3D Artillery 4x Plus

Software

Contenedores: Docker

Shell / scripting: Bash

Editores: VS Code (+ plugins), Kate

Control de versiones: Git + GitHub

Lenguajes: Python, YAML



[Sep 21 00:53]  .
├── [Sep 19 07:31]  ejemplo_detector.py
├── [Sep 19 07:31]  gafas_ia_integrado.py
├── [Sep 21 00:53]  README.md
├── [Sep 17 07:34]  src
│   ├── [Sep 16 22:35]  audio
│   │   ├── [Sep 16 22:35]  __pycache__
│   │   └── [Sep 19 07:31]  sintetizador_voz.py
│   ├── [Sep 16 02:10]  deteccion
│   │   ├── [Sep 19 07:31]  detector_objetos.py
│   │   └── [Sep 16 02:21]  __pycache__
│   └── [Sep 21 00:53]  ocr
│       └── [Sep 21 22:33]  lector_texto.py
└── [Sep 16 02:07]  yolov8n.pt

7 directories, 7 files

'''

INSTALLATION

CLONAR EL REPOSITORIO

https://github.com/USUARIO/MN03SGO.git


Estructura del proyecto:

├── src
│   ├── audio
│   │   └── sintetizador_voz.py
│   ├── deteccion
│   │   └── detector_objetos.py
│   └── ocr
│       └── lector_texto.py
├── ejemplo_detector.py
├── gafas_ia_integrado.py
├── yolov8n.pt
├── README.md
└── .gitignore


Instalación de dependencias

sudo apt update
sudo apt install python3-pip git -y
pip3 install -r requirements.txt

Dar permisos y ejecutar scripts

chmod +x src/audio/*.py
chmod +x src/deteccion/*.py
chmod +x src/ocr/*.py

Uso del proyecto

Ejemplo de ejecución del detector de objetos

python3 ejemplo_detector.py

Ejemplo de ejecución del sistema integrado:

python3 gafas_ia_integrado.py

Salida esperada

[INFO] Detector iniciado
[INFO] Objeto detectado: silla
[INFO] Texto leído: "Bienvenido"

Tecnologías y herramientas

OS: Debian GNU/Linux 13 (Trixie)

Hardware: Raspberry Pi 4, PC de escritorio, cámara Arducam 5MP

Lenguajes: Python, YAML, Bash

Software: Docker, VS Code, Git/GitHub

Contribuir

Hacer un fork del proyecto

Crear una rama (git checkout -b feature/nueva-funcion)

Hacer commit de los cambios

Abrir un Pull Request

Licencia

MIT License

Autores

Inspirado en la necesidad de apoyar a mi abuela

Desarrollado por ANTONI SIGARAN 
                 NAYELI MORALES
