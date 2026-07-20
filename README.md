# Veritas

App movil con Expo + backend Python local para detectar señales de IA y fraude en imagenes, audios, videos y enlaces.

La app esta orientada a evitar fraudes, estafas, deepfakes, clonacion de voz y enlaces sospechosos. No usa base de datos y no guarda historial: cada analisis se procesa y se borra del backend al terminar.

Por el momento trabaja con repositorios de agentes entrenados, ya que la primera version de veritas utiliza api de hugginface que pide pagos, utilice diferentes cuentas de google de familiares y amigos para poder crear èsta y la primer version de veritas, ya que no lleguè a los tokens gratuitos y desde mi cuenta de google no puedo usar codex porque pide un codigo que envia  a un numero de telefono que ya no tengo. Este proyecto fue creado de 0 con codex y chatgpt utlizando diferentes prompts para las mejoras graduales, la idea es que Veritas sea un detector de IA que se pueda utilizar en diferentes contextos desde telefonos de usarios individuales hazta entidades bancarias incluyendo llamadas y videollamdas a futuro.

## Arquitectura

- `App.tsx`: app Expo llamada Veritas.
- `backend/main.py`: API FastAPI local.
- `image_detector.py`: detector Hugging Face para imagenes.
- `audio_detector.py`: detector Hugging Face para audio.
- `video_detector.py`: detector simplificado por frames.
- `link_detector.py`: heuristica local para enlaces no multimedia.
- `pipeline.py`: CLI para correr los detectores por terminal.

## Requisitos

- Node.js compatible con Expo SDK 57.
- Python 3.10+ recomendado.
- Internet en la primera ejecucion del backend si los modelos Hugging Face no estan cacheados.

## Instalacion

```bash
pip install -r requirements.txt
npm install
```

## Correr backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Correr app Expo

En otra terminal:

```bash
npm start
```

Si usas Expo Go en un celular fisico, cambia la URL del backend por la IP LAN de tu PC:

```powershell
$env:EXPO_PUBLIC_API_URL="http://TU_IP_LOCAL:8000"
npm start
```

En Android emulator suele servir:

```powershell
$env:EXPO_PUBLIC_API_URL="http://10.0.2.2:8000"
npm start
```

## Uso desde la app

1. Elegi `Archivo` para subir imagen, audio o video desde el movil.
2. Tambien podes tomar foto/video con camara o elegir desde galeria.
3. Elegi `Enlace` para analizar una URL.
4. Si la URL apunta directo a una imagen/audio/video, el backend la descarga y analiza.
5. Si la URL es una pagina normal, Veritas aplica heuristicas antifraude del enlace.
6. La respuesta muestra porcentaje estimado de IA, riesgo y recomendacion. Si el riesgo es alto, muestra `NO CONFIAR`.

## Uso por terminal

```bash
python pipeline.py --image foto.jpg --audio llamada.wav --video reunion.mp4
```

Tambien podes pedir JSON:

```bash
python pipeline.py --image foto.jpg --json
```

## Fallback de demo

Por defecto, el backend tiene `VERITAS_ALLOW_HEURISTIC_FALLBACK=true`.

Esto evita que la demo quede inutilizable si Torch/Hugging Face falla o si el modelo aun no se pudo descargar. En ese caso devuelve una estimacion heuristica local marcada como `local_heuristic_fallback`. Para una demo estricta solo con modelos, usa:

```powershell
$env:VERITAS_ALLOW_HEURISTIC_FALLBACK="false"
```

## Limitaciones a tener en cuenta (importante)

1. **Son modelos de la comunidad, no de empresas grandes.** Entrenados en datasets acotados, perfectos para mostrar el concepto funcionando, pero su precision en el mundo real puede ser menor que la de un producto comercial como Resemble, Hive o Reality Defender.
2. **El detector de video es un enfoque simplificado**: analiza frames sueltos con el modelo de imagen, no captura inconsistencias temporales como parpadeo o flujo de movimiento como lo haria un modelo especifico tipo CNN+LSTM entrenado para video.
3. **"Llamadas en vivo" no esta cubierto de forma nativa.** Para simular eso en el demo, se puede grabar un segmento de la llamada y pasarlo por `audio_detector.py` / `video_detector.py`. Un sistema real de deteccion en tiempo real durante una llamada requiere streaming e infraestructura adicional fuera del alcance de este demo.
4. Los modelos de audio e imagen esperan formatos comunes: `wav`, `mp3`, `m4a`, `jpg`, `png`, `webp`. Videos deben poder abrirse con OpenCV: `mp4`, `mov`, `avi`.

## Nota de seguridad

Veritas es una demo tecnica. Para casos reales de fraude, no debe ser el unico criterio de decision. Si hay dinero, identidad, documentos, claves o presion por urgencia, valida siempre por un canal independiente.
