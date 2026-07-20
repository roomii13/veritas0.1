# Veritas

App movil con Expo + backend Python local para detectar senales de IA y fraude en imagenes, audios, videos y enlaces.

La app esta orientada a evitar fraudes, estafas, deepfakes, clonacion de voz y enlaces sospechosos. No usa base de datos y no guarda historial: cada analisis se procesa y se borra del backend al terminar.

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

## Configuracion util

- `DEEPFAKE_MODEL_NAME`: modelo de imagen. Default `prithivMLmods/deepfake-detector-model-v1`.
- `AUDIO_DEEPFAKE_MODEL_NAME`: modelo de audio. Default `Gustking/wav2vec2-large-xlsr-deepfake-audio-classification`.
- `VERITAS_DEVICE`: `auto`, `cpu`, `cuda` o `mps`.
- `VERITAS_USE_FP16`: `auto`, `true` o `false`; solo se usa en CUDA.
- `IMAGE_UNTRUSTED_THRESHOLD`: default `30`. Desde este valor una imagen se marca como no confiable.
- `AUDIO_MAX_DURATION_SEC`: default `60`, para no cargar audios gigantes en memoria.
- `VIDEO_MAX_FRAMES`: default `12`.
- `VIDEO_SAMPLE_METHOD`: `uniform` o `random`.
- `VIDEO_FAKE_RATIO_THRESHOLD`: default `0.30`.
- `LINK_TRUSTED_DOMAINS`: dominios confiables separados por coma.
- `LINK_EXPAND_SHORTENERS`: default `false`; si es `true`, intenta expandir acortadores con una llamada HTTP.

Los endpoints FastAPI ejecutan los detectores pesados fuera del event loop usando `asyncio.to_thread`.

## Limitaciones a tener en cuenta (importante para la demo)

1. **Son modelos de la comunidad, no de empresas grandes.** Entrenados en datasets acotados, perfectos para mostrar el concepto funcionando, pero su precision en el mundo real puede ser menor que la de un producto comercial como Resemble, Hive o Reality Defender.
2. **El detector de video es un enfoque simplificado**: analiza frames sueltos con el modelo de imagen, no captura inconsistencias temporales como parpadeo o flujo de movimiento como lo haria un modelo especifico tipo CNN+LSTM entrenado para video.
3. **"Llamadas en vivo" no esta cubierto de forma nativa.** Para simular eso en el demo, se puede grabar un segmento de la llamada y pasarlo por `audio_detector.py` / `video_detector.py`. Un sistema real de deteccion en tiempo real durante una llamada requiere streaming e infraestructura adicional fuera del alcance de este demo.
4. Los modelos de audio e imagen esperan formatos comunes: `wav`, `mp3`, `m4a`, `jpg`, `png`, `webp`. Videos deben poder abrirse con OpenCV: `mp4`, `mov`, `avi`.

## Nota de seguridad

Veritas es una demo tecnica. Para casos reales de fraude, no debe ser el unico criterio de decision. Si hay dinero, identidad, documentos, claves o presion por urgencia, valida siempre por un canal independiente.
