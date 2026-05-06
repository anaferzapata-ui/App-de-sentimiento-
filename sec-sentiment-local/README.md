# SEC Sentiment Local

App local en Streamlit para subir un archivo de la SEC y pedirle a un modelo Instruct, como `Qwen/Qwen2-0.5B-Instruct`, una clasificación simple de sentimiento. La app permite analizar el documento completo, una sección SEC detectada (`Item 1A`, `Item 7`, etc.) o un texto personalizado.

![Captura de la app](assets/app_screenshot.png)

## Funciones

- Subida de archivos `.txt`, `.html`, `.htm`, `.md` y `.pdf`.
- Limpieza básica de HTML y extracción opcional de texto PDF.
- Detección de secciones SEC por encabezados tipo `Item 1A`, `Item 7`, `Item 8`.
- Ejecución local con Hugging Face Transformers.
- Selector para usar solo modelos ya descargados localmente.

## Instalación

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para descargar el modelo una vez:

```bash
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; m='Qwen/Qwen2-0.5B-Instruct'; AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)"
```

Después de eso, la app puede correr en modo local usando el cache de Hugging Face.

## Uso

```bash
streamlit run app.py
```

Abre la URL local que aparezca en terminal, normalmente `http://localhost:8501`.

Dentro de la app:

1. Sube un archivo SEC o usa el ejemplo incluido.
2. Elige `Documento completo`, `Sección detectada` o `Texto personalizado`.
3. Presiona `Analizar sentimiento`.

## Modelo

El valor por defecto es:

```text
Qwen/Qwen2-0.5B-Instruct
```

También puedes usar una carpeta local:

```bash
SEC_MODEL_PATH=./models/qwen2-0.5b-instruct streamlit run app.py
```

## Nota

Este proyecto es educativo. Un modelo pequeño puede equivocarse, especialmente con archivos SEC largos o lenguaje financiero técnico.

## GitHub Pages

La app real usa Python y un LLM local, así que GitHub Pages no puede ejecutarla directamente. Para Pages se puede publicar este README o una página estática de demostración, pero el análisis debe correr en tu computadora.
