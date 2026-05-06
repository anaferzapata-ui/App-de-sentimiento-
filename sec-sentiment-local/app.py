from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from sec_sentiment import (
    DEFAULT_MODEL_ID,
    analyze_sentiment,
    decode_uploaded_file,
    find_sec_sections,
    load_generator,
    preview_lines,
    truncate_text,
)


st.set_page_config(
    page_title="SEC Sentiment Local",
    page_icon="📄",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def cached_generator(model_id: str, local_files_only: bool):
    return load_generator(model_id=model_id, local_files_only=local_files_only)


def read_sample_file() -> tuple[str, bytes]:
    sample = Path("sample_data/sec_sample_10k.txt")
    return sample.name, sample.read_bytes()


st.title("SEC Sentiment Local")
st.caption("Analiza sentimiento de reportes SEC con un modelo Instruct ejecutado en tu computadora.")

with st.sidebar:
    st.header("Modelo")
    model_id = st.text_input(
        "Modelo o ruta local",
        value=os.getenv("SEC_MODEL_PATH", DEFAULT_MODEL_ID),
        help="Puedes usar Qwen/Qwen2-0.5B-Instruct desde cache local o una carpeta descargada.",
    )
    local_only = st.checkbox(
        "Usar solo archivos locales",
        value=True,
        help="Actívalo para evitar descargas. Desactívalo solo si quieres descargar el modelo una vez.",
    )
    max_chars = st.slider(
        "Caracteres enviados al LLM",
        min_value=1500,
        max_value=12000,
        value=6000,
        step=500,
    )

uploaded = st.file_uploader(
    "Sube un archivo de la SEC",
    type=["txt", "html", "htm", "md", "pdf"],
)

sample_mode = st.toggle("Usar archivo de ejemplo", value=uploaded is None)

file_name = ""
raw_data: bytes | None = None
if uploaded is not None:
    file_name = uploaded.name
    raw_data = uploaded.getvalue()
elif sample_mode:
    file_name, raw_data = read_sample_file()

if raw_data is None:
    st.info("Sube un archivo .txt, .html o .pdf de la SEC para empezar.")
    st.stop()

try:
    filing_text = decode_uploaded_file(file_name, raw_data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

sections = find_sec_sections(filing_text)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Archivo")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Nombre", file_name)
    metric_b.metric("Caracteres", f"{len(filing_text):,}")
    metric_c.metric("Secciones", len(sections))

    scope = st.radio(
        "Alcance del análisis",
        ["Documento completo", "Sección detectada", "Texto personalizado"],
        horizontal=True,
    )

    selected_scope = "Documento completo"
    selected_text = filing_text

    if scope == "Sección detectada":
        if not sections:
            st.warning("No encontré encabezados tipo Item 1A, Item 7, Item 8. Usaré el documento completo.")
        else:
            section_name = st.selectbox("Sección SEC", list(sections.keys()))
            selected_scope = section_name
            selected_text = sections[section_name]
    elif scope == "Texto personalizado":
        selected_scope = "Texto personalizado"
        selected_text = st.text_area(
            "Pega o edita el texto a analizar",
            value=truncate_text(filing_text, 2500),
            height=280,
        )

    st.subheader("Vista previa")
    preview = "\n".join(preview_lines(selected_text, limit=12))
    st.text_area("Texto seleccionado", value=preview, height=220, disabled=True)

with right:
    st.subheader("Resultado")
    run = st.button("Analizar sentimiento", type="primary", use_container_width=True)

    if run:
        if not selected_text.strip():
            st.error("No hay texto para analizar.")
            st.stop()

        with st.spinner("Cargando modelo local y generando análisis..."):
            try:
                generator = cached_generator(model_id, local_only)
                result = analyze_sentiment(
                    generator=generator,
                    text=selected_text,
                    source_name=file_name,
                    scope_name=selected_scope,
                    max_chars=max_chars,
                )
            except Exception as exc:
                st.error(
                    "No pude cargar o ejecutar el modelo. Revisa que el modelo esté "
                    "descargado localmente o desactiva 'Usar solo archivos locales' "
                    "para descargarlo una vez."
                )
                st.exception(exc)
                st.stop()

        st.metric("Sentimiento", result.label)
        st.metric("Confianza", result.confidence)
        st.write(result.explanation)

        with st.expander("Respuesta completa del LLM"):
            st.code(result.raw_response)
    else:
        st.info("Selecciona el alcance y presiona el botón para correr el LLM local.")

st.divider()
st.caption(
    "La app está diseñada para fines educativos. El sentimiento puede ser incorrecto, "
    "especialmente con modelos pequeños y textos financieros largos."
)
