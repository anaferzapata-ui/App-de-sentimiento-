from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Iterable


DEFAULT_MODEL_ID = "Qwen/Qwen2-0.5B-Instruct"


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: str
    explanation: str
    raw_response: str


def decode_uploaded_file(file_name: str, data: bytes) -> str:
    """Decode common SEC filing formats into plain text."""
    suffix = Path(file_name).suffix.lower()
    if suffix == ".pdf":
        return _decode_pdf(data)

    text = _decode_bytes(data)
    if suffix in {".html", ".htm", ".xhtml"} or "<html" in text[:1000].lower():
        return html_to_text(text)
    return normalize_whitespace(text)


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def _decode_pdf(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "Para leer PDFs instala pypdf: pip install pypdf. "
            "Los archivos .txt y .html de la SEC funcionan sin pypdf."
        ) from exc

    import io

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    return normalize_whitespace("\n".join(pages))


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</p\s*>", "\n", html)
    html = re.sub(r"(?i)</div\s*>", "\n", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    return normalize_whitespace(unescape(html))


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


SEC_ITEM_PATTERN = re.compile(
    r"(?im)(^|\n)\s*(item\s+"
    r"(?:1a|1b|1c|1|2|3|4|5|6|7a|7|8|9a|9b|9c|9|10|11|12|13|14|15|16)"
    r"[\.\s:-]+[^\n]{0,120})"
)


def find_sec_sections(text: str) -> dict[str, str]:
    """Return a map of SEC item headings to their section text."""
    matches = list(SEC_ITEM_PATTERN.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        heading = clean_heading(match.group(2))
        start = match.start(2)
        end = matches[index + 1].start(2) if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        # SEC tables of contents often repeat item headings. Keep the longer body.
        if heading not in sections or len(body) > len(sections[heading]):
            sections[heading] = body
    return sections


def clean_heading(heading: str) -> str:
    heading = normalize_whitespace(heading)
    heading = re.sub(r"\s+", " ", heading)
    return heading[:140].strip(" .:-")


def build_prompt(text: str, source_name: str, scope_name: str) -> str:
    return (
        "You are a financial sentiment analyst. Analyze the sentiment of the "
        "following SEC filing text. Return exactly four short lines in Spanish:\n"
        "Sentimiento: Positivo, Negativo o Neutral\n"
        "Confianza: Baja, Media o Alta\n"
        "Razon: one sentence\n"
        "Evidencia: one short phrase from the text\n\n"
        f"Archivo: {source_name}\n"
        f"Alcance: {scope_name}\n"
        "Texto SEC:\n"
        f"{text}"
    )


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3) :]
    return f"{head}\n\n[... texto recortado para el contexto del modelo ...]\n\n{tail}"


def load_generator(model_id: str = DEFAULT_MODEL_ID, local_files_only: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=120,
        do_sample=False,
        temperature=None,
    )


def analyze_sentiment(
    generator,
    text: str,
    source_name: str,
    scope_name: str,
    max_chars: int = 6000,
) -> SentimentResult:
    prompt = build_prompt(truncate_text(text, max_chars), source_name, scope_name)
    output = generator(prompt)[0]["generated_text"]
    response = output[len(prompt) :].strip() if output.startswith(prompt) else output.strip()
    return parse_response(response)


def parse_response(response: str) -> SentimentResult:
    label = _field(response, "Sentimiento") or _infer_label(response)
    confidence = _field(response, "Confianza") or "No especificada"
    explanation = _field(response, "Razon") or _field(response, "Razón") or response[:350]
    return SentimentResult(
        label=label,
        confidence=confidence,
        explanation=explanation,
        raw_response=response,
    )


def _field(text: str, field: str) -> str:
    pattern = re.compile(rf"(?im)^\s*{re.escape(field)}\s*:\s*(.+?)\s*$")
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _infer_label(text: str) -> str:
    lowered = text.lower()
    labels = ("positivo", "negativo", "neutral")
    hits = [label.title() for label in labels if label in lowered]
    return hits[0] if hits else "Neutral"


def preview_lines(text: str, limit: int = 8) -> Iterable[str]:
    for line in text.splitlines()[:limit]:
        if line.strip():
            yield line.strip()
