from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "app_screenshot.png"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def round_rect(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def text(draw: ImageDraw.ImageDraw, xy, value: str, size: int, fill, bold: bool = False):
    draw.text(xy, value, font=font(size, bold=bold), fill=fill)


def main() -> None:
    width, height = 1440, 900
    img = Image.new("RGB", (width, height), "#f7f8fb")
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, width, 84), fill="#ffffff")
    draw.line((0, 84, width, 84), fill="#dde3eb", width=1)
    text(draw, (52, 24), "SEC Sentiment Local", 34, "#14213d", True)
    text(draw, (52, 62), "Analisis local de sentimiento con Qwen2 0.5B Instruct", 15, "#526173")

    round_rect(draw, (42, 118, 376, 820), 8, "#ffffff", "#d7dee8")
    text(draw, (70, 148), "Modelo", 22, "#14213d", True)
    text(draw, (70, 194), "Modelo o ruta local", 14, "#536173")
    round_rect(draw, (70, 218, 348, 264), 6, "#f7f9fc", "#cbd5e1")
    text(draw, (86, 232), "Qwen/Qwen2-0.5B-Instruct", 15, "#1f2937")
    round_rect(draw, (70, 292, 92, 314), 4, "#16a34a", "#15803d")
    text(draw, (104, 290), "Usar solo archivos locales", 15, "#1f2937")
    text(draw, (70, 350), "Caracteres enviados al LLM", 14, "#536173")
    draw.line((70, 390, 348, 390), fill="#94a3b8", width=5)
    draw.ellipse((205, 380, 225, 400), fill="#2563eb")
    text(draw, (70, 430), "Archivo cargado", 18, "#14213d", True)
    round_rect(draw, (70, 464, 348, 532), 6, "#eef6ff", "#93c5fd")
    text(draw, (94, 486), "sec_sample_10k.txt", 17, "#1d4ed8", True)

    round_rect(draw, (410, 118, 936, 820), 8, "#ffffff", "#d7dee8")
    text(draw, (438, 148), "Archivo", 24, "#14213d", True)
    for i, (label, value) in enumerate([("Nombre", "sec_sample_10k.txt"), ("Caracteres", "796"), ("Secciones", "4")]):
        x = 438 + i * 160
        round_rect(draw, (x, 188, x + 144, 258), 6, "#f8fafc", "#e2e8f0")
        text(draw, (x + 14, 202), label, 13, "#64748b")
        text(draw, (x + 14, 226), value, 17, "#0f172a", True)

    text(draw, (438, 292), "Alcance del analisis", 16, "#334155", True)
    pills = [("Documento completo", False), ("Seccion detectada", True), ("Texto personalizado", False)]
    x = 438
    for label, active in pills:
        w = 150 if label != "Texto personalizado" else 166
        round_rect(draw, (x, 324, x + w, 364), 20, "#2563eb" if active else "#f1f5f9", "#2563eb" if active else "#cbd5e1")
        text(draw, (x + 18, 335), label, 14, "#ffffff" if active else "#334155", True)
        x += w + 12

    text(draw, (438, 404), "Seccion SEC", 15, "#536173")
    round_rect(draw, (438, 430, 884, 476), 6, "#f8fafc", "#cbd5e1")
    text(draw, (454, 444), "Item 1A. Risk Factors", 16, "#1f2937")

    text(draw, (438, 526), "Vista previa", 22, "#14213d", True)
    round_rect(draw, (438, 566, 884, 770), 6, "#0f172a", "#0f172a")
    preview = [
        "Item 1A. Risk Factors",
        "Our business faces risks related to competition,",
        "cybersecurity incidents, and uncertain macroeconomic",
        "conditions. If customers reduce budgets, our revenue",
        "and profitability could decline materially.",
    ]
    y = 590
    for line in preview:
        text(draw, (462, y), line, 16, "#e2e8f0")
        y += 30

    round_rect(draw, (970, 118, 1398, 820), 8, "#ffffff", "#d7dee8")
    text(draw, (998, 148), "Resultado", 24, "#14213d", True)
    round_rect(draw, (998, 198, 1370, 252), 8, "#2563eb", "#1d4ed8")
    text(draw, (1112, 215), "Analizar sentimiento", 18, "#ffffff", True)

    round_rect(draw, (998, 302, 1176, 388), 6, "#fff7ed", "#fed7aa")
    text(draw, (1020, 322), "Sentimiento", 14, "#9a3412")
    text(draw, (1020, 350), "Negativo", 28, "#9a3412", True)
    round_rect(draw, (1192, 302, 1370, 388), 6, "#eff6ff", "#bfdbfe")
    text(draw, (1214, 322), "Confianza", 14, "#1d4ed8")
    text(draw, (1214, 350), "Media", 28, "#1d4ed8", True)

    round_rect(draw, (998, 430, 1370, 570), 6, "#f8fafc", "#e2e8f0")
    text(draw, (1022, 456), "Razon", 16, "#334155", True)
    text(draw, (1022, 490), "La seccion enfatiza competencia,", 17, "#334155")
    text(draw, (1022, 520), "ciberseguridad y presion en ingresos.", 17, "#334155")

    round_rect(draw, (998, 614, 1370, 732), 6, "#f8fafc", "#e2e8f0")
    text(draw, (1022, 640), "Respuesta completa del LLM", 16, "#334155", True)
    text(draw, (1022, 676), "Sentimiento: Negativo", 15, "#475569")
    text(draw, (1022, 704), "Confianza: Media", 15, "#475569")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
