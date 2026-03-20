from __future__ import annotations

from pathlib import Path

from memetalk.core.models import OCRExtraction
from memetalk.core.providers import OCRProvider


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(self, language: str = "ch") -> None:
        self.language = language
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise RuntimeError("The paddleocr extra is not installed. Install with `pip install -e .[ocr]`.") from exc
            self._client = PaddleOCR(use_angle_cls=True, lang=self.language)
        return self._client

    def extract_text(self, image_path: Path) -> OCRExtraction:
        result = self._get_client().ocr(str(image_path), cls=True)
        lines: list[str] = []
        for group in result or []:
            for item in group or []:
                if item and len(item) > 1 and item[1]:
                    lines.append(item[1][0].strip())
        text = " ".join(line for line in lines if line)
        return OCRExtraction(text=text, has_text=bool(text), raw_lines=lines)
