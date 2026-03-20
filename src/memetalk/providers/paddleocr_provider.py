from __future__ import annotations

import platform
from pathlib import Path

from memetalk.core.models import OCRExtraction, OCRStatus
from memetalk.core.providers import OCRProvider


class PaddleOCRProvider(OCRProvider):
    name = "paddleocr"

    def __init__(self, language: str = "ch", enable_mkldnn: bool = False) -> None:
        self.language = language
        self.enable_mkldnn = enable_mkldnn
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                msg = str(exc)
                if "paddle" in msg.lower() and "paddleocr" not in msg.lower():
                    raise RuntimeError(
                        f"paddleocr 已安裝但缺少 paddlepaddle 引擎：{msg}\n"
                        "請執行：pip install paddlepaddle"
                    ) from exc
                raise RuntimeError(
                    "paddleocr 尚未安裝。請執行：pip install -e .[ocr]"
                ) from exc
            # Disable OneDNN/MKL-DNN to avoid fused_conv2d crashes on Windows
            import paddle
            paddle.set_flags({
                "FLAGS_use_mkldnn": False,
            })
            self._client = PaddleOCR(
                use_angle_cls=True,
                lang=self.language,
                enable_mkldnn=self.enable_mkldnn,
            )
        return self._client

    def _translate_runtime_error(self, exc: Exception) -> Exception:
        message = str(exc)
        if "OneDnnContext does not have the input Filter" in message or "fused_conv2d" in message:
            platform_label = f"{platform.system()} {platform.machine()}".strip()
            return RuntimeError(
                "PaddleOCR 在目前的 CPU 推理路徑發生已知相容性錯誤 "
                f"({platform_label}; {message}). "
                "Windows CPU 環境請固定使用 `paddlepaddle==3.1.1` 與 `paddleocr==2.10.0`。"
            )
        return exc

    def extract_text(self, image_path: Path) -> OCRExtraction:
        try:
            result = self._get_client().ocr(str(image_path), cls=True)
        except Exception as exc:
            raise self._translate_runtime_error(exc) from exc
        lines: list[str] = []
        confidences: list[float] = []
        for group in result or []:
            for item in group or []:
                if item and len(item) > 1 and item[1]:
                    lines.append(item[1][0].strip())
                    try:
                        confidences.append(float(item[1][1]))
                    except (TypeError, ValueError, IndexError):
                        continue
        text = " ".join(line for line in lines if line)
        status = OCRStatus.SUCCESS if text else OCRStatus.EMPTY
        confidence = (sum(confidences) / len(confidences)) if confidences else None
        return OCRExtraction(text=text, has_text=bool(text), raw_lines=lines, status=status, confidence=confidence)
