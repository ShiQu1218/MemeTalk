import sys
from types import ModuleType

from memetalk.providers.paddleocr_provider import PaddleOCRProvider


def test_paddleocr_provider_disables_mkldnn(monkeypatch) -> None:
    captured_kwargs: dict[str, object] = {}

    class DummyPaddleOCR:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def ocr(self, *_args, **_kwargs):
            return []

    fake_module = ModuleType("paddleocr")
    fake_module.PaddleOCR = DummyPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    fake_paddle = ModuleType("paddle")
    fake_paddle.set_flags = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)

    provider = PaddleOCRProvider()
    provider.extract_text(__import__("pathlib").Path("dummy.png"))

    assert captured_kwargs["enable_mkldnn"] is False
    assert captured_kwargs["lang"] == "ch"


def test_paddleocr_provider_surfaces_actionable_runtime_guidance(monkeypatch) -> None:
    class DummyPaddleOCR:
        def __init__(self, **kwargs):
            pass

        def ocr(self, *_args, **_kwargs):
            raise RuntimeError(
                "(NotFound) OneDnnContext does not have the input Filter. "
                "[operator < fused_conv2d > error]"
            )

    fake_module = ModuleType("paddleocr")
    fake_module.PaddleOCR = DummyPaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    fake_paddle = ModuleType("paddle")
    fake_paddle.set_flags = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "paddle", fake_paddle)

    provider = PaddleOCRProvider()

    try:
        provider.extract_text(__import__("pathlib").Path("dummy.png"))
        raise AssertionError("Expected runtime guidance error")
    except RuntimeError as exc:
        assert "paddlepaddle==3.1.1" in str(exc)
        assert "paddleocr==2.10.0" in str(exc)
