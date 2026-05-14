import sys
import os
import types
import threading
import warnings
from pathlib import Path

# Wycisz deprecation warnings z zewnętrznych pakietów (torchvision, diffusers)
warnings.filterwarnings("ignore", message=".*pretrained.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*upcast_vae.*", category=FutureWarning)

# Wycisz gadatliwe logi transformers (LOAD REPORT, UNEXPECTED keys itp.)
try:
    import transformers
    transformers.logging.set_verbosity_error()
except Exception:
    pass
try:
    import diffusers
    diffusers.logging.set_verbosity_error()
except Exception:
    pass

# ── Dodaj DLL CUDA z PyTorch do PATH (wymagane przez onnxruntime-gpu) ────────
try:
    import torch as _torch
    _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
    if os.path.isdir(_torch_lib) and _torch_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")
        os.add_dll_directory(_torch_lib)  # Python 3.8+ Windows
    del _torch, _torch_lib
except Exception:
    pass

# ── Patch: basicsr/realesrgan używają starego API torchvision ────────────────
try:
    import torchvision.transforms.functional_tensor  # noqa
except ImportError:
    import torchvision.transforms.functional as _F
    _ft = types.ModuleType('torchvision.transforms.functional_tensor')
    _ft.rgb_to_grayscale = _F.rgb_to_grayscale
    sys.modules['torchvision.transforms.functional_tensor'] = _ft
    import torchvision.transforms as _T
    _T.functional_tensor = _ft

from PIL import Image, ImageFilter, ImageEnhance

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QButtonGroup, QScrollArea,
    QFileDialog, QFrame, QSizePolicy, QMessageBox,
    QGraphicsDropShadowEffect, QDialog, QSlider,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QPoint, QRect, QRectF, QTimer, QEvent,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPainter, QPen, QBrush, QColor, QFont,
    QLinearGradient, QPainterPath, QCursor, QPalette,
    QDragEnterEvent, QDropEvent,
)

DOWNLOADS_DIR = Path.home() / "Downloads"

# ── Folder tymczasowy aplikacji — czyszczony przy starcie i zamknięciu ────────
import tempfile as _tempfile, shutil as _shutil, atexit as _atexit

_APP_TEMP = Path(_tempfile.gettempdir()) / "ai_image_enhancer"
_APP_TEMP.mkdir(exist_ok=True)

def _cleanup_app_temp():
    """Usuwa cały folder tymczasowy aplikacji."""
    try:
        _shutil.rmtree(_APP_TEMP, ignore_errors=True)
    except Exception:
        pass

# Wyczyść pozostałości z poprzedniej sesji przy starcie
_cleanup_app_temp()
_APP_TEMP.mkdir(exist_ok=True)

# Wyczyść przy zamknięciu (normalnym i przez wyjątek)
_atexit.register(_cleanup_app_temp)

# Przekieruj PyTorch i HuggingFace do folderu aplikacji
os.environ["TMPDIR"]        = str(_APP_TEMP)
os.environ["TEMP"]          = str(_APP_TEMP)
os.environ["TMP"]           = str(_APP_TEMP)
os.environ["HF_HUB_CACHE"]  = str(Path.home() / ".cache" / "huggingface" / "hub")
os.environ["TORCH_HOME"]    = str(Path.home() / ".cache" / "torch")

# ══════════════════════════════════════════════════════════════════════════════
#  STYL APLIKACJI
# ══════════════════════════════════════════════════════════════════════════════

APP_QSS = """
* { font-family: 'Segoe UI', Arial, sans-serif; }

QMainWindow, QWidget#root { background: #0b0f1a; }

QWidget#sidebar {
    background: #111827;
    border-right: 1px solid #1c2740;
}
QWidget#canvas_area { background: #0b0f1a; }

QScrollArea { background: transparent; border: none; }
QScrollArea > QWidget > QWidget { background: transparent; }

QScrollBar:vertical {
    background: #111827; width: 5px; border-radius: 3px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #1e3a5f; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #3b82f6; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

QPushButton#btn_load {
    background: #1a2540;
    color: #b0c4de;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 11px 16px;
    font-size: 13px;
    font-weight: 600;
    text-align: left;
}
QPushButton#btn_load:hover {
    background: #1e3a6e; border-color: #3b82f6; color: #ffffff;
}
QPushButton#btn_load:pressed { background: #153060; }

QPushButton#btn_generate {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #4f46e5, stop:1 #3730a3);
    color: #ffffff;
    border: none;
    border-radius: 10px;
    padding: 14px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
QPushButton#btn_generate:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #6366f1, stop:1 #4f46e5);
}
QPushButton#btn_generate:pressed { background: #312e81; }
QPushButton#btn_generate:disabled { background: #1f2937; color: #4b5563; }

QPushButton#btn_save {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #065f46, stop:1 #064e3b);
    color: #6ee7b7;
    border: 1px solid #047857;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 13px;
    font-weight: 600;
}
QPushButton#btn_save:hover {
    background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
        stop:0 #047857, stop:1 #065f46);
    color: #ffffff; border-color: #10b981;
}
QPushButton#btn_save:disabled {
    background: #111827; color: #374151; border-color: #1f2937;
}
QPushButton#btn_mark_wm {
    background: #1e2d4a;
    color: #93c5fd;
    border: 1px solid #2d4a7a;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#btn_mark_wm:hover { background: #263d6a; color: #bfdbfe; border-color: #3b82f6; }
QPushButton#btn_mark_wm[masked="true"] {
    background: #1a3320; color: #6ee7b7; border-color: #047857;
}

QLabel#lbl_title {
    color: #f1f5f9; font-size: 17px; font-weight: 700; letter-spacing: 0.3px;
}
QLabel#lbl_sub   { color: #3b82f6; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; }
QLabel#lbl_section { color: #475569; font-size: 10px; font-weight: 600; letter-spacing: 1.5px; }
QLabel#lbl_status  { color: #64748b; font-size: 11px; }
QLabel#lbl_info    { color: #334155; font-size: 10px; }

QPushButton#op_btn {
    background: transparent;
    color: #94a3b8;
    border: none;
    border-radius: 8px;
    padding: 8px 12px 8px 14px;
    font-size: 12px;
    font-weight: 500;
    text-align: left;
}
QPushButton#op_btn:hover { background: #1a2540; color: #e2e8f0; }
QPushButton#op_btn[active="true"] {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #1a3160, stop:1 #1a2540);
    color: #60a5fa;
    font-weight: 700;
    border-left: 3px solid #3b82f6;
    padding-left: 11px;
}

QFrame#divider { background: #1c2740; max-height: 1px; min-height: 1px; }
"""

# ══════════════════════════════════════════════════════════════════════════════
#  MODUŁY PRZETWARZANIA
# ══════════════════════════════════════════════════════════════════════════════

def get_torch():
    import torch
    return torch

def get_device():
    torch = get_torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def process_remove_bg(img: Image.Image) -> Image.Image:
    import numpy as np
    import onnxruntime as ort
    import os

    model_path = os.path.join(os.path.expanduser("~"), ".u2net", "birefnet-massive.onnx")

    # Pobierz model jeśli brak (rembg go pobiera)
    if not os.path.exists(model_path):
        from rembg.sessions.birefnet_massive import BiRefNetSessionMassive
        BiRefNetSessionMassive.download_models()

    # ORT_DISABLE_ALL pomija shape-inference, która crashuje DirectML na dużych modelach
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=providers)

    # Preprocessing (identyczny jak rembg BiRefNetSessionGeneral)
    orig_size = img.size
    im = img.convert("RGB").resize((1024, 1024), Image.Resampling.LANCZOS)
    im_ary = np.array(im, dtype=np.float32)
    im_ary = im_ary / max(float(np.max(im_ary)), 1e-6)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    inp = np.zeros((1024, 1024, 3), dtype=np.float32)
    inp[:, :, 0] = (im_ary[:, :, 0] - mean[0]) / std[0]
    inp[:, :, 1] = (im_ary[:, :, 1] - mean[1]) / std[1]
    inp[:, :, 2] = (im_ary[:, :, 2] - mean[2]) / std[2]
    inp = np.expand_dims(inp.transpose((2, 0, 1)), 0).astype(np.float32)

    # Inference
    input_name = session.get_inputs()[0].name
    ort_outs = session.run(None, {input_name: inp})

    # Postprocessing
    pred = 1.0 / (1.0 + np.exp(-ort_outs[0][:, 0, :, :]))
    ma, mi = float(np.max(pred)), float(np.min(pred))
    pred = (pred - mi) / (ma - mi + 1e-8)
    pred = np.squeeze(pred)

    mask = Image.fromarray((pred * 255).astype(np.uint8), mode="L")
    mask = mask.resize(orig_size, Image.Resampling.LANCZOS)

    result = img.convert("RGBA")
    result.putalpha(mask)
    return result


def process_upscale(img: Image.Image, scale: int) -> Image.Image:
    torch = get_torch()
    device = get_device()
    # Próba HAT-L (najlepsza jakość) — wymaga basicsr z hat_arch + pliku wag
    try:
        from basicsr.archs.hat_arch import HAT          # ImportError → pomijamy cicho
        import numpy as np
        model_path = Path(__file__).parent / "models" / "HAT-L_SRx4_ImageNet-pretrain.pth"
        if not model_path.exists():
            raise FileNotFoundError  # brak wag → przejdź do Real-ESRGAN
        model = HAT(
            upscale=4, in_chans=3, img_size=64, window_size=16,
            compress_ratio=3, squeeze_factor=30, conv_scale=0.01,
            overlap_ratio=0.5, img_range=1., depths=[6,6,6,6,6,6],
            embed_dim=180, num_heads=[6,6,6,6,6,6],
            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'
        )
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state['params_ema'] if 'params_ema' in state else state, strict=True)
        model.eval().to(device)
        img_np = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
        out_np = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result = Image.fromarray((out_np * 255).astype(np.uint8))
        if scale == 8:
            img_t2 = torch.from_numpy(
                np.array(result).astype(np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                out2 = model(img_t2)
            out_np2 = out2.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            result = Image.fromarray((out_np2 * 255).astype(np.uint8))
        elif scale == 2:
            w, h = result.size
            result = result.resize((w // 2, h // 2), Image.LANCZOS)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return result
    except ImportError:
        pass  # HAT-L nie zainstalowany — Real-ESRGAN poniżej
    except FileNotFoundError:
        pass  # brak wag HAT-L — Real-ESRGAN poniżej
    except Exception as e:
        print(f"[HAT-L] Blad: {e}")
    return _upscale_realesrgan(img, scale)


def _get_realesrgan_weights(scale: int = 4) -> Path:
    """Zwraca ścieżkę do wag Real-ESRGAN; pobiera automatycznie jeśli brak."""
    import urllib.request, time

    MODELS = {
        2:  ("RealESRGAN_x2plus.pth",
             "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
        4:  ("RealESRGAN_x4plus.pth",
             "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
        # model ogólny: zdjęcia + grafiki + screenshoty (bez artefaktów na UI)
        42: ("realesr-general-x4v3.pth",
             "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"),
    }
    key = 2 if scale == 2 else (42 if scale == 42 else 4)
    name, url = MODELS[key]

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(exist_ok=True)
    dest = models_dir / name

    if not dest.exists():
        print(f"[Real-ESRGAN] Pobieranie modelu {name} (~64 MB)...")
        tmp = dest.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dest)
            print(f"[Real-ESRGAN] Model zapisany: {dest}")
        except Exception as e:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"Nie mozna pobrac modelu Real-ESRGAN: {e}") from e

    return dest


def _upscale_realesrgan(img: Image.Image, scale: int, model_key: int = 0, cpu_only: bool = False) -> Image.Image:
    """
    model_key=0   → auto (x2plus lub x4plus zależnie od scale)
    model_key=42  → realesr-general-x4v3 (screenshoty / grafiki / UI)
    cpu_only=True → wymusza CPU, żeby zwolnić VRAM dla kolejnego modelu
    """
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import numpy as np, gc

    device = "cpu" if cpu_only else get_device()

    if model_key == 42:
        try:
            from basicsr.archs.srvgg_arch import SRVGGNetCompact
            model = SRVGGNetCompact(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_conv=32, upscale=4, act_type='prelu'
            )
        except ImportError:
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                            num_block=23, num_grow_ch=32, scale=4)
        model_path = _get_realesrgan_weights(42)
        model_scale = 4
    else:
        model_scale = 2 if scale == 2 else 4
        model_path = _get_realesrgan_weights(model_scale)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=model_scale)

    upsampler = RealESRGANer(
        scale=model_scale,
        model_path=str(model_path),
        model=model,
        tile=512, tile_pad=10, pre_pad=0,
        half=False if cpu_only else (device == "cuda"),
        device=device,
    )
    img_np = np.array(img.convert("RGB"))
    out, _ = upsampler.enhance(img_np, outscale=scale)

    # Jawne zwolnienie modelu z pamięci
    del upsampler, model
    gc.collect()

    return Image.fromarray(out)


def process_upscale_screen(img: Image.Image) -> Image.Image:
    """Upscaling x4 dla screenshotów/grafik — model general-x4v3, brak artefaktów na UI."""
    return _upscale_realesrgan(img, 4, model_key=42)


def process_restore_face(img: Image.Image) -> Image.Image:
    """
    Przywróć twarz — GFPGAN v1.4.
    Wykrywa i naprawia twarze (budowa: transformer + codebook).
    Fallback: Real-ESRGAN+maska gdy brak modelu/pakietu.
    """
    import numpy as np, cv2

    model_path = Path(__file__).parent / "models" / "GFPGANv1.4.pth"
    try:
        if not model_path.exists():
            raise FileNotFoundError("Brak GFPGANv1.4.pth")
        from gfpgan import GFPGANer
        # Przekieruj pobieranie modeli detekcji do naszego folderu models/
        import facexlib.utils.face_restoration_helper as _frh
        _frh.ROOT_DIR = str(Path(__file__).parent / "models")
        restorer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=None,
        )
        img_bgr = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        _, _, output = restorer.enhance(
            img_bgr, has_aligned=False, only_center_face=False, paste_back=True
        )
        return Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    except Exception as e:
        print(f"[GFPGAN] {e} — fallback ESRGAN")
        # Fallback: ESRGAN x4 na twarzach (Haar cascade)
        src = img.convert("RGB")
        src_arr = np.array(src)
        h, w = src_arr.shape[:2]
        gray = cv2.cvtColor(src_arr, cv2.COLOR_RGB2GRAY)
        cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = cascade.detectMultiScale(gray, 1.1, 4, minSize=(40, 40))
        if len(faces) == 0:
            up = _upscale_realesrgan(src, 4, cpu_only=False)
            return up.resize((w, h), Image.LANCZOS)
        result_arr = src_arr.copy()
        for (fx, fy, fw, fh) in faces:
            px, py = int(fw * 0.4), int(fh * 0.4)
            x1, y1 = max(0, fx - px), max(0, fy - py)
            x2, y2 = min(w, fx + fw + px), min(h, fy + fh + py)
            crop = Image.fromarray(src_arr[y1:y2, x1:x2])
            cw, ch = crop.size
            restored = np.array(_upscale_realesrgan(crop, 4, cpu_only=False).resize((cw, ch), Image.LANCZOS))
            cy2, cx2 = ch / 2.0, cw / 2.0
            Y, X = np.ogrid[:ch, :cw]
            mask = np.clip(1.0 - ((X - cx2) / max(cx2, 1))**2 - ((Y - cy2) / max(cy2, 1))**2, 0, 1)**0.5
            m3 = mask[:, :, np.newaxis]
            result_arr[y1:y2, x1:x2] = (restored * m3 + result_arr[y1:y2, x1:x2] * (1 - m3)).astype(np.uint8)
        return Image.fromarray(result_arr)


def process_diffbir(img: Image.Image) -> Image.Image:
    """
    Denoise/Deblur — dwuetapowy pipeline bez modeli generatywnych:
      1. Odszumianie: bilateral filter (edge-preserving, nie rozmywa krawędzi)
      2. Deblur: Unsharp Mask (wyostrzenie przez odjęcie rozmytej kopii)
    Zachowuje 100% oryginalnych kolorów i rozdzielczości.
    """
    import numpy as np, cv2

    img_np = np.array(img.convert("RGB"))

    # Krok 1: bilateral denoise — usuwa szum zachowując krawędzie
    # d=9: rozmiar okna; sigmaColor=75: tolerancja koloru; sigmaSpace=75: tolerancja przestrzenna
    denoised = cv2.bilateralFilter(img_np, d=9, sigmaColor=75, sigmaSpace=75)

    # Krok 2: Unsharp Mask — deblur
    # blur → odejmij od oryginału → wzmocnij krawędzie
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.5)
    sharpened = cv2.addWeighted(denoised, 1.6, blurred, -0.6, 0)

    return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))


def process_colorize(img: Image.Image) -> Image.Image:
    torch = get_torch()
    device = get_device()
    try:
        import numpy as np, cv2
        from ddcolor import DDColor as DDColorModel
        model_path = Path(__file__).parent / "models" / "ddcolor_artistic.pth"
        if not model_path.exists():
            raise FileNotFoundError("Brak modelu DDColor")
        model = DDColorModel(encoder_name='convnext-l', input_size=[512, 512],
                             num_output_channels=2, last_norm='Spectral',
                             do_normalize=False).to(device)
        ckpt = torch.load(model_path, map_location=device)
        model.load_state_dict(ckpt['params'])
        model.eval()
        img_gray = img.convert("L").convert("RGB")
        img_np = np.array(img_gray).astype(np.float32) / 255.0
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img_t)
        out_np = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        result = Image.fromarray((out_np * 255).astype(np.uint8))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        return result
    except Exception as e:
        print(f"DDColor niedostepny ({e}), podstawowa koloryzacja...")
        return ImageEnhance.Color(img.convert("RGB")).enhance(1.5)


def process_sharpen(img: Image.Image) -> Image.Image:
    """
    Wyostrzenie z odtwarzaniem detali (jak Topaz Sharpen AI):
      1. Real-ESRGAN x4 — sieć neuronowa rekonstruuje utracone pike
      2. Przeskalowanie do oryginalnego rozmiaru (LANCZOS)
      3. Lekki Unsharp Mask na warstwach mikro / krawędź / sylwetka
    Fallback (brak GPU / modelu): tylko wielopoziomowy USM.
    """
    import numpy as np, cv2

    orig_w, orig_h = img.size
    has_alpha = img.mode == "RGBA"
    if has_alpha:
        r, g, b, a = img.split()
        img_rgb = Image.merge("RGB", (r, g, b))
    else:
        img_rgb = img.convert("RGB")

    # ── Etap 1: Real-ESRGAN — odtwarzanie detali ─────────────────────────────
    try:
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        device = get_device()
        model_path = _get_realesrgan_weights(4)
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        upsampler = RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=512, tile_pad=10, pre_pad=0,
            half=(device == "cuda"),
            device=device,
        )
        img_np = np.array(img_rgb)
        out_4x, _ = upsampler.enhance(img_np, outscale=4)
        # Przeskaluj z powrotem do oryginalnego rozmiaru — detal pozostaje
        enhanced = Image.fromarray(out_4x).resize((orig_w, orig_h), Image.LANCZOS)
        base = np.array(enhanced, dtype=np.float32)
    except Exception as e:
        print(f"[Wyostrzenie] Real-ESRGAN niedostepny ({e}), uzywam samego USM")
        base = np.array(img_rgb, dtype=np.float32)

    # ── Etap 2: Wielopoziomowy Unsharp Mask (wykończenie) ────────────────────
    def usm(src, sigma, amount):
        blur = cv2.GaussianBlur(src, (0, 0), sigma)
        return src + amount * (src - blur)

    sharp = usm(base, 0.6, 0.4)   # mikro-detale
    sharp = usm(sharp, 1.5, 0.3)  # krawędzie
    sharp = usm(sharp, 4.0, 0.15) # sylwetka (delikatnie)

    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    result = Image.fromarray(sharp, "RGB")

    if has_alpha:
        result = result.convert("RGBA")
        result.putalpha(a)
    return result


def _detect_watermark_mask(img_np) -> "np.ndarray":
    """
    Smart watermark detection — skanuje cały obraz bez zakodowanych pozycji.

    Znak wodny identyfikujemy przez 3 niezależne dowody:
      · Anomalia kolorystyczna — region kolorystycznie "obcy" wobec tła
        (oceniana przez porównanie z silnie rozmytym tłem)
      · Gęstość krawędzi — tekst i logo mają dużo krawędzi
      · Semi-transparentność — AI watermarki to odbarwione overlaye

    Twarda reguła: znak wodny ≤ 12% powierzchni obrazu.
    Główna treść (logo USB, główny obiekt) jest za duża — automatycznie odrzucana.
    """
    import numpy as np
    import cv2

    h, w   = img_np.shape[:2]
    gray   = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    img_f  = img_np.astype(np.float32)
    total  = h * w
    max_wm = total * 0.12          # twarda granica: >12% = treść, nie watermark

    # ── MAP DOWODÓW: każdy detektor dodaje swój głos ──────────────────────────
    score_map = np.zeros((h, w), dtype=np.float32)

    # 1. MSER — wykrywa litery, cyfry, ikony (każdy region < 6% obrazu)
    mser = cv2.MSER_create(delta=5, min_area=20, max_area=int(max_wm * 0.5))
    regions_mser, _ = mser.detectRegions(gray)
    for pts in regions_mser:
        hull = cv2.convexHull(pts.reshape(-1, 1, 2))
        if cv2.contourArea(hull) < max_wm * 0.5:
            cv2.fillConvexPoly(score_map, hull, 1.0)

    # 2. Jasne obszary (jasne logo na ciemnym tle) — oceniane relative do tła
    blur_sz = min(max(min(w, h) // 20 * 2 + 1, 31), 101)
    bg_coarse = cv2.GaussianBlur(gray, (blur_sz, blur_sz), 0).astype(np.float32)
    bright_res = np.clip(gray.astype(np.float32) - bg_coarse - 20, 0, None)
    if bright_res.max() > 0:
        score_map = np.maximum(score_map, bright_res / bright_res.max())

    # 3. Semi-transparentny overlay (odbarwiony + powyżej średniej jasności)
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    avg_val  = float(np.mean(val))
    semi = ((sat < 35) & (val > avg_val + 15)).astype(np.float32)
    semi = cv2.morphologyEx(semi, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    score_map = np.maximum(score_map, semi * 0.7)

    # ── Zamknij luki i oznacz spójne regiony ─────────────────────────────────
    binary = (score_map > 0.35).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=3)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Globalne tło (duże rozmycie) — do oceny anomalii kolorystycznej
    k_bg = min(max(min(w, h) // 5 * 2 + 1, 51), 101)
    bg_large = cv2.GaussianBlur(img_np, (k_bg, k_bg), 0).astype(np.float32)
    edges_all = cv2.Canny(gray, 50, 150)

    # ── OCENA każdego kandydata ───────────────────────────────────────────────
    candidates = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        bw   = stats[i, cv2.CC_STAT_WIDTH]
        bh   = stats[i, cv2.CC_STAT_HEIGHT]

        # Twarde odrzucenia
        if area < 80:              continue  # szum
        if area / total > 0.12:    continue  # za duży = główna treść obrazu
        if area / (bw * bh + 1) < 0.04:  continue  # zbyt rozrzucony

        region = labels == i

        # Anomalia: jak bardzo ten region różni się od szacowanego tła?
        pixel_diff = np.abs(img_f[region] - bg_large[region])
        anomaly = float(np.mean(pixel_diff)) / 127.5   # 0..~2, normalnie 0..1

        # Gęstość krawędzi: tekst/logo = dużo krawędzi
        edge_dens = float(np.mean(edges_all[region] > 0))

        # Oddalenie od centrum (watermarki preferują brzegi, ale nie jest to decydujące)
        cx, cy = centroids[i]
        dist_ctr = np.sqrt(((cx / w) - 0.5) ** 2 + ((cy / h) - 0.5) ** 2) * 1.414

        # Wynik łączny — anomalia ma największy wpływ
        score = 0.55 * min(anomaly, 1.0) + 0.30 * edge_dens + 0.15 * dist_ctr
        candidates.append((score, area / total, i))

    if not candidates:
        return np.zeros((h, w), dtype=np.uint8)

    candidates.sort(reverse=True)

    # Akceptuj kandydatów: wynik ≥ 50% najlepszego I ≥ 0.15 bezwzględnie
    best_score = candidates[0][0]
    threshold  = max(best_score * 0.50, 0.15)

    result = np.zeros((h, w), dtype=np.uint8)
    taken  = 0.0
    for score, rel_area, idx in candidates:
        if score < threshold or taken + rel_area > 0.12:
            break
        result[labels == idx] = 255
        taken += rel_area

    # ── Finalna dylatacja i wygładzenie ──────────────────────────────────────
    result = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)), iterations=2)
    result = cv2.GaussianBlur(result, (9, 9), 0)
    _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)

    return result


def process_remove_watermark(img: Image.Image, manual_mask=None) -> Image.Image:
    """Usuwa znaki wodne — LaMa inpainting z maską ręczną lub auto-detekcją."""
    import numpy as np

    img_rgb = img.convert("RGB")
    img_np  = np.array(img_rgb)

    # Użyj maski ręcznej jeśli dostarczona, inaczej auto-detekcja
    if manual_mask is not None:
        mask = manual_mask.astype(np.uint8)
    else:
        mask = _detect_watermark_mask(img_np)
        if mask.max() == 0:
            raise RuntimeError(
                "Nie wykryto znaku wodnego automatycznie.\n"
                "Użyj przycisku 'Zaznacz znak wodny' aby wskazać go ręcznie."
            )

    mask_pil = Image.fromarray(mask).convert("L")

    # ── Próba LaMa (najlepsza jakość) ────────────────────────────────────────
    try:
        from simple_lama_inpainting import SimpleLama
        lama   = SimpleLama()
        result = lama(img_rgb, mask_pil)
        return result
    except Exception as e_lama:
        print(f"LaMa niedostepny ({e_lama}), fallback OpenCV telea...")

    # ── Fallback: OpenCV Telea inpainting ─────────────────────────────────────
    import cv2
    result_np = cv2.inpaint(img_np, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    return Image.fromarray(result_np)


_UPSCALE_OPTIONS = [
    ("x2",         "Upscaling x2",          "Real-ESRGAN",  lambda img: process_upscale(img, 2)),
    ("x4",         "Upscaling x4",          "Real-ESRGAN",  lambda img: process_upscale(img, 4)),
    ("x8",         "Upscaling x8",          "Real-ESRGAN",  lambda img: process_upscale(img, 8)),
    ("screenshot", "Upscaling screenshot",  "general-x4v3", process_upscale_screen),
]

OPERATIONS = [
    ("ULEPSZ OBRAZ",        None,          None),
    ("Usuń tło",            "BiRefNet",    process_remove_bg),
    ("Usuń znak wodny",     "LaMa",        process_remove_watermark),
    ("Wyostrzenie",         "OpenCV",      process_sharpen),
    ("UPSCALING",           None,          None),
    ("Upscaling",           "Real-ESRGAN", "_UPSCALE_MENU"),
    ("RESTAURACJA",         None,          None),
    ("Przywroc twarz",      "GFPGAN",      process_restore_face),
    ("Denoise/Deblur",      "DiffBIR",     process_diffbir),
    ("Koloryzacja",         "DDColor",     process_colorize),
]


# ══════════════════════════════════════════════════════════════════════════════
#  WORKER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class WorkerThread(QThread):
    done  = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, img):
        super().__init__()
        self._fn  = fn
        self._img = img

    def run(self):
        try:
            self.done.emit(self._fn(self._img.copy()))
        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  PIL -> QPixmap
# ══════════════════════════════════════════════════════════════════════════════

def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    if img.mode == "RGBA":
        data = img.tobytes("raw", "RGBA")
        bpl  = img.width * 4
        fmt  = QImage.Format.Format_RGBA8888
    else:
        img  = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        bpl  = img.width * 3
        fmt  = QImage.Format.Format_RGB888
    # PyQt6: zawsze podaj bytesPerLine (5-arg) — bez tego 4-arg constructor
    # może zinterpretować Format enum jako bytesPerLine, powodując slant
    qimg = QImage(data, img.width, img.height, bpl, fmt)
    return QPixmap.fromImage(qimg)


# ══════════════════════════════════════════════════════════════════════════════
#  WIDGET: POROWNANIE PRZED / PO
# ══════════════════════════════════════════════════════════════════════════════

class BeforeAfterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._before_pil: Image.Image | None = None
        self._after_pil:  Image.Image | None = None
        self._split      = 0.5
        self._dragging   = False
        self._generating = False
        self._pulse_val  = 80
        self._pulse_dir  = 1
        self.setAcceptDrops(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(500, 360)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._do_pulse)

    def set_before(self, img: Image.Image):
        self._before_pil = img
        self._after_pil  = None
        self._generating = False
        self._timer.stop()
        self.update()

    def set_after(self, img: Image.Image):
        self._after_pil  = img
        self._split      = 0.5
        self._generating = False
        self._timer.stop()
        self.update()

    def show_generating(self):
        self._generating = True
        self._pulse_val  = 80
        self._pulse_dir  = 1
        self._timer.start(18)
        self.update()

    def clear(self):
        self._before_pil = None
        self._after_pil  = None
        self._generating = False
        self._timer.stop()
        self.update()

    def _do_pulse(self):
        self._pulse_val += self._pulse_dir * 4
        if self._pulse_val >= 255:
            self._pulse_val = 255; self._pulse_dir = -1
        elif self._pulse_val <= 80:
            self._pulse_val = 80;  self._pulse_dir =  1
        self.update()

    # Drag & drop
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            win = self.window()
            if hasattr(win, '_load_path'):
                win._load_path(path)

    # Mysz
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton and self._after_pil:
            self._dragging = True
            self._split = max(0.0, min(1.0, ev.pos().x() / self.width()))
            self.update()

    def mouseMoveEvent(self, ev):
        if self._after_pil:
            sx = int(self._split * self.width())
            near = abs(ev.pos().x() - sx) < 32
            self.setCursor(QCursor(
                Qt.CursorShape.SizeHorCursor if near else Qt.CursorShape.ArrowCursor
            ))
        if self._dragging and self._after_pil:
            self._split = max(0.04, min(0.96, ev.pos().x() / self.width()))
            self.update()

    def mouseReleaseEvent(self, ev):
        self._dragging = False

    # Rysowanie
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor("#0b0f1a"))

        if self._generating:
            self._paint_generating(p, W, H)
        elif self._before_pil and self._after_pil:
            self._paint_comparison(p, W, H)
        elif self._before_pil:
            self._paint_single(p, W, H)
        else:
            self._paint_placeholder(p, W, H)
        p.end()

    def _fit(self, img: Image.Image, W: int, H: int) -> QPixmap:
        return pil_to_qpixmap(img).scaled(
            W, H,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _paint_placeholder(self, p: QPainter, W: int, H: int):
        cx, cy = W // 2, H // 2
        bw = min(W - 100, 440)
        bh = min(H - 100, 260)
        rx, ry = cx - bw // 2, cy - bh // 2

        pen = QPen(QColor("#1c2d45"), 2, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        path = QPainterPath()
        path.addRoundedRect(QRectF(rx, ry, bw, bh), 20, 20)
        p.drawPath(path)

        ir = QRect(cx - 28, cy - 56, 56, 46)
        p.setBrush(QBrush(QColor("#0d1f35")))
        p.setPen(QPen(QColor("#1e3a5f"), 1))
        p.drawRoundedRect(ir, 12, 12)
        p.setFont(QFont("Segoe UI", 22))
        p.setPen(QColor("#1e3a8a"))
        p.drawText(ir, Qt.AlignmentFlag.AlignCenter, "\u2191")

        p.setFont(QFont("Segoe UI", 14, QFont.Weight.Medium))
        p.setPen(QColor("#2d4a6e"))
        p.drawText(QRect(0, cy + 10, W, 28), Qt.AlignmentFlag.AlignCenter,
                   "Przeciagnij zdjecie tutaj")
        p.setFont(QFont("Segoe UI", 11))
        p.setPen(QColor("#1e2d3d"))
        p.drawText(QRect(0, cy + 40, W, 22), Qt.AlignmentFlag.AlignCenter,
                   "lub kliknij  Wczytaj zdjecie  po lewej")
        p.setFont(QFont("Segoe UI", 10))
        p.setPen(QColor("#172030"))
        p.drawText(QRect(0, cy + 64, W, 18), Qt.AlignmentFlag.AlignCenter,
                   "JPG  \u00b7  PNG  \u00b7  BMP  \u00b7  TIFF  \u00b7  WebP")

    def _paint_single(self, p: QPainter, W: int, H: int):
        px = self._fit(self._before_pil, W, H)
        ox = (W - px.width())  // 2
        oy = (H - px.height()) // 2
        p.drawPixmap(ox, oy, px)
        self._badge(p, 16, 16, "ORYGINAL", QColor("#475569"))

    def _paint_comparison(self, p: QPainter, W: int, H: int):
        sx = int(W * self._split)
        bef = self._fit(self._before_pil, W, H)
        aft = self._fit(self._after_pil,  W, H)
        ox  = (W - bef.width())  // 2
        oy  = (H - bef.height()) // 2

        p.save()
        p.setClipRect(0, 0, sx, H)
        p.drawPixmap(ox, oy, bef)
        p.restore()

        p.save()
        p.setClipRect(sx, 0, W - sx, H)
        p.drawPixmap(ox, oy, aft)
        p.restore()

        # linia
        p.setPen(QPen(QColor(0, 0, 0, 100), 1))
        p.drawLine(sx, 0, sx, H)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(sx, 0, sx, H)

        # kolo suwaka
        r  = 20
        cy = H // 2
        grad = QLinearGradient(sx - r, cy - r, sx + r, cy + r)
        grad.setColorAt(0.0, QColor("#6366f1"))
        grad.setColorAt(1.0, QColor("#3b82f6"))
        p.setBrush(QBrush(grad))
        p.setPen(QPen(QColor("#c7d2fe"), 2))
        p.drawEllipse(QPoint(sx, cy), r, r)
        p.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        p.setPen(QColor("#ffffff"))
        p.drawText(QRect(sx - r, cy - r, r * 2, r * 2),
                   Qt.AlignmentFlag.AlignCenter, "\u27fa")

        if sx > 70:
            self._badge(p, 16, 16, "PRZED", QColor("#64748b"))
        if sx < W - 50:
            self._badge(p, sx + 12, 16, "PO", QColor("#60a5fa"))

    def _badge(self, p: QPainter, x: int, y: int, text: str, color: QColor):
        p.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        fm  = p.fontMetrics()
        tw  = fm.horizontalAdvance(text)
        bw, bh = tw + 20, 24
        p.setBrush(QBrush(QColor(11, 15, 26, 210)))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRect(x, y, bw, bh), 5, 5)
        p.setPen(color)
        p.drawText(QRect(x, y, bw, bh), Qt.AlignmentFlag.AlignCenter, text)

    def _paint_generating(self, p: QPainter, W: int, H: int):
        if self._before_pil:
            px = self._fit(self._before_pil, W, H)
            ox = (W - px.width())  // 2
            oy = (H - px.height()) // 2
            p.setOpacity(0.18)
            p.drawPixmap(ox, oy, px)
            p.setOpacity(1.0)

        p.fillRect(0, 0, W, H, QColor(11, 15, 26, 180))

        col = QColor("#3b82f6")
        col.setAlpha(self._pulse_val)
        p.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        p.setPen(col)
        p.drawText(QRect(0, H // 2 - 36, W, 52),
                   Qt.AlignmentFlag.AlignCenter, "Generuje\u2026")

        col2 = QColor("#475569")
        col2.setAlpha(self._pulse_val)
        p.setFont(QFont("Segoe UI", 12))
        p.setPen(col2)
        p.drawText(QRect(0, H // 2 + 22, W, 28),
                   Qt.AlignmentFlag.AlignCenter,
                   "Model AI przetwarza zdjecie, prosze czekac")

        # pasek postepuping-pong
        bar_w = int(W * 0.38)
        bar_h = 3
        bx    = (W - bar_w) // 2
        by    = H // 2 + 62
        p.setBrush(QBrush(QColor("#1e2d4a")))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(QRect(bx, by, bar_w, bar_h), 2, 2)
        seg_w = int(bar_w * 0.3)
        ratio = (self._pulse_val - 80) / 175.0
        seg_x = bx + int((bar_w - seg_w) * ratio)
        g = QLinearGradient(seg_x, 0, seg_x + seg_w, 0)
        g.setColorAt(0.0, QColor(59, 130, 246, 0))
        g.setColorAt(0.5, QColor("#3b82f6"))
        g.setColorAt(1.0, QColor(59, 130, 246, 0))
        p.setBrush(QBrush(g))
        p.drawRoundedRect(QRect(seg_x, by, seg_w, bar_h), 2, 2)


# ══════════════════════════════════════════════════════════════════════════════
#  PRZYCISK OPERACJI
# ══════════════════════════════════════════════════════════════════════════════

class MaskPainterDialog(QDialog):
    """
    Okno do ręcznego zaznaczania znaku wodnego pędzlem.
    Użytkownik maluje czerwonym pędzlem na obrazie; maska trafia do LaMa.
    """

    def __init__(self, image: Image.Image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Zaznacz znak wodny  —  maluj gdzie chcesz usunąć")
        self.setModal(True)
        self.setStyleSheet("background:#0d1526; color:#e2e8f0;")

        self._image   = image
        self._mask_np = None
        self._drawing = False
        self._last_pt = None
        self._brush_r = 20

        # Skalowanie obrazu do ekranu (max 88% ekranu)
        screen = QApplication.primaryScreen().availableGeometry()
        iw, ih = image.size
        scale  = min(screen.width() * 0.88 / iw, screen.height() * 0.82 / ih, 1.0)
        self._dw    = max(int(iw * scale), 1)
        self._dh    = max(int(ih * scale), 1)
        self._scale = scale

        # Tło
        img_rgb = image.convert("RGB")
        raw     = img_rgb.tobytes("raw", "RGB")
        qimg    = QImage(raw, iw, ih, iw * 3, QImage.Format.Format_RGB888)
        self._bg = QPixmap.fromImage(qimg).scaled(
            self._dw, self._dh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        # Warstwa maski (przezroczysta)
        self._overlay = QPixmap(self._dw, self._dh)
        self._overlay.fill(Qt.GlobalColor.transparent)

        self._build_ui()
        self.resize(self._dw + 32, self._dh + 90)

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(10)

        # Canvas
        self._lbl = QLabel()
        self._lbl.setFixedSize(self._dw, self._dh)
        self._lbl.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._lbl.setMouseTracking(True)
        self._lbl.installEventFilter(self)
        lay.addWidget(self._lbl, 0, Qt.AlignmentFlag.AlignCenter)
        self._redraw()

        # Pasek narzędzi
        bar = QHBoxLayout()
        bar.setSpacing(10)

        bar.addWidget(QLabel("Pędzel:"))

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(4, 120)
        self._slider.setValue(self._brush_r)
        self._slider.setFixedWidth(180)
        self._slider.setStyleSheet(
            "QSlider::groove:horizontal{height:4px;background:#1e3a5f;border-radius:2px;}"
            "QSlider::handle:horizontal{width:14px;height:14px;margin:-5px 0;"
            "background:#3b82f6;border-radius:7px;}"
            "QSlider::sub-page:horizontal{background:#3b82f6;border-radius:2px;}"
        )
        self._slider.valueChanged.connect(lambda v: (
            setattr(self, "_brush_r", v),
            self._lbl_sz.setText(f"{v} px"),
        ))
        bar.addWidget(self._slider)

        self._lbl_sz = QLabel(f"{self._brush_r} px")
        self._lbl_sz.setFixedWidth(44)
        bar.addWidget(self._lbl_sz)
        bar.addStretch()

        btn_clear = QPushButton("Wyczyść")
        btn_clear.setFixedHeight(34)
        btn_clear.setStyleSheet(
            "QPushButton{background:#1e2d4a;color:#cbd5e1;border:1px solid #2d4a7a;"
            "border-radius:6px;padding:0 14px;}"
            "QPushButton:hover{background:#263d6a;}"
        )
        btn_clear.clicked.connect(self._clear)
        bar.addWidget(btn_clear)

        btn_ok = QPushButton("✓  Zatwierdź maskę")
        btn_ok.setFixedHeight(34)
        btn_ok.setStyleSheet(
            "QPushButton{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,"
            "stop:0 #4f46e5,stop:1 #3b82f6);color:#fff;border:none;"
            "border-radius:6px;padding:0 18px;font-weight:600;}"
            "QPushButton:hover{background:#4338ca;}"
        )
        btn_ok.clicked.connect(self._confirm)
        bar.addWidget(btn_ok)

        btn_cancel = QPushButton("Anuluj")
        btn_cancel.setFixedHeight(34)
        btn_cancel.setStyleSheet(btn_clear.styleSheet())
        btn_cancel.clicked.connect(self.reject)
        bar.addWidget(btn_cancel)

        lay.addLayout(bar)

    # ── Rysowanie ─────────────────────────────────────────────────────────────
    def _redraw(self):
        out = QPixmap(self._dw, self._dh)
        p = QPainter(out)
        p.drawPixmap(0, 0, self._bg)
        p.setOpacity(0.55)
        p.drawPixmap(0, 0, self._overlay)
        p.end()
        self._lbl.setPixmap(out)

    def _paint_point(self, pt: QPoint):
        p = QPainter(self._overlay)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(220, 50, 50, 255))
        r = self._brush_r
        if self._last_pt is not None and self._last_pt != pt:
            dx = pt.x() - self._last_pt.x()
            dy = pt.y() - self._last_pt.y()
            steps = max(int((dx**2 + dy**2) ** 0.5), 1)
            for i in range(steps + 1):
                t = i / steps
                ix = int(self._last_pt.x() + t * dx)
                iy = int(self._last_pt.y() + t * dy)
                p.drawEllipse(QPoint(ix, iy), r, r)
        else:
            p.drawEllipse(pt, r, r)
        p.end()
        self._last_pt = pt
        self._redraw()

    def _clear(self):
        self._overlay.fill(Qt.GlobalColor.transparent)
        self._last_pt = None
        self._redraw()

    # ── Event filter ──────────────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if obj is self._lbl:
            t = event.type()
            if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                self._drawing = True
                self._last_pt = None
                self._paint_point(event.pos())
            elif t == QEvent.Type.MouseMove and self._drawing:
                self._paint_point(event.pos())
            elif t == QEvent.Type.MouseButtonRelease and event.button() == Qt.MouseButton.LeftButton:
                self._drawing = False
                self._last_pt = None
        return super().eventFilter(obj, event)

    # ── Zatwierdzenie → maska numpy ───────────────────────────────────────────
    def _confirm(self):
        import numpy as np, cv2

        # Konwertuj overlay pixmap → numpy alpha channel
        qimg = self._overlay.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        w, h = qimg.width(), qimg.height()
        ptr  = qimg.bits()
        ptr.setsize(h * w * 4)
        arr   = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        alpha = arr[:, :, 3]                          # kanał alpha = maska
        binary = (alpha > 20).astype(np.uint8) * 255

        if binary.max() == 0:
            QMessageBox.warning(self, "Pusta maska",
                                "Nie zaznaczono żadnego obszaru.\nPomaluj znak wodny pędzlem.")
            return

        # Skaluj maskę do rozmiaru oryginalnego obrazu
        iw, ih = self._image.size
        if (w, h) != (iw, ih):
            binary = cv2.resize(binary, (iw, ih), interpolation=cv2.INTER_NEAREST)

        # Małe rozmycie i dylatacja — łagodne przejście
        binary = cv2.dilate(binary,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
                            iterations=2)
        binary = cv2.GaussianBlur(binary, (5, 5), 0)
        _, binary = cv2.threshold(binary, 100, 255, cv2.THRESH_BINARY)

        self._mask_np = binary
        self.accept()

    def get_mask(self):
        return self._mask_np


class OpButton(QPushButton):
    def __init__(self, label: str, tag: str, fn):
        super().__init__()
        self.op_label = label
        self.op_tag   = tag
        self.op_fn    = fn
        self.setObjectName("op_btn")
        self.setCheckable(False)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(36)
        self.setText(f"  {label}   {tag}")
        self.setProperty("active", "false")

    def set_active(self, v: bool):
        self.setProperty("active", "true" if v else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class UpscaleMenuButton(OpButton):
    """Przycisk Upscaling z wysuwanym menu wyboru skali."""
    def __init__(self):
        key, label, tag, fn = _UPSCALE_OPTIONS[1]  # domyślnie x4
        super().__init__(label, tag, fn)
        self._update_display(key, tag)

    def _update_display(self, key: str, tag: str):
        self._current_key = key
        self.setText(f"  Upscaling   {key}   {tag}  ▾")


# ══════════════════════════════════════════════════════════════════════════════
#  GLOWNE OKNO
# ══════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Image Enhancer")
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)

        self._original:  Image.Image | None = None
        self._result:    Image.Image | None = None
        self._worker:    WorkerThread | None = None
        self._active_btn: OpButton | None = None
        self._current_fn   = None
        self._current_name = ""
        self._manual_mask  = None   # numpy uint8 maska watermarku (ręczna)
        self._btn_mark    = None   # inicjalizowane w _make_sidebar

        self._build_ui()

    def _build_ui(self):
        root = QWidget(); root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QHBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._make_sidebar(), 0)
        lay.addWidget(self._make_canvas_area(), 1)

    # ─ Sidebar ───────────────────────────────────────────────────────────────
    def _make_sidebar(self) -> QWidget:
        sidebar = QWidget(); sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(268)
        vbox = QVBoxLayout(sidebar)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Header
        header = QWidget()
        header.setStyleSheet("background: #0d1526; border-bottom: 1px solid #1c2740;")
        hlay = QVBoxLayout(header)
        hlay.setContentsMargins(20, 18, 20, 16)
        hlay.setSpacing(4)
        lbl_title = QLabel("AI Image Enhancer"); lbl_title.setObjectName("lbl_title")
        lbl_sub   = QLabel("RTX 3080  \u00b7  100% OFFLINE"); lbl_sub.setObjectName("lbl_sub")
        hlay.addWidget(lbl_title)
        hlay.addWidget(lbl_sub)
        vbox.addWidget(header)

        # Wczytaj
        lw = QWidget()
        lwl = QVBoxLayout(lw); lwl.setContentsMargins(14, 14, 14, 8)
        self._btn_load = QPushButton("  \u2295   Wczytaj zdjecie")
        self._btn_load.setObjectName("btn_load")
        self._btn_load.setFixedHeight(42)
        self._btn_load.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_load.clicked.connect(self._load_dialog)
        lwl.addWidget(self._btn_load)
        vbox.addWidget(lw)

        div1 = QFrame(); div1.setObjectName("divider"); vbox.addWidget(div1)

        # Operacje (scroll)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        ops_w = QWidget()
        ops_l = QVBoxLayout(ops_w)
        ops_l.setContentsMargins(10, 10, 10, 10)
        ops_l.setSpacing(2)

        self._op_buttons: list[OpButton] = []
        first_btn = None

        for label, tag, fn in OPERATIONS:
            if fn is None:
                sec = QLabel(label); sec.setObjectName("lbl_section")
                sec.setContentsMargins(8, 10, 0, 4)
                ops_l.addWidget(sec)
            elif fn == "_UPSCALE_MENU":
                btn = UpscaleMenuButton()
                btn.clicked.connect(lambda checked, b=btn: self._show_upscale_menu(b))
                self._op_buttons.append(btn)
                ops_l.addWidget(btn)
                if first_btn is None:
                    first_btn = btn
            else:
                btn = OpButton(label, tag, fn)
                btn.clicked.connect(lambda checked, b=btn: self._select_op(b))
                self._op_buttons.append(btn)
                ops_l.addWidget(btn)
                if first_btn is None:
                    first_btn = btn

        ops_l.addStretch()
        scroll.setWidget(ops_w)
        vbox.addWidget(scroll, 1)

        if first_btn:
            self._select_op(first_btn)

        div2 = QFrame(); div2.setObjectName("divider"); vbox.addWidget(div2)

        # Przyciski dolne
        bottom = QWidget()
        bl = QVBoxLayout(bottom); bl.setContentsMargins(14, 12, 14, 16); bl.setSpacing(8)

        self._btn_gen = QPushButton("  \u26a1   Generuj")
        self._btn_gen.setObjectName("btn_generate")
        self._btn_gen.setFixedHeight(46)
        self._btn_gen.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_gen.clicked.connect(self._run_processing)
        bl.addWidget(self._btn_gen)

        self._btn_mark = QPushButton("  \u270f   Zaznacz znak wodny")
        self._btn_mark.setObjectName("btn_mark_wm")
        self._btn_mark.setFixedHeight(38)
        self._btn_mark.setVisible(False)
        self._btn_mark.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_mark.clicked.connect(self._open_mask_painter)
        bl.addWidget(self._btn_mark)

        self._btn_save = QPushButton("  \u2913   Zapisz do Pobranych")
        self._btn_save.setObjectName("btn_save")
        self._btn_save.setFixedHeight(38)
        self._btn_save.setEnabled(False)
        self._btn_save.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._btn_save.clicked.connect(self._save_result)
        bl.addWidget(self._btn_save)

        self._lbl_status = QLabel("Gotowy")
        self._lbl_status.setObjectName("lbl_status")
        self._lbl_status.setWordWrap(True)
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        bl.addWidget(self._lbl_status)

        vbox.addWidget(bottom)
        return sidebar

    # ─ Canvas area ────────────────────────────────────────────────────────────
    def _make_canvas_area(self) -> QWidget:
        area = QWidget(); area.setObjectName("canvas_area")
        lay = QVBoxLayout(area)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._canvas = BeforeAfterWidget()
        lay.addWidget(self._canvas, 1)

        hint_bar = QWidget()
        hint_bar.setStyleSheet("background: #0d1526; border-top: 1px solid #1c2740;")
        hlay = QHBoxLayout(hint_bar)
        hlay.setContentsMargins(16, 6, 16, 6)
        lbl_hint = QLabel("Przeciagnij suwak na zdjecie aby porownac  PRZED / PO")
        lbl_hint.setObjectName("lbl_info")
        hlay.addWidget(lbl_hint, 1)
        self._lbl_dims = QLabel("")
        self._lbl_dims.setObjectName("lbl_info")
        self._lbl_dims.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        hlay.addWidget(self._lbl_dims)
        lay.addWidget(hint_bar, 0)
        return area

    # ── Akcje ─────────────────────────────────────────────────────────────────
    def _show_upscale_menu(self, btn: "UpscaleMenuButton"):
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.setObjectName("upscale_menu")
        for key, label, tag, fn in _UPSCALE_OPTIONS:
            act = menu.addAction(f"{label}   {tag}")
            act.setData((key, label, tag, fn))
        chosen = menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        if chosen:
            key, label, tag, fn = chosen.data()
            btn.op_fn    = fn
            btn.op_label = label
            btn.op_tag   = tag
            btn._update_display(key, tag)
        self._select_op(btn)

    def _select_op(self, btn: OpButton):
        if self._active_btn:
            self._active_btn.set_active(False)
        self._active_btn   = btn
        btn.set_active(True)
        self._current_fn   = btn.op_fn
        self._current_name = btn.op_label
        # Pokaż przycisk do zaznaczania maski tylko dla operacji watermark
        if self._btn_mark is not None:
            is_wm = (btn.op_fn is process_remove_watermark)
            self._btn_mark.setVisible(is_wm)
            if not is_wm:
                self._manual_mask = None
                self._btn_mark.setProperty("masked", False)
                self._btn_mark.style().unpolish(self._btn_mark)
                self._btn_mark.style().polish(self._btn_mark)

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Wybierz zdjecie", "",
            "Wszystkie obrazy ("
            "*.jpg *.jpeg *.jpe *.jfif "
            "*.png *.bmp *.dib *.tiff *.tif "
            "*.webp *.avif *.heic *.heif "
            "*.gif *.ico *.tga *.ppm *.pgm *.pbm *.pcx "
            "*.exr "
            "*.cr2 *.cr3 *.nef *.nrw *.arw *.srf *.sr2 "
            "*.dng *.orf *.rw2 *.pef *.raf *.raw *.3fr "
            "*.mrw *.x3f *.rwl *.srw *.iiq "
            ");;"
            "JPEG (*.jpg *.jpeg *.jpe *.jfif);;"
            "PNG (*.png);;"
            "WebP / AVIF / HEIC (*.webp *.avif *.heic *.heif);;"
            "RAW aparaty (*.cr2 *.cr3 *.nef *.nrw *.arw *.dng *.orf *.rw2 *.raf *.pef *.raw);;"
            "Inne (*.bmp *.tiff *.tif *.gif *.ico *.tga *.ppm *.pcx *.exr);;"
            "Wszystkie pliki (*.*)"
        )
        if path:
            self._load_path(path)

    def _load_path(self, path: str):
        try:
            img = self._open_any(path)
            self._original   = img
            self._result     = None
            self._manual_mask = None
            self._btn_save.setEnabled(False)
            self._btn_mark.setText("  \u270f   Zaznacz znak wodny")
            self._btn_mark.setProperty("masked", False)
            self._btn_mark.style().unpolish(self._btn_mark)
            self._btn_mark.style().polish(self._btn_mark)
            self._canvas.set_before(img)
            w, h = img.size
            self._lbl_status.setText(f"Wczytano: {Path(path).name}")
            self._lbl_dims.setText(f"{w} x {h} px  \u00b7  {Path(path).suffix.upper()[1:]}")
        except Exception as e:
            QMessageBox.critical(self, "Blad", f"Nie mozna otworzyc pliku:\n{e}")

    @staticmethod
    def _open_any(path: str) -> Image.Image:
        """Otwiera dowolny popularny format obrazu. Zawsze koryguje orientację EXIF."""
        import numpy as np, cv2

        ext = Path(path).suffix.lower()

        # ── Formaty RAW z aparatów — rawpy ────────────────────────────────────
        RAW_EXT = {".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2",
                   ".dng", ".orf", ".rw2", ".pef", ".raf", ".raw", ".3fr",
                   ".mrw", ".x3f", ".rwl", ".srw", ".iiq"}
        if ext in RAW_EXT:
            try:
                import rawpy
                with rawpy.imread(path) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        no_auto_bright=False,
                        output_bps=8,
                    )
                return Image.fromarray(rgb)   # rawpy już koryguje orientację
            except ImportError:
                pass  # fallback do cv2

        # ── AVIF / HEIC — pillow-heif (obsługuje irot/imir box) ──────────────────
        if ext in (".avif", ".heic", ".heif"):
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
                img = Image.open(path)
                img.load()
                img = MainWindow._fix_orientation(img)
                return img.convert("RGBA") if img.mode in ("RGBA", "LA", "P") else img.convert("RGB")
            except ImportError:
                pass
            # fallback: pillow-avif-plugin
            try:
                import pillow_avif  # noqa
                img = Image.open(path)
                img.load()
                img = MainWindow._fix_orientation(img)
                return img.convert("RGBA") if img.mode in ("RGBA", "LA") else img.convert("RGB")
            except Exception:
                pass
            raise ValueError(f"Brak pillow-heif lub pillow-avif-plugin. Zainstaluj: pip install pillow-heif")

        # ── EXR — OpenCV (HDR → tone-map do 8-bit) ────────────────────────────
        if ext == ".exr":
            bgr = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Nie można otworzyć: {path}")
            tm = cv2.createTonemap(gamma=2.2)
            bgr8 = np.clip(tm.process(bgr.astype(np.float32)) * 255, 0, 255).astype(np.uint8)
            return Image.fromarray(cv2.cvtColor(bgr8, cv2.COLOR_BGR2RGB))

        # ── GIF — first frame ─────────────────────────────────────────────────
        if ext == ".gif":
            img = Image.open(path)
            img.seek(0)
            return img.convert("RGBA")

        # ── Wszystkie inne — Pillow (JPEG, PNG, BMP, TIFF, WebP, TGA, PCX...)─
        img = Image.open(path)
        img.load()
        return MainWindow._fix_orientation(img)

    @staticmethod
    def _fix_orientation(img: Image.Image) -> Image.Image:
        """Koryguje obrót na podstawie znacznika EXIF Orientation."""
        from PIL import ImageOps
        try:
            return ImageOps.exif_transpose(img)
        except Exception:
            return img

    def _run_processing(self):
        if self._original is None:
            QMessageBox.warning(self, "Brak zdjecia", "Najpierw wczytaj zdjecie.")
            return
        if self._worker and self._worker.isRunning():
            return
        self._btn_gen.setEnabled(False)
        self._btn_save.setEnabled(False)
        self._canvas.show_generating()
        self._lbl_status.setText(f"Przetwarzam: {self._current_name}...")
        # Jeśli operacja watermark i jest ręczna maska — przekaż ją
        fn = self._current_fn
        if fn is process_remove_watermark and self._manual_mask is not None:
            import functools
            mask_copy = self._manual_mask.copy()
            fn = functools.partial(process_remove_watermark, manual_mask=mask_copy)
        self._worker = WorkerThread(fn, self._original)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _open_mask_painter(self):
        if self._original is None:
            QMessageBox.warning(self, "Brak zdjecia", "Najpierw wczytaj zdjecie.")
            return
        dlg = MaskPainterDialog(self._original, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._manual_mask = dlg.get_mask()
            self._btn_mark.setText("  \u2713   Maska zaznaczona \u2014 edytuj")
            self._btn_mark.setProperty("masked", True)
            self._btn_mark.style().unpolish(self._btn_mark)
            self._btn_mark.style().polish(self._btn_mark)
            self._lbl_status.setText("Maska gotowa. Kliknij Generuj aby usunac znak wodny.")

    def _on_error(self, msg: str):
        self._btn_gen.setEnabled(True)
        if self._original:
            self._canvas.set_before(self._original)
        else:
            self._canvas.clear()
        self._lbl_status.setText(f"Blad: {msg}")
        QMessageBox.critical(self, "Blad przetwarzania", msg)

    def _auto_save(self):
        if self._result is None:
            return
        path = self._unique_path()
        self._write_img(path)
        self._lbl_status.setText(f"Gotowe  \u00b7  zapisano: {path.name}")

    def _save_result(self):
        if self._result is None:
            return
        path = self._unique_path()
        self._write_img(path)
        QMessageBox.information(self, "Zapisano", f"Plik zapisany w:\n{path}")

    def _unique_path(self) -> Path:
        slug = self._current_name.lower().replace(" ", "_").replace("x", "x")
        i = 1
        while True:
            p = DOWNLOADS_DIR / f"ai_{slug}_{i}.png"
            if not p.exists():
                return p
            i += 1

    def _write_img(self, path: Path):
        img = self._result
        if img.mode == "RGBA":
            img.save(path, format="PNG")
        else:
            img.convert("RGB").save(path, format="PNG", optimize=True)

    def closeEvent(self, event):
        """Przy zamknięciu okna — wyczyść temp aplikacji."""
        _cleanup_app_temp()
        event.accept()

    def _on_done(self, result: Image.Image):
        self._result = result
        self._canvas.set_after(result)
        self._btn_gen.setEnabled(True)
        self._btn_save.setEnabled(True)
        w, h = result.size
        self._lbl_status.setText(f"Gotowe  \u00b7  wynik: {w} x {h} px  \u2014 kliknij Zapisz")
        self._lbl_dims.setText(f"{w} x {h} px")
        # Wyczyść temp po przetworzeniu — nie gromadź plików między operacjami
        _cleanup_app_temp()
        _APP_TEMP.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  URUCHOMIENIE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)

    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window,          QColor("#0b0f1a"))
    pal.setColor(QPalette.ColorRole.WindowText,      QColor("#f1f5f9"))
    pal.setColor(QPalette.ColorRole.Base,            QColor("#111827"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   QColor("#1f2937"))
    pal.setColor(QPalette.ColorRole.Text,            QColor("#e2e8f0"))
    pal.setColor(QPalette.ColorRole.Button,          QColor("#1e2d4a"))
    pal.setColor(QPalette.ColorRole.ButtonText,      QColor("#f1f5f9"))
    pal.setColor(QPalette.ColorRole.Highlight,       QColor("#3b82f6"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())
