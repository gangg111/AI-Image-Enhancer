# AI Image Enhancer 🚀

Zaawansowana, w 100% działająca offline aplikacja desktopowa do poprawy jakości, upscalingu i edycji obrazów przy użyciu najnowocześniejszych modeli sztucznej inteligencji. Aplikacja została napisana w Pythonie (PyQt6) i wykorzystuje akcelerację GPU (CUDA) do błyskawicznego przetwarzania.

![AI Image Enhancer GUI](https://github.com/gangg111/AI-Image-Enhancer/blob/main/screenshot.png)

## ✨ Główne funkcje

Aplikacja łączy w sobie potęgę wielu dedykowanych modeli sieci neuronowych, oferując przyjazny interfejs z interaktywnym suwakiem porównawczym (Przed / Po) oraz obsługą przeciągnij-i-upuść (Drag & Drop).

* **🔍 Upscaling (Real-ESRGAN / HAT-L)** – Zwiększanie rozdzielczości x2, x4 i x8 z zachowaniem detali. Dostępny również dedykowany tryb dla zrzutów ekranu i UI, minimalizujący artefakty.
* **✂️ Usuwanie tła (BiRefNet)** – Ekstremalnie precyzyjne wycinanie głównego obiektu z tła.
* **💧 Usuwanie znaków wodnych (LaMa)** – Automatyczna detekcja znaków wodnych lub manualne zaznaczanie niechcianych elementów za pomocą wbudowanego narzędzia pędzla.
* **👁️ Restauracja twarzy (GFPGAN)** – Odtwarzanie i wyostrzanie rozmytych twarzy na starych lub słabej jakości zdjęciach.
* **🎨 Koloryzacja (DDColor)** – Nadawanie naturalnych kolorów czarno-białym fotografiom.
* **⚡ Wyostrzanie i Denoise** – Wielopoziomowe odzyskiwanie detali (OpenCV USM) oraz usuwanie szumu przy jednoczesnym zachowaniu ostrych krawędzi (Bilateral Filter).
* **📷 Szeroka obsługa formatów** – Obsługa standardowych plików (JPG, PNG, WebP) oraz plików RAW z aparatów (CR2, NEF, ARW) i formatów nowoczesnych (HEIC, AVIF, EXR).

## ⚙️ Wymagania i instalacja

Aplikacja została zoptymalizowana do działania na kartach graficznych NVIDIA (np. RTX 3080) przy użyciu środowiska **CUDA**.

## 🙏 Podziękowania / Acknowledgments

Ten projekt opiera się na wybitnych osiągnięciach społeczności open-source oraz badaczy AI. Ogromne podziękowania dla twórców poniższych technologii i modeli, bez których ta aplikacja nie mogłaby powstać:

* **[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) & [GFPGAN](https://github.com/TencentARC/GFPGAN)** (Xintao Wang / Tencent ARC) – za niesamowite algorytmy do upscalingu i bezbłędnej restauracji twarzy.
* **[BiRefNet](https://github.com/ZhengxiaZou/BiRefNet)** (Zhengxia Zou) – za wybitny, potężny model do precyzyjnego wycinania tła (oraz dla twórców pakietu `rembg`).
* **[LaMa (Resolution-robust Large Mask Inpainting)](https://github.com/advimman/lama)** – za świetny model do inpaintingu, który stanowi rdzeń systemu usuwania znaków wodnych (via `simple_lama_inpainting`).
* **[HAT (Hybrid Attention Transformer)](https://github.com/XPixelGroup/HAT)** – za dostarczenie architektur do najwyższej jakości upscalingu SRx4.
* **[DDColor](https://github.com/piddnad/DDColor)** – za wspaniałe algorytmy do realistycznej koloryzacji historycznych i czarno-białych zdjęć.
* Twórcom i rozwijającym potężne biblioteki stanowiące fundament projektu: **[PyTorch](https://pytorch.org/)**, **[PyQt6](https://riverbankcomputing.com/software/pyqt/)**, **[OpenCV](https://opencv.org/)** oraz **[Pillow](https://python-pillow.org/)**.
