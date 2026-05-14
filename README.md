# AI-Image-Enhancer
Zaawansowana, w 100% działająca offline aplikacja desktopowa do poprawy jakości, upscalingu i edycji obrazów przy użyciu najnowocześniejszych modeli sztucznej inteligencji. Aplikacja została napisana w Pythonie (PyQt6) i wykorzystuje akcelerację GPU (CUDA) do błyskawicznego przetwarzania.

# AI Image Enhancer 🚀

Zaawansowana, w 100% działająca offline aplikacja desktopowa do poprawy jakości, upscalingu i edycji obrazów przy użyciu najnowocześniejszych modeli sztucznej inteligencji. Aplikacja została napisana w Pythonie (PyQt6) i wykorzystuje akcelerację GPU (CUDA) do błyskawicznego przetwarzania.

![AI Image Enhancer GUI](https://via.placeholder.com/800x450.png?text=Wstaw+tutaj+screenshot+z+aplikacji)
*(Zalecane: wstaw tutaj zrzut ekranu pokazujący suwak Przed/Po)*

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

### 1. Klonowanie repozytorium
```bash
git clone [https://github.com/TWOJ_NICK/ai-image-enhancer.git](https://github.com/TWOJ_NICK/ai-image-enhancer.git)
cd ai-image-enhancer
