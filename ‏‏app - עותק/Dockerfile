FROM python:3.11-slim

# Install system dependencies for pdf2image and tesseract
# RUN apt-get update && \
#     -qO- https://github.com/tesseract-ocr/tesseract/archive/4.1.1.tar.gz | tar -xzf - && \
#     apt-get install -y tesseract-ocr tesseract-ocr-heb libtesseract-dev poppler-utils ghostscript unpaper qpdf && \
#     wget \
#     rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    tesseract-ocr \
    tesseract-ocr-heb \
    libtesseract-dev \
    poppler-utils \
    ghostscript \
    unpaper \
    qpdf \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    imagemagick \
    jbig2 \
    pngquant \
    && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y \
#     tesseract-ocr \
#     libtesseract-dev \
#     tesseract-ocr-heb \
#     tesseract-ocr-eng \
#     ghostscript \
#     poppler-utils \
#     imagemagick \
#     jbig2 \
#     pngquant \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY heb.traineddata /usr/share/tesseract-ocr/5/tessdata/heb.traineddata

COPY arial.ttf /usr/share/fonts/truetype/arial.ttf

# ==========================================

# יצירת התיקייה שבה EasyOCR מחפש את המודלים כברירת מחדל
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"


COPY main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]



