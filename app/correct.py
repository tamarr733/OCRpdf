import asyncio
import logging
import os
import pathlib
import re
import subprocess
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uvicorn
import cv2
import easyocr
import numpy as np
import ocrmypdf
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from pdf2image import convert_from_path
from PIL import Image, UnidentifiedImageError
from pytesseract import Output
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from scipy.spatial import distance_matrix

logger = logging.getLogger("ocrpdf")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

OCR_JOBS = int(os.getenv("OCR_JOBS", "4"))
EASYOCR_CONFIDENCE_THRESHOLD = float(os.getenv("EASYOCR_CONFIDENCE_THRESHOLD", "0.7"))
TESSERACT_FALLBACK_PSM = os.getenv("TESSERACT_FALLBACK_PSM", "11")
ENABLE_OSD_ROTATION = os.getenv("ENABLE_OSD_ROTATION", "1") == "1"

# NOTE: EasyOCR doesn't support Hebrew well; keep only English for detection.
reader = easyocr.Reader(["en"], gpu=False, quantize=True)

app = FastAPI()


def _safe_basename(name: Optional[str]) -> str:
    """
    Prevent path traversal and weird filenames.
    """
    if not name:
        return "upload"
    name = pathlib.Path(name).name  # strips directories
    # Keep simple chars only; avoid spaces/UTF surprises for downstream tools
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "upload"


def _suffix_from_filename(name: Optional[str]) -> str:
    base = _safe_basename(name)
    suf = pathlib.Path(base).suffix
    return suf if suf else ".bin"


def convert_image_to_pdf(input_path: str, output_pdf_path: str) -> bool:
    """
    Convert an image file to PDF. Returns True if input is a valid image.
    """
    try:
        with Image.open(input_path) as img:
            rgb_img = img.convert("RGB")  # avoid alpha issues
            rgb_img.save(output_pdf_path, "PDF", resolution=300)
            return True
    except (OSError, UnidentifiedImageError):
        return False


def repair_pdf(input_path: str, output_path: str) -> bool:
    """
    Attempt to repair broken PDF using Ghostscript.
    """
    try:
        subprocess.run(
            [
                "gs",
                "-o",
                output_path,
                "-sDEVICE=pdfwrite",
                "-dColorConversionStrategy=/RGB",
                "-dProcessColorModel=/DeviceRGB",
                "-dCompatibilityLevel=1.4",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                input_path,
            ],
            check=True,
            capture_output=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else "Unknown error"
        logger.warning("Ghostscript repair failed: %s", stderr)
        return False


def compress_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """
    Compress PDF bytes using Ghostscript (stdin -> stdout) and normalize pages to A4.
    """
    try:
        process = subprocess.Popen(
            [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/ebook",
                "-sPAPERSIZE=a4",
                "-dFIXEDMEDIA",
                "-dPDFFitPage",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                "-sOutputFile=-",
                "-",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        compressed_pdf, stderr = process.communicate(input=pdf_bytes)
        if process.returncode == 0 and compressed_pdf:
            logger.info("Compressed PDF %d -> %d bytes", len(pdf_bytes), len(compressed_pdf))
            return compressed_pdf

        logger.warning("Ghostscript compression warning: %s", stderr.decode(errors="ignore"))
        return pdf_bytes
    except Exception as e:
        logger.warning("Compression failed: %s", e)
        return pdf_bytes


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def fix_image_rotation_osd(image_bgr: np.ndarray) -> np.ndarray:
    """
    Rotation fix via Tesseract OSD.
    """
    if not ENABLE_OSD_ROTATION:
        return image_bgr
    try:
        # Tesseract works better with RGB for OSD
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        osd = pytesseract.image_to_osd(rgb)
        m = re.search(r"Rotate:\s+(\d+)", osd)
        if not m:
            return image_bgr
        angle = int(m.group(1))
        if angle == 0:
            return image_bgr
        logger.info("Detected rotation %d degrees; rotating", angle)
        if angle == 90:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        if angle == 180:
            return cv2.rotate(image_bgr, cv2.ROTATE_180)
        if angle == 270:
            return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image_bgr
    except Exception as e:
        logger.info("Rotation detection failed: %s", e)
        return image_bgr


def _preprocess_for_easyocr(image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (enhanced_gray, image_rgb_for_background).
    """
    image_bgr = fix_image_rotation_osd(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape[:2]
    if w > 0 and w < 1000:
        scale = 1000 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return enhanced, image_rgb


def _read_input_as_pages(file_path: str, dpi: int = 300) -> List[np.ndarray]:
    """
    Returns a list of pages as BGR images.
    """
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=dpi)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
    img = cv2.imread(file_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported file or failed to read image.")
    return [img]


def _run_tesseract_on_crop(img_gray: np.ndarray, config: str) -> str:
    # Important: this runs a separate Tesseract process per call
    return pytesseract.image_to_string(img_gray, lang="heb+eng", config=config)


def _extract_boxes_with_tesseract_fallback(
    enhanced_gray: np.ndarray,
    easyocr_results: Iterable[Tuple[Any, str, float]],
    max_workers: int,
) -> List[Dict[str, Any]]:
    """
    For low-confidence EasyOCR regions, re-run Tesseract on the crop.
    """
    h, w = enhanced_gray.shape[:2]
    tasks: List[np.ndarray] = []
    indices: List[int] = []

    for i, (bbox, _text, conf) in enumerate(easyocr_results):
        if conf >= EASYOCR_CONFIDENCE_THRESHOLD:
            continue
        bbox_array = np.array(bbox)
        x_min = int(bbox_array[:, 0].min())
        y_min = int(bbox_array[:, 1].min())
        x_max = int(bbox_array[:, 0].max())
        y_max = int(bbox_array[:, 1].max())

        x_min = _clamp(x_min, 0, w)
        x_max = _clamp(x_max, 0, w)
        y_min = _clamp(y_min, 0, h)
        y_max = _clamp(y_max, 0, h)
        if x_max <= x_min or y_max <= y_min:
            continue

        crop = enhanced_gray[y_min:y_max, x_min:x_max]
        tasks.append(crop)
        indices.append(i)

    refined: Dict[int, str] = {}
    if tasks:
        config = f"--oem 3 --psm {TESSERACT_FALLBACK_PSM}"
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            outputs = list(executor.map(lambda im: _run_tesseract_on_crop(im, config), tasks))
        refined = dict(zip(indices, outputs))

    boxes: List[Dict[str, Any]] = []
    for i, (bbox, text, conf) in enumerate(easyocr_results):
        bbox_array = np.array(bbox)
        x_min = int(bbox_array[:, 0].min())
        y_min = int(bbox_array[:, 1].min())
        x_max = int(bbox_array[:, 0].max())
        y_max = int(bbox_array[:, 1].max())
        final_text = refined.get(i, text) or ""
        final_text = final_text.strip()
        boxes.append(
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "width": max(0, x_max - x_min),
                "height": max(0, y_max - y_min),
                "text": final_text,
                "confidence": float(conf),
            }
        )
    return boxes


def _register_font() -> str:
    """
    Prefer Arial if available; fallback to Helvetica.
    """
    arial_candidates = [
        # Dockerfile copies to this location
        "/usr/share/fonts/truetype/arial.ttf",
        "/usr/share/fonts/arial.ttf",
        # Local working dir copy
        str(pathlib.Path(__file__).with_name("arial.ttf")),
    ]
    for p in arial_candidates:
        try:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont("Arial", p))
                return "Arial"
        except Exception:
            continue
    return "Helvetica"


def create_pdf_multi_page_in_memory(
    pages_rgb: List[np.ndarray],
    boxes_by_page: List[List[Dict[str, Any]]],
    quality: int = 60,
    max_width: int = 1600,
) -> bytes:
    """
    Multi-page searchable PDF with A4 pages: page image background + invisible text overlay.
    """
    if len(pages_rgb) != len(boxes_by_page):
        raise ValueError("pages_rgb and boxes_by_page must have same length")

    font_name = _register_font()
    buffer = tempfile.SpooledTemporaryFile(max_size=20_000_000)
    c = None

    try:
        for page_idx, (image_rgb, text_boxes) in enumerate(zip(pages_rgb, boxes_by_page)):
            h_orig, w_orig, _ = image_rgb.shape
            page_width, page_height = A4

            # Uniform scale to fit original image inside A4 while preserving aspect ratio
            scale = min(page_width / float(w_orig or 1), page_height / float(h_orig or 1))
            draw_w = w_orig * scale
            draw_h = h_orig * scale

            # Center image in A4 page
            offset_x = (page_width - draw_w) / 2.0
            offset_y = (page_height - draw_h) / 2.0

            if c is None:
                c = canvas.Canvas(buffer, pagesize=A4)
            else:
                c.setPageSize(A4)

            # Background image: optional downscale before JPEG encode
            if w_orig > max_width:
                scale_factor = max_width / w_orig
                new_w = int(w_orig * scale_factor)
                new_h = int(h_orig * scale_factor)
                img_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                img_resized = image_rgb

            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
            is_success, buffer_img = cv2.imencode(
                ".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            )
            if is_success:
                from reportlab.lib.utils import ImageReader

                img_stream = tempfile.SpooledTemporaryFile(max_size=5_000_000)
                img_stream.write(buffer_img.tobytes())
                img_stream.seek(0)
                # Draw scaled image centered on A4
                c.drawImage(
                    ImageReader(img_stream),
                    offset_x,
                    offset_y,
                    width=draw_w,
                    height=draw_h,
                )

            # Invisible text overlay
            for box in text_boxes:
                text = (box.get("text") or "").strip()
                if not text:
                    continue

                x_min = float(box.get("x_min", 0))
                y_max = float(box.get("y_max", 0))
                box_width = float(box.get("width", 0))
                box_height = float(box.get("height", 0))

                # Map original image coordinates into A4 coordinates
                x = offset_x + x_min * scale
                # Original baseline y = h_orig - y_max; scale and shift to A4
                y = offset_y + (h_orig - y_max) * scale
                box_width *= scale
                box_height *= scale

                font_size = max(8.0, box_height * 0.8)
                text_object = c.beginText()
                text_object.setFont(font_name, font_size)
                text_object.setTextRenderMode(3)  # invisible

                # Simple width stretching
                text_width = c.stringWidth(text, font_name, font_size)
                scale_factor_text = 1.0
                if text_width > 0 and box_width > 0:
                    stretch_percent = (box_width / text_width) * 100.0
                    if 50.0 < stretch_percent < 200.0:
                        text_object.setHorizScale(stretch_percent)
                        scale_factor_text = stretch_percent / 100.0

                is_hebrew = any("\u0590" <= ch <= "\u05EA" for ch in text)
                if is_hebrew:
                    # Draw RTL by placing glyphs right-to-left.
                    cursor_x = x + box_width
                    for ch in text:
                        ch_w = c.stringWidth(ch, font_name, font_size) * scale_factor_text
                        cursor_x -= ch_w
                        text_object.setTextOrigin(cursor_x, y)
                        text_object.textOut(ch)
                else:
                    text_object.setTextOrigin(x, y)
                    text_object.textOut(text)

                c.drawText(text_object)

            if page_idx < len(pages_rgb) - 1:
                c.showPage()

        if c is None:
            raise ValueError("No pages to write")
        c.save()

        buffer.seek(0)
        return buffer.read()
    finally:
        try:
            buffer.close()
        except Exception:
            pass


def _run_ocrmypdf(
    source_file: str,
    out_file: str,
    lang: str = "heb+eng",
    psm: int = 3,
) -> None:
  
    sidecar_path = out_file + ".txt"
    ocrmypdf.ocr(
        source_file,
        out_file,
        image_dpi=300,
        clean=True,
        clean_final=True,
        deskew=True,
        rotate_pages=True,
        tesseract_pagesegmode=int(psm),
        # sidecar=sidecar_path,
        unpaper_args="--no-blackfilter --no-grayfilter --no-border --no-deskew",
        language=lang,
        force_ocr=True,
        jobs=OCR_JOBS,
        optimize=3,
        progress_bar=False,
    )


def get_first_page_image(file_path: str, dpi: int = 150) -> np.ndarray:
    """
    Return the first page of the file as a BGR image.
    """
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=dpi, first_page=1, last_page=1)
        image = np.array(pages[0])
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img = cv2.imread(file_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Unsupported file or failed to read image.")
    return img


def _analyze_tesseract_data(gray_image: np.ndarray, config: str) -> Dict[str, Any]:
    """
    Helper function to run image_to_data and calculate stats.
    """
    h_img, w_img = gray_image.shape[:2]
    image_area = h_img * w_img
    
    data = pytesseract.image_to_data(
        gray_image, lang="heb+eng", config=config, output_type=Output.DICT
    )

    boxes: List[Dict[str, int]] = []
    valid_confs: List[int] = []

    for i in range(len(data["text"])):
        try:
            conf = int(data["conf"][i])
        except ValueError:
            continue
        # Filter empty text
        if conf > 0 and data["text"][i].strip():
            boxes.append({
                "x": data["left"][i],
                "y": data["top"][i],
                "w": data["width"][i],
                "h": data["height"][i],
            })
            valid_confs.append(conf)

    if not boxes:
        return {
            "box_count": 0,
            "coverage_ratio": 0.0,
            "avg_nearest_dist": 0.0,
            "image_width": float(w_img),
            "avg_confidence": 0.0,
        }

    total_text_area = sum(b["w"] * b["h"] for b in boxes)
    coverage_ratio = total_text_area / float(image_area or 1)
    avg_confidence = float(np.mean(valid_confs)) if valid_confs else 0.0

    # Calculate distance dispersion
    centers = np.array([[b["x"] + b["w"] / 2, b["y"] + b["h"] / 2] for b in boxes])
    avg_nearest_dist = 0.0
    if len(centers) > 1:
        dist_mat = distance_matrix(centers, centers)
        np.fill_diagonal(dist_mat, np.inf)
        avg_nearest_dist = float(np.mean(np.min(dist_mat, axis=1)))

    return {
        "box_count": float(len(boxes)),
        "coverage_ratio": float(coverage_ratio),
        "avg_nearest_dist": float(avg_nearest_dist),
        "image_width": float(w_img),
        "avg_confidence": avg_confidence,
    }


def profile_image(image_bgr: np.ndarray) -> Dict[str, float]:
    """
    Use Tesseract to estimate density. 
    Smart Logic: Tries PSM 3 first. If confidence is low, tries PSM 11.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. ניסיון ראשון: מצב רגיל (PSM 3)
    stats = _analyze_tesseract_data(gray, config="--psm 3")
    
    logger.info("Standard profiling returned (%s). ", stats)

    # 2. בדיקה: האם התוצאה גרועה מאוד?
    # אם הביטחון נמוך מ-40 או שמצאנו פחות מ-5 תיבות טקסט -> אולי זה טקסט מפוזר?
    if stats["avg_confidence"] < 40 or stats["box_count"] < 5:
        logger.info("Standard profiling returned low confidence (%.2f). Trying PSM 11...", stats["avg_confidence"])
        
        # הרצה נוספת עם PSM 11 (Sparse Text)
        stats_sparse = _analyze_tesseract_data(gray, config="--psm 11")
        
        # 3. החלטה: האם PSM 11 שיפר את המצב?
        # אם הביטחון ב-PSM 11 גבוה משמעותית (למשל מעל 60) ויש טקסט -> זה כנראה מסמך מפוזר
        if stats_sparse["avg_confidence"] > 60 and stats_sparse["box_count"] > 5:
            logger.info("PSM 11 check passed with confidence %.2f. Using sparse profile.", stats_sparse["avg_confidence"])
            
            # אנחנו מחזירים את הסטטיסטיקה של PSM 11.
            # זה יגרום ל-avg_confidence להיות גבוה, ולכן זה *לא* ייפול ל-EasyOCR ב-choose_engine.
            # בנוסף, מכיוון שזה טקסט מפוזר, ה-avg_nearest_dist כנראה יהיה גבוה, 
            # מה שיגרום ל-choose_engine לבחור בנתיב ה-PSM 11 הקיים שלך.
            return stats_sparse

    return stats

def choose_engine(profile: Dict[str, float]) -> Dict[str, Any]:
    # 1. Very weak / few boxes -> EasyOCR
    logger.info("Choosing engine: %s", profile)
    if profile["avg_confidence"] < 40 or profile["box_count"] < 5:
        return {"engine": "easyocr", "psm": None}

    # 2. זיהוי PSM 11 (טקסט מפוזר)
    # עדכון: אם יחס הכיסוי נמוך (מעט דיו על הדף) אבל המרחק בין מילים גדול
    if profile["coverage_ratio"] < 0.10 and profile["avg_nearest_dist"] > (profile["image_width"] * 0.05):
         return {"engine": "ocrmypdf", "psm": 11}

    # 3. Dense, multi-block documents
    if profile["box_count"] > 50:
        return {"engine": "ocrmypdf", "psm": 3}

    # 4. Short / single-block documents
    return {"engine": "ocrmypdf", "psm": 6}


def compute_ocr_score(ocr_data: Dict[str, List[Any]]) -> float:
    """
    Compute a quality score from Tesseract image_to_data output.
    """
    texts: List[str] = []
    confs: List[int] = []

    for i in range(len(ocr_data["text"])):
        txt = str(ocr_data["text"][i]).strip()
        try:
            conf = int(ocr_data["conf"][i])
        except ValueError:
            continue
        if txt and conf > -1:
            texts.append(txt)
            confs.append(conf)

    if not texts:
        return 0.0

    text_len = sum(len(t) for t in texts)
    valid_chars = sum(len(re.findall(r"[A-Za-z\u0590-\u05EA]", t)) for t in texts)
    valid_ratio = valid_chars / float(text_len or 1)
    avg_conf = (np.mean(confs) / 100.0) if confs else 0.0

    short_words = sum(1 for t in texts if len(t) == 1)
    consistency = 1.0 - (short_words / float(len(texts) or 1))

    score = 0.5 * avg_conf + 0.3 * valid_ratio + 0.2 * consistency
    return float(round(score, 3))


def adaptive_ocr(image_bgr: np.ndarray) -> Dict[str, Any]:
    """
    Decide engine + PSM, and if Tesseract quality is low, fall back to EasyOCR.
    """
    prof = profile_image(image_bgr)
    decision = choose_engine(prof)

    result: Dict[str, Any] = {
        "engine": decision["engine"],
        "psm": decision["psm"],
        "fallback": False,
        "confidence_score": 0.0,
    }

    if decision["engine"] == "ocrmypdf":
        data = pytesseract.image_to_data(
            image_bgr,
            lang="heb+eng",
            config=f"--psm {decision['psm']}",
            output_type=Output.DICT,
        )
        score = compute_ocr_score(data)
        result["confidence_score"] = score

        if score < 0.75:
            # Fallback to EasyOCR
            result["engine"] = "easyocr"
            result["fallback"] = True
        else:
            return result

    # EasyOCR path / fallback default
    result["engine"] = "easyocr"
    if result["confidence_score"] == 0.0:
        result["confidence_score"] = 0.55
    return result


@app.post("/ocr/ocrmypdf-overlay")
async def ocr_pdf_overlay(file: UploadFile = File(...), psm: int = 3) -> Response:
    """
    OCRmyPDF overlay (supports PDFs and images).
    """
    safe_name = _safe_basename(file.filename)
    suffix = _suffix_from_filename(file.filename)

    with tempfile.TemporaryDirectory(prefix="ocrpdf_") as td:
        in_path = os.path.join(td, f"{uuid.uuid4().hex}{suffix}")
        fixed_path = os.path.join(td, f"{uuid.uuid4().hex}_fixed.pdf")
        out_path = os.path.join(td, f"{uuid.uuid4().hex}_out.pdf")

        content = await file.read()
        with open(in_path, "wb") as f:
            f.write(content)

        # If image -> convert. Else try repair but still allow original.
        source_file = in_path
        if convert_image_to_pdf(in_path, fixed_path):
            source_file = fixed_path
        else:
            if repair_pdf(in_path, fixed_path):
                source_file = fixed_path

        try:
            # _run_ocrmypdf(source_file, out_path, "heb+eng", psm)
            await asyncio.to_thread(_run_ocrmypdf, source_file, out_path, "heb+eng", psm)
        except Exception as e:
            logger.exception("OCRmyPDF failed")
            raise HTTPException(status_code=500, detail=f"OCR Process Failed: {e}")

        with open(out_path, "rb") as f:
            pdf_bytes = f.read()

        # pdf_bytes = compress_pdf_bytes(pdf_bytes)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={safe_name}_ocr.pdf"},
        )


@app.post("/ocr/easyocrpdf-overlay")
async def easyocrpdf_overlay(file: UploadFile = File(...)) -> Response:
    """
    EasyOCR (detection) + Tesseract (Hebrew text) overlay.
    Supports multi-page PDFs.
    """
    safe_name = _safe_basename(file.filename)
    suffix = _suffix_from_filename(file.filename)
    max_workers = max(1, min(4, (os.cpu_count() or 2)))

    with tempfile.TemporaryDirectory(prefix="ocrpdf_") as td:
        in_path = os.path.join(td, f"{uuid.uuid4().hex}{suffix}")
        with open(in_path, "wb") as f:
            f.write(await file.read())

        pages_bgr = await asyncio.to_thread(_read_input_as_pages, in_path, 300)

        pages_rgb: List[np.ndarray] = []
        boxes_by_page: List[List[Dict[str, Any]]] = []

        for page_bgr in pages_bgr:
            enhanced, page_rgb = await asyncio.to_thread(_preprocess_for_easyocr, page_bgr)
            pages_rgb.append(page_rgb)

            easy_results = await asyncio.to_thread(reader.readtext, enhanced)
            boxes = await asyncio.to_thread(
                _extract_boxes_with_tesseract_fallback, enhanced, easy_results, max_workers
            )
            boxes_by_page.append(boxes)

        pdf_bytes = await asyncio.to_thread(create_pdf_multi_page_in_memory, pages_rgb, boxes_by_page)
        pdf_bytes = compress_pdf_bytes(pdf_bytes)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={safe_name}_easyocr.pdf"},
        )


@app.post("/ocr/adaptive")
async def adaptive_ocr_endpoint(file: UploadFile = File(...)) -> Response:
    """
    Smart pipeline: choose between OCRmyPDF and EasyOCR+Tesseract based on page profile.
    Always processes the full document; decision is based on first page.
    """
    safe_name = _safe_basename(file.filename)
    suffix = _suffix_from_filename(file.filename)

    with tempfile.TemporaryDirectory(prefix="ocrpdf_") as td:
        in_path = os.path.join(td, f"{uuid.uuid4().hex}{suffix}")
        with open(in_path, "wb") as f:
            f.write(await file.read())

        # Decision step on first page only
        preview_bgr = await asyncio.to_thread(get_first_page_image, in_path)
        decision = await asyncio.to_thread(adaptive_ocr, preview_bgr)
        
        logger.info("Decision: %s", decision)

        # OCRmyPDF path
        if decision["engine"] == "ocrmypdf":
            out_path = os.path.join(td, f"{uuid.uuid4().hex}_out.pdf")
            fixed_path = os.path.join(td, f"{uuid.uuid4().hex}_fixed.pdf")
            # Same as ocr_pdf_overlay: convert images (e.g. PNG with alpha) to PDF so OCRmyPDF accepts them
            source_file = in_path
            if convert_image_to_pdf(in_path, fixed_path):
                source_file = fixed_path
            else:
                if repair_pdf(in_path, fixed_path):
                    source_file = fixed_path
            try:
                await asyncio.to_thread(
                    _run_ocrmypdf,
                    source_file,
                    out_path,
                    "heb+eng",
                    int(decision["psm"]) if decision["psm"] is not None else 3,
                )
            except Exception as e:
                logger.exception("Adaptive OCRmyPDF failed")
                raise HTTPException(status_code=500, detail=f"OCRmyPDF failed: {e}")

            with open(out_path, "rb") as f:
                pdf_bytes = f.read()

            pdf_bytes = compress_pdf_bytes(pdf_bytes)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={safe_name}_ocr.pdf",
                    "X-OCR-Engine": "ocrmypdf",
                    "X-OCR-PSM": str(decision["psm"]),
                    "X-OCR-Score": str(decision.get("confidence_score", 0.0)),
                },
            )

        # EasyOCR multi-page path
        max_workers = max(1, min(4, (os.cpu_count() or 2)))
        pages_bgr = await asyncio.to_thread(_read_input_as_pages, in_path, 300)

        pages_rgb: List[np.ndarray] = []
        boxes_by_page: List[List[Dict[str, Any]]] = []

        for page_bgr in pages_bgr:
            enhanced, page_rgb = await asyncio.to_thread(_preprocess_for_easyocr, page_bgr)
            pages_rgb.append(page_rgb)

            easy_results = await asyncio.to_thread(reader.readtext, enhanced)
            boxes = await asyncio.to_thread(
                _extract_boxes_with_tesseract_fallback, enhanced, easy_results, max_workers
            )
            boxes_by_page.append(boxes)

        pdf_bytes = await asyncio.to_thread(create_pdf_multi_page_in_memory, pages_rgb, boxes_by_page)
        pdf_bytes = compress_pdf_bytes(pdf_bytes)

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={safe_name}_adaptive_easyocr.pdf",
                "X-OCR-Engine": "easyocr",
                "X-OCR-Decision-Mode": "adaptive",
            },
        )


if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

