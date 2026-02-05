import datetime
import profile
import shutil
from bidi import get_display
import easyocr
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import ocrmypdf
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
from pdf2image import convert_from_path
import pytesseract
from scipy import io
from shapely import buffer
import uvicorn
import subprocess
from PIL import Image  # Required for image conversion
import cv2
import numpy as np
from reportlab.pdfgen import canvas
from reportlab.lib.colors import black
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import pathlib
import re
from pytesseract import Output


reader = easyocr.Reader(['en'], gpu=False, quantize=True)
app = FastAPI()


def convert_image_to_pdf(input_path, output_path):
    """
    מנסה להמיר תמונה ל-PDF.
    מחזיר True אם הצליח, False אם הקובץ אינו תמונה תקינה.
    """
    try:
        # מנסים לפתוח כדי לראות אם זו תמונה
        with Image.open(input_path) as img:
            print(f"Image detected: {img.format}, mode: {img.mode}")
            
            # המרה ל-RGB (למניעת שגיאות שקיפות)
            rgb_img = img.convert('RGB')
            
            # שמירה כ-PDF
            rgb_img.save(output_path, "PDF", resolution=300)
            return True
            
    except (IOError, Image.UnidentifiedImageError):
        # במקרה שזו לא תמונה או ש-PIL לא הצליח לזהות
        return False


def repair_pdf(input_path, output_path):
    """
    תיקון PDF שבור באמצעות Ghostscript
    """
    try:
        subprocess.run(
            [
                "gs",
                "-o", output_path,
                "-sDEVICE=pdfwrite",
                "-dColorConversionStrategy=/RGB",
                "-dProcessColorModel=/DeviceRGB",
                "-dCompatibilityLevel=1.4",
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                input_path
            ],
            check=True,
            capture_output=True # כדי לא ללכלך את הלוגים אם נכשל
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ghostscript failed: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return False



def fix_image_rotation(image):
    try:
        # שימוש ב-OSD של Tesseract לזיהוי מהיר של כיוון
        osd = pytesseract.image_to_osd(image)
        
        # חיפוש הזווית בתוך הפלט (למשל: "Rotate: 90")
        rotation_match = re.search(r'Rotate: (\d+)', osd)
        
        if rotation_match:
            angle = int(rotation_match.group(1))
            
            if angle == 0:
                return image # התמונה כבר ישרה
            
            print(f"Detected rotation of {angle} degrees. Fixing...")
            
            # סיבוב התמונה בהתאם לזווית שנמצאה
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                
            return image
            
    except Exception as e:
        # במקרה של כישלון (למשל תמונה ללא טקסט ברור), נחזיר את המקור
        print(f"Rotation detection failed: {e}")
        return image
    
    return image

@app.post("/ocr/ocrmypdf-overlay")
async def ocr_pdf_overlay(file: UploadFile = File(...), psm: int = 3):
    tmp_in_path = None
    tmp_fixed_path = None
    tmp_out_path = None
    
    try:
        # 1. חילוץ הסיומת המקורית (חשוב ל-PIL!)
        original_ext = pathlib.Path(file.filename).suffix
        if not original_ext:
            original_ext = ".tmp"

        # 2. שמירת הקובץ וסגירה מיידית (חשוב!)
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp_in:
            content = await file.read()
            tmp_in.write(content)
            tmp_in.flush() # וידוא כתיבה לדיסק
            tmp_in_path = tmp_in.name
            # הקובץ נסגר אוטומטית ביציאה מה-block הזה
            
        print(f"File saved to: {tmp_in_path}")

        # נתיבים לקבצי ביניים
        tmp_fixed_path = tmp_in_path + "_fixed.pdf"
        tmp_out_path = tmp_in_path + "_out.pdf"
        
        # 3. בדיקה האם זו תמונה והמרה
        is_image = False # convert_image_to_pdf(tmp_in_path, tmp_fixed_path)
        
        source_file = None

        if is_image:
            print("File was identified as an image and converted to PDF.")
            source_file = tmp_fixed_path
        else:
            print("File is not an image. Attempting PDF repair...")
            # רק אם זו לא תמונה, ננסה לתקן כ-PDF
            # זה מונע את קריסת Ghostscript על קבצי PNG
            is_repaired = False # repair_pdf(tmp_in_path, tmp_fixed_path)
            
            if is_repaired:
                source_file = tmp_fixed_path
            else:
                # אם התיקון נכשל, נשתמש במקור (אולי הוא PDF תקין ו-GS סתם נכשל)
                print("Repair failed or unnecessary, using original file.")
                source_file = tmp_in_path

        # 4. הרצת OCRmyPDF
        print(f"Starting OCR on {source_file}")
        sidecar_path = tmp_out_path + ".txt"
        ocrmypdf.ocr(
            source_file, 
            tmp_out_path, 
            image_dpi=300,
            clean=True,       
            clean_final=True, 
            deskew=True,       
            rotate_pages=True, 
            tesseract_pagesegmode=11,
            unpaper_args="--no-blackfilter --no-grayfilter --no-border --no-deskew",
            language="heb+eng",
            force_ocr=True,  
            jobs=4,            
            sidecar=sidecar_path,    
        )
        # ocrmypdf.ocr(
        #     tmp_in_path, 
        #     tmp_out_path, 
        #     image_dpi=300,
        #     clean=True,       
        #     clean_final=True, 
        #     deskew=True,       
        #     rotate_pages=True, 
        #     tesseract_pagesegmode=psm,
        #     unpaper_args="--no-blackfilter --no-grayfilter --no-border --no-deskew",
        #     lang="heb+eng",
        #     pdf_renderer='hocr',
        #     jobs=4,   
        #     sidecar=sidecar_path,   
        #     force_ocr=True,
        #     skip_text=False,
        #     optimize=1,
        #     progress_bar=False      
        # )

        with open(tmp_out_path, "rb") as f:
            pdf_bytes = f.read()
        if os.path.exists(sidecar_path):
            print("Sidecar exists. First few lines:")
            with open(sidecar_path, 'r', encoding='utf-8') as sc:
                print('\n'.join(sc.readlines()[:10]))
        # pdf_bytes = compress_pdf_bytes(pdf_bytes)
        return Response(
            content=pdf_bytes, 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename={file.filename}_ocr.pdf"}
        )

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc() # הדפסת שגיאה מלאה ללוג
        raise HTTPException(status_code=500, detail=f"OCR Process Failed: {str(e)}")
        
    finally:
        # ניקוי
        for path in [tmp_in_path, tmp_fixed_path, tmp_out_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except:
                    pass

async def get_file_path(file: UploadFile):
    # יצירת תיקייה זמנית אם לא קיימת
    upload_dir = "/tmp/ocr_uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # בניית הנתיב המלא לקובץ
    file_location = f"{upload_dir}/{file.filename}"
    
    # שמירת הקובץ מהזיכרון לדיסק
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return file_location

@app.post("/ocr/easyocrpdf-overlay")
async def easyocrpdf_overlay(file: UploadFile = File(...)):
    file_path = await get_file_path(file)
    return await ocr_easyocr_pdf_overlay(file_path)

async def ocr_easyocr_pdf_overlay(file_path: str):
    # file_path = await get_file_path(file)
    pages = None
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path, 300)
    if pages:
        image = np.array(pages[0])
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(file_path)
    
    image = fix_image_rotation(image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"Image shape: {image_rgb.shape}")

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Only upscale if needed
    height, width = gray.shape
    if width < 1000:
        scale = 1000 / width  
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image_rgb = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    print(f"Preprocessed image shape: {enhanced.shape}")
    print("Performing OCR...")
    print( datetime.datetime.now())

# results = reader.readtext(enhanced)
    results = reader.readtext(
        enhanced,
        # decoder='greedy', 
        # contrast_ths=0.0,       # ביטול ריצה כפולה על טקסטים חיוורים
        # adjust_contrast=0.0,    # לא רלוונטי כשהסף הוא 0, אבל ליתר ביטחון
        # batch_size=64,       # הקטנת רזולוציית העיבוד (במקום 2560)
            # עיבוד מקבילי של מילים רבות
        #workers=4              # שימוש ב-Threads לטעינת נתונים (אם רלוונטי)
    )

    print(f"OCR found {len(results)} text regions")
    print( datetime.datetime.now())

    tasks = []
    indices_needing_ocr = []

    for i, (bbox, text, confidence) in enumerate(results):
        if confidence < 0.7:
            bbox_array = np.array(bbox)
            x_min = int(bbox_array[:, 0].min())
            y_min = int(bbox_array[:, 1].min())
            x_max = int(bbox_array[:, 0].max())
            y_max = int(bbox_array[:, 1].max())
            
            # Ensure coordinates are within image bounds to avoid errors
            crop = enhanced[max(0, y_min):y_max, max(0, x_min):x_max]
            
            # Store the crop and the index to map it back later
            tasks.append(crop)
            indices_needing_ocr.append(i)

    # 2. PROCESS: Run Tesseract in parallel
    # This runs multiple Tesseract processes at once, skipping the wait time
    def run_tesseract(img):
        return pytesseract.image_to_string(img, lang='heb+eng', config=custom_config)

    tesseract_outputs = []
    custom_config = r'--oem 3 --psm 11'
    if tasks:
        print(f"Refining {len(tasks)} low-confidence regions in parallel...")
        print( datetime.datetime.now())
        with ThreadPoolExecutor(max_workers=4) as executor:
            tesseract_outputs = list(executor.map(run_tesseract, tasks))

    # Map results back to a dictionary for easy lookup
    # { original_index: tesseract_text }
    refined_texts = dict(zip(indices_needing_ocr, tesseract_outputs))

    # 3. MERGE: Build your final list
    text_boxes = []
    print('finish hebrew OCR')

    print( datetime.datetime.now())

    # Extract all text with bounding boxes
    text_boxes = []
    for i, (bbox, text, confidence) in enumerate(results):
        # bbox is a list of 4 corner points
        bbox_array = np.array(bbox)
        x_min = int(bbox_array[:, 0].min())
        y_min = int(bbox_array[:, 1].min())
        x_max = int(bbox_array[:, 0].max())
        y_max = int(bbox_array[:, 1].max())
        # ind=indices_needing_ocr.index(i) if i in indices_needing_ocr else -1
        if i in refined_texts:
            final_text = refined_texts[i]
        else:
            final_text = text # Use original EasyOCR text

        text_boxes.append({
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min,
            'text': final_text.strip(),#if heb_text.strip() else text.strip(),
            'confidence': confidence,
            'center_x': (x_min + x_max) / 2,
            'center_y': (y_min + y_max) / 2
        })

    print(f"first 5 text boxes: {text_boxes[:5]}")        
    pdf_bytes = create_pdf_in_memory(image_rgb, text_boxes)
    pdf_bytes = compress_pdf_bytes(pdf_bytes)
    return Response(
        content=pdf_bytes, 
        media_type="application/pdf", 
        headers={
            # שימי לב: הסרתי את הנתיב מהשם כדי שיהיה שם נקי להורדה
            "Content-Disposition": f"attachment; filename=ocr_easy_result.pdf"
        }
    )


import subprocess

def compress_pdf_bytes(pdf_bytes):
    """
    כיווץ PDF באמצעות Ghostscript
    """
    try:
        # הרצת תהליך GS שקורא מ-Stdin וכותב ל-Stdout
        process = subprocess.Popen(
            [
                "gs",
                "-sDEVICE=pdfwrite",
                "-dCompatibilityLevel=1.4",
                "-dPDFSETTINGS=/ebook", # אופציות: /screen (הכי נמוך), /ebook (מומלץ), /printer (גבוה)
                "-dNOPAUSE",
                "-dQUIET",
                "-dBATCH",
                "-sOutputFile=-", # פלט ל-stdout
                "-"               # קלט מ-stdin
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # שליחת ה-PDF המקורי וקבלת המכווץ
        compressed_pdf, stderr = process.communicate(input=pdf_bytes)
        
        if process.returncode == 0 and len(compressed_pdf) > 0:
            print(f"Compressed PDF from {len(pdf_bytes)} to {len(compressed_pdf)} bytes")
            return compressed_pdf
        else:
            print(f"Ghostscript warning: {stderr.decode()}")
            return pdf_bytes # מחזירים את המקורי אם נכשל
            
    except Exception as e:
        print(f"Compression failed: {e}")
        return pdf_bytes

# --- בתוך ה-Endpoint ---
# pdf_bytes = create_pdf_in_memory(...)
# final_pdf = compress_pdf_bytes(pdf_bytes)
# return Response(content=final_pdf...)
# הפונקציה כעת מחזירה bytes ולא שומרת קובץ

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import io
import cv2
import os

def create_pdf_in_memory(image_rgb, text_boxes, quality=60, max_width=1600):
    """
    יצירת PDF אופטימלי:
    1. שומר על גודל דף מקורי (כדי שהקואורדינטות של הטקסט לא יהרסו).
    2. מקטין פיזית את התמונה (Resize) כדי לחסוך מקום.
    3. דוחס ב-JPEG.
    """
    # גודל מקורי של התמונה (והדף)
    h_orig, w_orig, _ = image_rgb.shape
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(w_orig, h_orig))

    # --- שלב 1: הקטנת רזולוציה חכמה (Downscaling) ---
    # אנחנו יוצרים עותק מוקטן של התמונה רק בשביל הרקע
    scale_factor = 1.0
    if w_orig > max_width:
        scale_factor = max_width / w_orig
        new_w = int(w_orig * scale_factor)
        new_h = int(h_orig * scale_factor)
        # הקטנת התמונה הפיזית
        img_resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_resized = image_rgb

    # --- שלב 2: דחיסת JPEG ---
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    is_success, buffer_img = cv2.imencode(".jpg", img_bgr, encode_params)
    
    if is_success:
        img_stream = io.BytesIO(buffer_img)
        from reportlab.lib.utils import ImageReader
        
        # --- שלב 3: הציור (המתיחה) ---
        # אנחנו מציירים את התמונה המוקטנת (img_stream)
        # אבל אומרים ל-PDF לפרוס אותה על הגודל המקורי (w_orig, h_orig)
        # זה שומר על חדות סבירה לקריאה, אבל משקל קובץ נמוך מאוד
        c.drawImage(ImageReader(img_stream), 0, 0, width=w_orig, height=h_orig)

    # --- 4. הגדרת פונט ---
    font_name = "Helvetica" 
    font_path = '/usr/share/fonts/arial.ttf' # וודאי נתיב

    try:
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf'))
        font_name = 'Arial'
    except Exception as e:
        print(f"Warning: Failed to load font: {e}")
    
    # --- 5. ציור הטקסט (לפי הקואורדינטות המקוריות - לא צריך לשנות כלום!) ---
    for box in text_boxes:
        text = box['text']
        if not text: continue

        is_hebrew = any("\u0590" <= char <= "\u05EA" for char in text)

        x = box['x_min']
        y = h_orig - box['y_max'] # משתמשים בגובה המקורי
        box_width = box['width']
        box_height = box['height']
        
        font_size = box_height * 0.8
        if font_size <= 0: font_size = 10

        text_object = c.beginText()
        text_object.setFont(font_name, font_size)
        text_object.setTextRenderMode(3) 

        # חישוב רוחב ומתיחה (אותו קוד שעבד לך קודם)
        text_width_pixels = c.stringWidth(text, font_name, font_size)
        scale_factor_text = 1.0
        
        if text_width_pixels > 0 and box_width > 0:
            stretch_percent = (box_width / text_width_pixels) * 100
            if 50 < stretch_percent < 200:
                text_object.setHorizScale(stretch_percent)
                scale_factor_text = stretch_percent / 100.0

        if is_hebrew:
            # ציור RTL ידני
            cursor_x = x + box_width
            for char in text:
                char_w = c.stringWidth(char, font_name, font_size) * scale_factor_text
                cursor_x -= char_w
                text_object.setTextOrigin(cursor_x, y)
                text_object.textOut(char)
        else:
            text_object.setTextOrigin(x, y)
            text_object.textOut(text)

        c.drawText(text_object)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# def create_pdf_in_memory(image_rgb, text_boxes,quality=60):
#     """
#     יצירת PDF בזיכרון עם טקסט בלתי נראה (מתוקן)
#     """
#     h_img, w_img, _ = image_rgb.shape
#     buffer = io.BytesIO()
#     c = canvas.Canvas(buffer, pagesize=(w_img, h_img))

#     # 1. ציור תמונת הרקע
#     img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#     # הגדרת איכות JPEG (בין 0 ל-100). 60-70 זה הממוצע המומלץ למסמכים.
#     encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    
#     is_success, buffer_img = cv2.imencode(".jpg", img_bgr, encode_params)

#     if is_success:
#         img_stream = io.BytesIO(buffer_img)
#         from reportlab.lib.utils import ImageReader
#         c.drawImage(ImageReader(img_stream), 0, 0, width=w_img, height=h_img)

#     # 2. הגדרת פונט (חובה לעברית!)
#     font_name = "Helvetica" # ברירת מחדל
#     try:
#         # נסיון לטעון פונט אריאל (וודאי שהקובץ קיים אצלך בדוקר!)
#         # c.setFont מקבל שם פונט שנרשם.
#         pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf')) 
#         font_name = "Arial"
#     except:
#         pass

#     # 3. כתיבת הטקסט הבלתי נראה - התיקון כאן!
#     for box in text_boxes:
#         text = box['text']
#         if not text: continue

#         # בדיקה אם הטקסט מכיל עברית
#         is_hebrew = any("\u0590" <= char <= "\u05EA" for char in text)

#         # נתונים גיאומטריים
#         x = box['x_min']
#         y = h_img - box['y_max'] # המרה לקואורדינטות PDF
#         box_width = box['width']
#         box_height = box['height']
        
#         # חישוב גודל פונט
#         font_size = box_height * 0.8
#         if font_size <= 0: font_size = 10

#         # יצירת אובייקט טקסט
#         text_object = c.beginText()
#         text_object.setFont(font_name, font_size)
#         text_object.setTextRenderMode(3) # בלתי נראה (Mode 3)

#         # חישוב רוחב הטקסט המקורי (לצורך מתיחה)
#         text_width_pixels = c.stringWidth(text, font_name, font_size)
#         scale_factor = 1.0
        
#         # חישוב פקטור מתיחה (Stretch)
#         if text_width_pixels > 0 and box_width > 0:
#             stretch_percent = (box_width / text_width_pixels) * 100
#             # הגבלה למניעת עיוותים קיצוניים
#             if 50 < stretch_percent < 200:
#                 text_object.setHorizScale(stretch_percent)
#                 scale_factor = stretch_percent / 100.0

#         if is_hebrew:
#             # --- האלגוריתם לעברית (RTL ידני) ---
#             # אנחנו שומרים על הטקסט המקורי (לוגי) כדי שהחיפוש יעבוד,
#             # אבל מזיזים את המיקום של כל אות כדי לצייר אותה מימין לשמאל.
            
#             # מתחילים מהקצה הימני של התיבה
#             cursor_x = x + box_width
            
#             # עוברים אות-אות לפי הסדר הלוגי ("מ", "ע", "ב"...)
#             for char in text:
#                 # רוחב האות הנוכחית (כולל המתיחה)
#                 char_w = c.stringWidth(char, font_name, font_size) * scale_factor
                
#                 # מזיזים את הסמן שמאלה ("אחורה") ברוחב האות
#                 cursor_x -= char_w
                
#                 # מגדירים את המיקום החדש וכותבים את האות
#                 text_object.setTextOrigin(cursor_x, y)
#                 text_object.textOut(char)
                
#         else:
#             # --- אנגלית/מספרים (LTR רגיל) ---
#             text_object.setTextOrigin(x, y)
#             text_object.textOut(text)

#         c.drawText(text_object)

#     c.save()
#     buffer.seek(0)
#     return buffer.getvalue()
   


def get_first_page_image(file_path, dpi=150):
    if file_path.lower().endswith(".pdf"):
        pages = convert_from_path(file_path, dpi=dpi, first_page=1, last_page=1)
        image = np.array(pages[0])
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        return cv2.imread(file_path)

def profile_image(image_bgr):
    h_img, w_img, _ = image_bgr.shape
    image_area = h_img * w_img
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    data = pytesseract.image_to_data(gray, lang="heb+eng", config="--psm 3", output_type=Output.DICT)

    # חילוץ קואורדינטות של תיבות עם טקסט
    boxes = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0 and data['text'][i].strip():
            boxes.append({
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i]
            })

    if not boxes:
        return {"is_sparse": False, "box_count": 0, "avg_confidence": 0}

    # חישוב צפיפות (Density) - כמה מהדף מכוסה בטקסט
    total_text_area = sum([b['w'] * b['h'] for b in boxes])
    coverage_ratio = total_text_area / image_area

    # חישוב פיזור (Dispersion) - נבדוק את המרחק הממוצע בין מרכזי התיבות
    centers = np.array([ [b['x'] + b['w']/2, b['y'] + b['h']/2] for b in boxes])
    
    # אם יש מעט קופסאות, נבדוק אם הן רחוקות מאוד
    if len(boxes) > 1:
        # מרחק ממוצע לנקודה הקרובה ביותר (Nearest Neighbor)
        from scipy.spatial import distance_matrix
        dist_mat = distance_matrix(centers, centers)
        np.fill_diagonal(dist_mat, np.inf)
        avg_nearest_dist = np.mean(np.min(dist_mat, axis=1))
    else:
        avg_nearest_dist = 0

    return {
        "box_count": len(boxes),
        "coverage_ratio": coverage_ratio,
        "avg_nearest_dist": avg_nearest_dist,
        "image_width": w_img,
        "avg_confidence": np.mean([int(data['conf'][i]) for i in range(len(data['text'])) if int(data['conf'][i]) > 0])
    }

def choose_engine(profile):
    # 1. טיפול ב-EasyOCR (מסמכים חלשים מאוד/צילומים)
    if profile["avg_confidence"] < 40 or profile["box_count"] < 5:
        return {"engine": "easyocr", "psm": None}

    # 2. זיהוי PSM 11 (טקסט מפוזר)
    # תנאי: מעט טקסט (coverage נמוך) והמילים רחוקות אחת מהשנייה
    if profile["coverage_ratio"] < 0.05 and profile["avg_nearest_dist"] > (profile["image_width"] * 0.1):
        return {
            "engine": "ocrmypdf",
            "psm": 11
        }

    # 3. מסמך רגיל (Multi-column / Blocks)
    if profile["box_count"] > 50:
        return {
            "engine": "ocrmypdf",
            "psm": 3  # אוטומטי מלא
        }

    # 4. מסמך קצר/בלוק יחיד
    return {
        "engine": "ocrmypdf",
        "psm": 6
    }

def compute_ocr_score(ocr_data):
    """
    ocr_data = pytesseract.image_to_data(...)
    """
    texts = []
    confs = []

    for i in range(len(ocr_data['text'])):
        txt = ocr_data['text'][i].strip()
        conf = int(ocr_data['conf'][i])
        if txt and conf > -1:
            texts.append(txt)
            confs.append(conf)

    if not texts:
        return 0.0

    text_len = sum(len(t) for t in texts)
    valid_chars = sum(len(re.findall(r'[A-Za-z\u0590-\u05EA]', t)) for t in texts)

    valid_ratio = valid_chars / max(text_len, 1)
    avg_conf = np.mean(confs) / 100.0

    # מילים חד-אותיות = רע
    short_words = sum(1 for t in texts if len(t) == 1)
    consistency = 1 - (short_words / max(len(texts), 1))

    score = (
        0.5 * avg_conf +
        0.3 * valid_ratio +
        0.2 * consistency
    )

    return round(score, 3)


def adaptive_ocr(image_bgr):
    profile = profile_image(image_bgr)
    decision = choose_engine(profile)

    result = {
        "engine": decision["engine"],
        "psm": decision["psm"],
        "fallback": False,
        "confidence_score": 0
    }

    # ניסיון OCRmyPDF / Tesseract
    if decision["engine"] == "ocrmypdf":
        data = pytesseract.image_to_data(
            image_bgr,
            lang="heb+eng",
            config=f"--psm {decision['psm']}",
            output_type=Output.DICT
        )

        score = compute_ocr_score(data)
        result["confidence_score"] = score

        # fallback
        if score < 0.75:
            result["engine"] = "easyocr"
            result["fallback"] = True

        else:
            return result

    # EasyOCR path
    result["engine"] = "easyocr"
    result["confidence_score"] = 0.55  # תעדכני לפי הלוגיקה שלך

    return result

def safe_ocrmypdf(input_pdf, output_pdf, psm):
    
    tmp_fixed_path = input_pdf + "_fixed.pdf"
    is_image = convert_image_to_pdf(input_pdf, tmp_fixed_path)
    
    source_file = input_pdf

    if is_image:
        print("File was identified as an image and converted to PDF.")
        source_file = tmp_fixed_path
    else:
        print("File is not an image. Attempting PDF repair...")
        # רק אם זו לא תמונה, ננסה לתקן כ-PDF
        # זה מונע את קריסת Ghostscript על קבצי PNG
        # is_repaired = repair_pdf(input_pdf, tmp_fixed_path)
        
        # if is_repaired:
        #     source_file = tmp_fixed_path
        # else:
        #     # אם התיקון נכשל, נשתמש במקור (אולי הוא PDF תקין ו-GS סתם נכשל)
        #     print("Repair failed or unnecessary, using original file.")
        #     source_file = input_pdf
   
   
    try:
        ocrmypdf.ocr(
            source_file, 
            output_pdf, 
            image_dpi=300,
            clean=True,       
            clean_final=True, 
            deskew=True,       
            rotate_pages=True, 
            tesseract_pagesegmode=psm,
            sidecar=True,
            unpaper_args="--no-blackfilter --no-grayfilter --no-border --no-deskew",
            lang="heb+eng",
            force_ocr=True,  
            jobs=4,            
            progress_bar=False      
        )
        return True, input_pdf

    except Exception:
        # ניסיון תיקון
        # fixed_pdf = input_pdf + "_fixed.pdf"
        repaired = False
        if not is_image:
            repaired = repair_pdf(input_pdf, tmp_fixed_path)
        if not repaired:
            return False, input_pdf

        # ניסיון OCR מחדש
        ocrmypdf.ocr(
            tmp_fixed_path, 
            output_pdf, 
            image_dpi=300,
            clean=True,       
            clean_final=True, 
            deskew=True,       
            rotate_pages=True, 
            tesseract_pagesegmode=psm,
            sidecar=True,
            unpaper_args="--no-blackfilter --no-grayfilter --no-border --no-deskew",
            lang="heb+eng",
            force_ocr=True,  
            jobs=4,            
            progress_bar=False      
        )
        return True, tmp_fixed_path


@app.post("/ocr/adaptive")
async def adaptive_ocr_endpoint(file: UploadFile = File(...)):
    tmp_in_path = None
    tmp_out_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=pathlib.Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            tmp_in_path = tmp.name

        # 🧠 שלב החלטה
        preview_image = get_first_page_image(tmp_in_path)
        decision = adaptive_ocr(preview_image)

        tmp_out_path = tmp_in_path + "_out.pdf"

        # 🚦 ניתוב
        if decision["engine"] == "ocrmypdf":
            success, fixed_path = safe_ocrmypdf(
                input_pdf=tmp_in_path,
                output_pdf=tmp_out_path,
                psm=decision["psm"]
            )
            if not success:
                raise HTTPException(status_code=500, detail="OCRmyPDF failed even after repair.")
            with open(tmp_out_path, "rb") as f:
                pdf_bytes = f.read()
            pdf_bytes = compress_pdf_bytes(pdf_bytes)
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f"attachment; filename={file.filename}_ocr.pdf",
                    "X-OCR-Engine": "ocrmypdf",
                    "X-OCR-PSM": str(decision["psm"]),
                    "X-OCR-Score": str(decision["confidence_score"])
                }
            )

        # 🟥 EasyOCR fallback
        return await ocr_easyocr_pdf_overlay(tmp_in_path)

    finally:
        for p in [tmp_in_path, tmp_out_path]:
            if p and os.path.exists(p):
                os.remove(p)




if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
























# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse, StreamingResponse
# import tempfile
# import os
# from pdf2image import convert_from_path
# import pytesseract
# from PIL import Image, ImageDraw
# import ocrmypdf
# from fastapi.responses import Response

# app = FastAPI()

# @app.post("/ocr")
# async def ocr_pdf(file: UploadFile = File(...)):
#     # Save uploaded PDF to temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(await file.read())
#         tmp_path = tmp.name
#     # Convert PDF pages to images
#     images = convert_from_path(tmp_path)
#     result = []
#     for page_num, img in enumerate(images):
#         ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
#         page_result = []
#         n_boxes = len(ocr_data['text'])
#         for i in range(n_boxes):
#             if ocr_data['text'][i].strip():
#                 page_result.append({
#                     'text': ocr_data['text'][i],
#                     'left': ocr_data['left'][i],
#                     'top': ocr_data['top'][i],
#                     'width': ocr_data['width'][i],
#                     'height': ocr_data['height'][i],
#                     'conf': ocr_data['conf'][i]
#                 })
#         result.append({'page': page_num + 1, 'items': page_result})
#     os.remove(tmp_path)
#     return JSONResponse(content={'pages': result})

# @app.post("/ocr/pdf-overlay")
# async def ocr_pdf_overlay(file: UploadFile = File(...)):
#     tmp_in_path = None
#     tmp_out_path = None
    
#     try:
#         # יצירת קובץ קלט זמני
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
#             tmp_in.write(await file.read())
#             tmp_in_path = tmp_in.name
            
#         # יצירת שם לקובץ הפלט
#         tmp_out_path = tmp_in_path + "_out.pdf"

#         # הרצת OCRmyPDF
#         # --force-ocr: מכריח ביצוע OCR גם אם יש קצת טקסט
#         # -l heb+eng: תמיכה בעברית ואנגלית
#         # --jobs 4: שימוש ב-4 ליבות לשיפור מהירות
#         ocrmypdf.ocr(
#             tmp_in_path, 
#             tmp_out_path, 

#             deskew=True,
#             clean=True,          # Cleans up image noise before OCR
#             clean_final=True,    # Keeps cleaned image in final PDF (optional)
#             rotate_pages=True,   # Auto-rotates if needed
#             unpaper_args="--no-blackfilter --no-grayfilter --no-border --no-deskew",  # Safe defaults; test and tweak
#             tesseract_config={"load_system_dawg": "0", "language_model_penalty_non_dict_word": "0"},
#             lang="heb+eng",
#             force_ocr=True,
#             jobs=4,              # Use multiple cores
#             progress_bar=False
#         )

#         with open(tmp_out_path, "rb") as f:
#             pdf_bytes = f.read()

#         return Response(
#             content=pdf_bytes, 
#             media_type="application/pdf", 
#             headers={"Content-Disposition": "attachment; filename=ocr_result.pdf"}
#         )

#     except Exception as e:
#         # הדפסת השגיאה ללוג
#         print(f"Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
        
#     finally:
#         # ניקוי קבצים זמניים בכל מצב
#         if tmp_in_path and os.path.exists(tmp_in_path):
#             os.remove(tmp_in_path)
#         if tmp_out_path and os.path.exists(tmp_out_path):
#             os.remove(tmp_out_path)

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)