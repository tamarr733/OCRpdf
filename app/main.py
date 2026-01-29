import datetime
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

@app.post("/ocr/ocrmypdf-overlay")
async def ocr_pdf_overlay(file: UploadFile = File(...)):
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
        is_image = convert_image_to_pdf(tmp_in_path, tmp_fixed_path)
        
        source_file = None

        if is_image:
            print("File was identified as an image and converted to PDF.")
            source_file = tmp_fixed_path
        else:
            print("File is not an image. Attempting PDF repair...")
            # רק אם זו לא תמונה, ננסה לתקן כ-PDF
            # זה מונע את קריסת Ghostscript על קבצי PNG
            is_repaired = repair_pdf(tmp_in_path, tmp_fixed_path)
            
            if is_repaired:
                source_file = tmp_fixed_path
            else:
                # אם התיקון נכשל, נשתמש במקור (אולי הוא PDF תקין ו-GS סתם נכשל)
                print("Repair failed or unnecessary, using original file.")
                source_file = tmp_in_path

        # 4. הרצת OCRmyPDF
        print(f"Starting OCR on {source_file}")
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
            lang="heb+eng",
            force_ocr=True,  
            jobs=4,            
            progress_bar=False      
        )

        with open(tmp_out_path, "rb") as f:
            pdf_bytes = f.read()

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
async def ocr_easyocr_pdf_overlay(file: UploadFile = File(...)):
    file_path = await get_file_path(file)
    pages = None
    if file_path.lower().endswith('.pdf'):
        pages = convert_from_path(file_path, 300)
    if pages:
        image = np.array(pages[0])
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.imread(file_path)
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
        if confidence < 0.5:
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
    return Response(
        content=pdf_bytes, 
        media_type="application/pdf", 
        headers={
            # שימי לב: הסרתי את הנתיב מהשם כדי שיהיה שם נקי להורדה
            "Content-Disposition": f"attachment; filename=ocr_easy_result.pdf"
        }
    )

# הפונקציה כעת מחזירה bytes ולא שומרת קובץ
def create_pdf_in_memory(image_rgb, text_boxes):
    """
    יצירת PDF בזיכרון עם טקסט בלתי נראה (מתוקן)
    """
    h_img, w_img, _ = image_rgb.shape
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=(w_img, h_img))

    # 1. ציור תמונת הרקע
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    is_success, buffer_img = cv2.imencode(".jpg", img_bgr)
    if is_success:
        img_stream = io.BytesIO(buffer_img)
        from reportlab.lib.utils import ImageReader
        c.drawImage(ImageReader(img_stream), 0, 0, width=w_img, height=h_img)

    # 2. הגדרת פונט (חובה לעברית!)
    font_name = "Helvetica" # ברירת מחדל
    try:
        # נסיון לטעון פונט אריאל (וודאי שהקובץ קיים אצלך בדוקר!)
        # c.setFont מקבל שם פונט שנרשם.
        pdfmetrics.registerFont(TTFont('Arial', 'arial.ttf')) 
        font_name = "Arial"
    except:
        pass

    # 3. כתיבת הטקסט הבלתי נראה - התיקון כאן!
    for box in text_boxes:
        text = box['text']
        if not text: continue

        # בדיקה אם הטקסט מכיל עברית
        is_hebrew = any("\u0590" <= char <= "\u05EA" for char in text)

        # נתונים גיאומטריים
        x = box['x_min']
        y = h_img - box['y_max'] # המרה לקואורדינטות PDF
        box_width = box['width']
        box_height = box['height']
        
        # חישוב גודל פונט
        font_size = box_height * 0.8
        if font_size <= 0: font_size = 10

        # יצירת אובייקט טקסט
        text_object = c.beginText()
        text_object.setFont(font_name, font_size)
        text_object.setTextRenderMode(3) # בלתי נראה (Mode 3)

        # חישוב רוחב הטקסט המקורי (לצורך מתיחה)
        text_width_pixels = c.stringWidth(text, font_name, font_size)
        scale_factor = 1.0
        
        # חישוב פקטור מתיחה (Stretch)
        if text_width_pixels > 0 and box_width > 0:
            stretch_percent = (box_width / text_width_pixels) * 100
            # הגבלה למניעת עיוותים קיצוניים
            if 50 < stretch_percent < 200:
                text_object.setHorizScale(stretch_percent)
                scale_factor = stretch_percent / 100.0

        if is_hebrew:
            # --- האלגוריתם לעברית (RTL ידני) ---
            # אנחנו שומרים על הטקסט המקורי (לוגי) כדי שהחיפוש יעבוד,
            # אבל מזיזים את המיקום של כל אות כדי לצייר אותה מימין לשמאל.
            
            # מתחילים מהקצה הימני של התיבה
            cursor_x = x + box_width
            
            # עוברים אות-אות לפי הסדר הלוגי ("מ", "ע", "ב"...)
            for char in text:
                # רוחב האות הנוכחית (כולל המתיחה)
                char_w = c.stringWidth(char, font_name, font_size) * scale_factor
                
                # מזיזים את הסמן שמאלה ("אחורה") ברוחב האות
                cursor_x -= char_w
                
                # מגדירים את המיקום החדש וכותבים את האות
                text_object.setTextOrigin(cursor_x, y)
                text_object.textOut(char)
                
        else:
            # --- אנגלית/מספרים (LTR רגיל) ---
            text_object.setTextOrigin(x, y)
            text_object.textOut(text)

        c.drawText(text_object)

    c.save()
    buffer.seek(0)
    return buffer.getvalue()
   



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