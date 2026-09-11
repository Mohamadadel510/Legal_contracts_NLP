import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display

def export_contract_to_pdf(contract_text, output_filename="LegalLens_Contract_Final.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # تسجيل خط Arial الداعم للعربية
    try:
        font_path = "C:/Windows/Fonts/arial.ttf"
        if os.path.exists(font_path):
            pdfmetrics.registerFont(TTFont('ArabicFont', font_path))
            font_name = 'ArabicFont'
        else:
            font_name = 'Helvetica'
    except Exception:
        font_name = 'Helvetica'

    arabic_style = ParagraphStyle(
        'ArabicStyle',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=11,
        leading=16,
        alignment=2, # محاذاة لليمين
        textColor=colors.black
    )
    
    story = []
    clean_text = contract_text.replace("**", "").replace("###", "")
    
    for line in clean_text.split("\n"):
        if not line.strip():
            story.append(Spacer(1, 8))
            continue
            
        # إعادة تشكيل الحروف العربية وعكس اتجاه النص ليظهر مضبوطاً
        try:
            reshaped_text = arabic_reshaper.reshape(line)
            bidi_text = get_display(reshaped_text)
            safe_line = bidi_text.encode('utf-8', 'ignore').decode('utf-8')
        except Exception:
            safe_line = line
            
        p = Paragraph(safe_line, arabic_style)
        story.append(p)
    
    doc.build(story)
    print(f"\n[SUCCESS] PDF successfully generated (reshaped & fixed): {output_filename}")