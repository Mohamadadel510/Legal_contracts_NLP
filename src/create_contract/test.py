import sys
from pathlib import Path

# إضافة جذر المشروع لمسارات بايثون
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import json
from src.create_contract.pipeline import create_contract
from src.create_contract.pdf_exporter import export_contract_to_pdf  # استيراد دالة الـ PDF

test_request = """
عايز أعمل عقد بيع شقة.
المؤجر أو البائع: أحمد محمود علي، العنوان: 10 شارع الهرم، الجيزة.
المشتري: محمد سامي حسن، رقم قومي: 29001010123456، العنوان: 25 شارع فيصل، الجيزة.
العقار يقع في 15 شارع جامعة الدول العربية، المهندسين، الجيزة، ومساحته 120 متر مربع.
الثمن الإجمالي 2,500,000 جنيه مصري، دفعة مقدمة 1,000,000 جنيه، والباقي على 3 دفعات شهرية.
مدة العقد شهرين.
"""

print("Running LegalLens Pipeline...")
result = create_contract(
    user_message=test_request,
    top_k=5,
    verbose=True
)

print("\n" + "="*50)
print("FINAL CONTRACT OUTPUT:")
print("="*50)
print(result["contract_text"])

print("\n" + "="*50)
print("VALIDATION REPORT:")
print("="*50)
print(json.dumps(result["validation"], ensure_ascii=False, indent=2))

# ==========================================
# تصدير العقد النهائي إلى ملف PDF تلقائياً
# ==========================================
output_pdf_path = "final_contract_output.pdf"
export_contract_to_pdf(result["contract_text"], output_filename=output_pdf_path)