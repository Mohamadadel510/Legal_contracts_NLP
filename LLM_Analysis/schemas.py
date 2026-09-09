"""Pydantic data contracts shared by the analysis and drafting engines."""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Risk analysis (clause-by-clause review of an existing contract)
# --------------------------------------------------------------------------
class RiskLevel(str, Enum):
    RED = "أحمر"  # high risk / absolutely void / abusive clause
    YELLOW = "أصفر"  # medium risk / ambiguous / one-sided clause
    GREEN = "أخضر"  # safe / compliant with the law


class ClauseRiskAnalysis(BaseModel):
    clause_id: int = Field(description="رقم البند")
    clause_label: str = Field(description="عنوان أو مسمى البند")
    clause_text: str = Field(description="النص الأصلي للبند")

    reasoning_steps: List[str] = Field(
        description="خطوات التحليل المنطقي والقانوني خطوة بخطوة (Chain of Thought)"
    )
    risk_level: RiskLevel = Field(
        description="مستوى الخطورة: أحمر (مرتفع)، أصفر (متوسط)، أخضر (آمن)"
    )
    risk_score: int = Field(
        description="درجة الخطورة من 1 (آمن جداً) إلى 10 (باطل ومجحف للغاية)"
    )

    is_void_legal_term: bool = Field(
        description="هل البند يعتبر باطلاً بطلاناً مطلقاً أو نسبياً وفقاً للقانون المصري؟"
    )
    cited_law_articles: List[str] = Field(
        description="أسماء وأرقام المواد القانونية المعتمد عليها المأخوذة من الـ RAG"
    )

    simple_explanation: str = Field(
        description="شرح مبسط للبند ولماذا يشكل خطورة بلغة عامية بسيطة"
    )
    legal_rationale: str = Field(
        description="التعليل القانوني الدقيق والربط بالمواد القانونية المطبقة"
    )
    suggested_balanced_clause: Optional[str] = Field(
        default=None,
        description="الصياغة البديلة المقترحة والمعدلة لتحقيق التوازن العقدية",
    )

    @field_validator("risk_score")
    @classmethod
    def check_score_range(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError("Risk score must be between 1 and 10")
        return v


class OverallSummarySchema(BaseModel):
    contract_title: str = Field(description="مسمى العقد بناءً على تحليل المخرجات")
    overall_risk_score: int = Field(description="التقييم الإجمالي للعقد من 1 إلى 10")
    overall_risk_summary: str = Field(
        description="ملخص تنفيذي للمخاطر وتوصيات قانونية شامله للمستخدم"
    )


# --------------------------------------------------------------------------
# Drafting (constructing a new contract from user input)
# --------------------------------------------------------------------------
class ContractUserInput(BaseModel):
    contract_type: str = Field(description="نوع العقد (مثال: عقد إيجار مسكن، عقد عمل، عقد بيع)")
    party_one_name: str = Field(description="اسم الطرف الأول وصفته (مثال: المؤجر / البائع)")
    party_two_name: str = Field(description="اسم الطرف الثاني وصفته (مثال: المستأجر / المشتري)")
    key_terms: List[str] = Field(description="الشروط الأساسية (القيمة المالية، المدة، مكان العين، إلخ)")
    special_requests: Optional[List[str]] = Field(
        default_factory=list, description="أي شروط خاصة يطلبها المستخدم"
    )


class ContractClauseDraft(BaseModel):
    clause_number: int = Field(description="رقم البند")
    clause_title: str = Field(description="عنوان البند")
    clause_text: str = Field(description="النص القانوني الملزم للبند")
    legal_basis: str = Field(description="المادة القانونية المعتمد عليها من القانون المصري لضمان شرعية البند")
    balance_explanation: str = Field(
        description="توضيح كيف يحقق هذا البند التوازن وعدم الإجحاف بأحد الطرفين"
    )


class FullContractDocument(BaseModel):
    contract_title: str = Field(description="العنوان الرسمي للعقد")
    preamble: str = Field(description="ديباجة العقد وتحديد هوية الأطراف والأهلية القانونية")
    clauses: List[ContractClauseDraft] = Field(description="قائمة بنود العقد المصاغة قانونياً")
    closing_and_signatures: str = Field(
        description="صيغة الخاتمة والإقرار بالاستلام وعدد النسخ والتوقيعات"
    )
