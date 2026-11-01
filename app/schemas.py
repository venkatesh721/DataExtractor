from datetime import date
from typing import Annotated, Generic, Literal, TypeVar

import pycountry
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    StrictStr,
    computed_field,
    field_validator,
)

from app.config import settings

Category = Literal["food", "travel", "utilities", "software", "other"]
Amount = StrictInt | Annotated[StrictFloat, Field(allow_inf_nan=False)] | None
Confidence = Annotated[StrictFloat, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
InternalConfidence = Annotated[float, Field(allow_inf_nan=False)]
CONFIDENCE_THRESHOLD = 0.75
T = TypeVar("T")


class ExtractRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=settings.max_input_characters,
        description="Raw, unstructured text to extract structured data from.",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain at least one non-whitespace character.")
        return value


class InternalFieldResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T
    confidence: InternalConfidence
    reason: StrictStr = Field(min_length=1)


class InternalExtraction(BaseModel):
    """Strict full response contract expected from Gemini, including private reasons."""

    model_config = ConfigDict(extra="forbid")

    vendor_name: InternalFieldResult[StrictStr | None]
    amount: InternalFieldResult[Amount]
    currency: InternalFieldResult[StrictStr | None]
    date: InternalFieldResult[StrictStr | None]
    category: InternalFieldResult[Category | None]
    description: InternalFieldResult[StrictStr | None]
    invoice_id: InternalFieldResult[StrictStr | None]


class PublicFieldResult(BaseModel, Generic[T]):
    model_config = ConfigDict(extra="forbid")

    value: T
    confidence: Confidence

    @computed_field
    @property
    def needs_review(self) -> bool:
        return self.confidence < CONFIDENCE_THRESHOLD


class VendorNameFieldResult(PublicFieldResult[StrictStr | None]):
    pass


class AmountFieldResult(PublicFieldResult[Amount]):
    pass


class CurrencyFieldResult(PublicFieldResult[StrictStr | None]):
    @field_validator("value")
    @classmethod
    def currency_must_be_iso_4217(cls, value: str | None) -> str | None:
        if value is not None and pycountry.currencies.get(alpha_3=value) is None:
            raise ValueError("Currency must be a valid ISO 4217 code.")
        return value


class DateFieldResult(PublicFieldResult[StrictStr | None]):
    @field_validator("value")
    @classmethod
    def date_must_be_normalized(cls, value: str | None) -> str | None:
        if value is not None:
            if len(value) != 10 or date.fromisoformat(value).isoformat() != value:
                raise ValueError("Date must use the YYYY-MM-DD format.")
        return value


class CategoryFieldResult(PublicFieldResult[Category | None]):
    pass


class DescriptionFieldResult(PublicFieldResult[StrictStr | None]):
    pass


class InvoiceIdFieldResult(PublicFieldResult[StrictStr | None]):
    pass


class ExtractedFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor_name: VendorNameFieldResult
    amount: AmountFieldResult
    currency: CurrencyFieldResult
    date: DateFieldResult
    category: CategoryFieldResult
    description: DescriptionFieldResult
    invoice_id: InvoiceIdFieldResult


class ExtractResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: ExtractedFields

    @computed_field
    @property
    def review_required(self) -> bool:
        return any(
            field.needs_review
            for field in (
                self.fields.vendor_name,
                self.fields.amount,
                self.fields.currency,
                self.fields.date,
                self.fields.category,
                self.fields.description,
                self.fields.invoice_id,
            )
        )


class HealthResponse(BaseModel):
    status: str
