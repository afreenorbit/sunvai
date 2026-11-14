"""Data model shared by every stage of the pipeline."""

from __future__ import annotations

from typing import Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator

SpanType = Literal["amount", "digits", "name", "consent", "date"]
Split = Literal["train", "val", "test"]


class CriticalSpan(BaseModel):
    type: SpanType
    surface: str = Field(description="Exact substring of the turn text")
    value: Union[int, str] = Field(
        description="Canonical value: rupees (int) for amount, digit string for digits, "
        "'yes'/'no' for consent, ISO date for date, romanised name for name"
    )


class Turn(BaseModel):
    index: int
    speaker: Literal["customer", "agent"]
    text: str
    spans: list[CriticalSpan] = []

    @model_validator(mode="after")
    def spans_appear_in_text(self) -> "Turn":
        lowered = self.text.lower()
        for span in self.spans:
            if span.surface.lower() not in lowered:
                raise ValueError(f"span surface {span.surface!r} not found in turn text")
            if span.type == "amount" and not isinstance(span.value, int):
                raise ValueError("amount value must be an integer number of rupees")
            if span.type == "digits" and not str(span.value).isdigit():
                raise ValueError("digits value must contain only digits")
            if span.type == "consent" and span.value not in ("yes", "no"):
                raise ValueError("consent value must be 'yes' or 'no'")
        return self


class CallScript(BaseModel):
    call_id: str
    lang: Literal["hi-IN", "kn-IN", "ta-IN"]
    domain: str
    split: Optional[Split] = None
    turns: list[Turn]

    def customer_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.speaker == "customer"]


class Clip(BaseModel):
    """One audio rendering of one customer turn."""

    clip_id: str  # {call_id}_t{turn_index}
    call_id: str
    turn_index: int
    lang: str
    voice_source: str  # timbre_v3 | edge | human
    path: str  # relative path of the clean wav


class Transcript(BaseModel):
    clip_id: str
    voice_source: str
    condition: str
    provider: str
    text: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class ExtractedFields(BaseModel):
    """Output schema for the field extractor."""

    amount_inr: Optional[int] = None
    digits: list[str] = []
    name: Optional[str] = None
    consent: Optional[Literal["yes", "no"]] = None
    date: Optional[str] = None
