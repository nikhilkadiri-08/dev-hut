"""Provenance tagging for every engineering constant in this application.

This is a high-voltage decision-support tool, so an unsourced number is a liability.
Each constant carries where it came from and whether it still needs checking against
authoritative material, and the API exposes the whole register to the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Source(str, Enum):
    """Where a value came from."""

    PROBLEM_STATEMENT = "problem_statement"
    WORKBOOK = "workbook"
    DATASET = "dataset"
    STANDARD = "standard"
    LITERATURE = "literature"
    ASSUMED = "assumed"


class Status(str, Enum):
    """How much the value can be trusted."""

    SUPPLIED = "supplied"
    """Stated directly in the supplied problem statement, workbook or dataset."""

    NEEDS_VERIFICATION = "needs_verification"
    """Plausible but not confirmed against authoritative material. Must be checked."""

    CONFLICT = "conflict"
    """The supplied material disagrees with itself or with the applicable standard."""

    PLACEHOLDER = "placeholder"
    """Invented purely so the interface is usable. Carries no engineering authority."""


@dataclass(frozen=True)
class Provenance:
    source: Source
    locator: str
    """Where to look to check this, e.g. a workbook cell or a PDF page."""

    status: Status
    note: str = ""

    @property
    def is_trustworthy(self) -> bool:
        return self.status is Status.SUPPLIED

    def to_dict(self) -> dict:
        return {
            "source": self.source.value,
            "locator": self.locator,
            "status": self.status.value,
            "note": self.note,
        }


@dataclass(frozen=True)
class Assumption:
    """A single entry in the register shown to engineers and judges."""

    key: str
    subject: str
    value: str
    provenance: Provenance
    impact: str = ""
    """What goes wrong downstream if this assumption is incorrect."""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "subject": self.subject,
            "value": self.value,
            "impact": self.impact,
            **self.provenance.to_dict(),
        }


@dataclass
class AssumptionRegister:
    entries: list[Assumption] = field(default_factory=list)

    def add(self, assumption: Assumption) -> Assumption:
        self.entries.append(assumption)
        return assumption

    def requiring_attention(self) -> list[Assumption]:
        return [e for e in self.entries if not e.provenance.is_trustworthy]

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.entries]


def supplied(locator: str, note: str = "") -> Provenance:
    return Provenance(Source.WORKBOOK, locator, Status.SUPPLIED, note)


def from_problem_statement(locator: str, note: str = "") -> Provenance:
    return Provenance(Source.PROBLEM_STATEMENT, locator, Status.SUPPLIED, note)


def from_dataset(locator: str, note: str = "") -> Provenance:
    return Provenance(Source.DATASET, locator, Status.SUPPLIED, note)


def from_literature(locator: str, note: str) -> Provenance:
    """A public paper or openly cited standard table. Still needs laboratory check."""
    return Provenance(Source.LITERATURE, locator, Status.NEEDS_VERIFICATION, note)


def unverified(source: Source, locator: str, note: str) -> Provenance:
    return Provenance(source, locator, Status.NEEDS_VERIFICATION, note)


def conflicting(source: Source, locator: str, note: str) -> Provenance:
    return Provenance(source, locator, Status.CONFLICT, note)


def placeholder(locator: str, note: str) -> Provenance:
    return Provenance(Source.ASSUMED, locator, Status.PLACEHOLDER, note)
