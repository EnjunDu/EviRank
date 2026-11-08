"""
Evidence Frame: the core data structure for EviRank.

An Evidence Frame E(q) consists of six semantic slots—ENTITIES, ATTRIBUTES,
ACTIONS, RELATIONS, SCENE, KEY_DETAILS—each partitioned into three constraint
labels: REQUIRED, FORBIDDEN, IGNORABLE.

Reference: Section 3.1 of the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SlotType(Enum):
    """Six orthogonal semantic dimensions (Section 3.1)."""
    ENTITIES = "entities"
    ATTRIBUTES = "attributes"
    ACTIONS = "actions"
    RELATIONS = "relations"
    SCENE = "scene"
    KEY_DETAILS = "key_details"


class ConstraintLabel(Enum):
    """Three-way constraint labels mapping to semantic relations (Section 3.1).

    REQUIRED   -> entailment   : must be present in the target.
    FORBIDDEN  -> contradiction: must NOT be present in the target.
    IGNORABLE  -> invariance   : does not affect relevance judgement.
    """
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    IGNORABLE = "ignore"


@dataclass
class Constraint:
    """A single constraint sentence within a slot.

    All constraints are written as affirmative statements (Section I.3):
    - REQUIRED sentences describe what a correct match looks like.
    - FORBIDDEN sentences describe what a wrong match looks like (positive form).
    - IGNORABLE sentences describe tolerable variations.
    """
    text: str
    label: ConstraintLabel
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label.value,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Constraint":
        return cls(
            text=data["text"],
            label=ConstraintLabel(data["label"]),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class EvidenceSlot:
    """One semantic slot with its three constraint lists."""
    slot_type: SlotType
    required: List[Constraint] = field(default_factory=list)
    forbidden: List[Constraint] = field(default_factory=list)
    ignorable: List[Constraint] = field(default_factory=list)

    def add(self, text: str, label: ConstraintLabel, confidence: float = 1.0) -> None:
        constraint = Constraint(text=text, label=label, confidence=confidence)
        if label == ConstraintLabel.REQUIRED:
            self.required.append(constraint)
        elif label == ConstraintLabel.FORBIDDEN:
            self.forbidden.append(constraint)
        else:
            self.ignorable.append(constraint)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": [c.to_dict() for c in self.required],
            "forbidden": [c.to_dict() for c in self.forbidden],
            "ignore": [c.to_dict() for c in self.ignorable],
        }

    @classmethod
    def from_dict(cls, slot_type: SlotType, data: Dict[str, Any]) -> "EvidenceSlot":
        slot = cls(slot_type=slot_type)
        for item in data.get("required", []):
            if isinstance(item, str):
                slot.add(item, ConstraintLabel.REQUIRED)
            else:
                slot.add(item["text"], ConstraintLabel.REQUIRED, item.get("confidence", 1.0))
        for item in data.get("forbidden", []):
            if isinstance(item, str):
                slot.add(item, ConstraintLabel.FORBIDDEN)
            else:
                slot.add(item["text"], ConstraintLabel.FORBIDDEN, item.get("confidence", 1.0))
        for item in data.get("ignore", []):
            if isinstance(item, str):
                slot.add(item, ConstraintLabel.IGNORABLE)
            else:
                slot.add(item["text"], ConstraintLabel.IGNORABLE, item.get("confidence", 1.0))
        return slot


@dataclass
class EvidenceFrame:
    """
    Complete Evidence Frame E(q) for a query.

    Contains a global summary and six semantic slots, each with
    required/forbidden/ignorable constraints (Section 3.1).
    """
    global_summary: str = ""
    core_intent: str = ""
    expanded_query: str = ""
    confidence: float = 0.9
    slots: Dict[SlotType, EvidenceSlot] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure all six slots exist
        for st in SlotType:
            if st not in self.slots:
                self.slots[st] = EvidenceSlot(slot_type=st)

    def get_slot(self, slot_type: SlotType) -> EvidenceSlot:
        return self.slots[slot_type]

    def get_required_texts(self, slot_type: Optional[SlotType] = None) -> List[str]:
        """Get all required constraint texts, optionally filtered by slot."""
        if slot_type:
            return [c.text for c in self.slots[slot_type].required]
        texts = []
        for slot in self.slots.values():
            texts.extend(c.text for c in slot.required)
        return texts

    def get_forbidden_texts(self, slot_type: Optional[SlotType] = None) -> List[str]:
        """Get all forbidden constraint texts, optionally filtered by slot."""
        if slot_type:
            return [c.text for c in self.slots[slot_type].forbidden]
        texts = []
        for slot in self.slots.values():
            texts.extend(c.text for c in slot.forbidden)
        return texts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_summary": self.global_summary,
            "core_intent": self.core_intent,
            "expanded_query": self.expanded_query,
            "confidence": self.confidence,
            "entities": self.slots[SlotType.ENTITIES].to_dict(),
            "attributes": self.slots[SlotType.ATTRIBUTES].to_dict(),
            "actions": self.slots[SlotType.ACTIONS].to_dict(),
            "relations": self.slots[SlotType.RELATIONS].to_dict(),
            "scene": self.slots[SlotType.SCENE].to_dict(),
            "key_details": self.slots[SlotType.KEY_DETAILS].to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceFrame":
        frame = cls(
            global_summary=data.get("global_summary", ""),
            core_intent=data.get("core_intent", ""),
            expanded_query=data.get("expanded_query", ""),
            confidence=data.get("confidence", 0.9),
        )
        for st in SlotType:
            if st.value in data:
                slot_data = data[st.value]
                # Handle both list format (raw strings) and dict format
                if isinstance(slot_data, dict):
                    frame.slots[st] = EvidenceSlot.from_dict(st, slot_data)
        return frame

    def to_evidence_text(self) -> str:
        """
        Assemble evidence text for P-RANK prompt (Section I.5, Figure 9).

        Format:
            Core intent: ...
            Expanded query: ...
            Global summary: ...
            Slot constraints:
            [entities]
            required: sent1 | sent2 | ...
            forbidden: sent1 | sent2 | ...
            ...
        """
        lines = []
        lines.append(f"Core intent: {self.core_intent}")
        lines.append(f"Expanded query: {self.expanded_query}")
        lines.append(f"Global summary: {self.global_summary}")
        lines.append("Slot constraints:")

        for st in SlotType:
            slot = self.slots[st]
            lines.append(f"[{st.value}]")
            req_texts = [c.text for c in slot.required]
            forb_texts = [c.text for c in slot.forbidden]
            if req_texts:
                lines.append(f"required: {' | '.join(req_texts)}")
            if forb_texts:
                lines.append(f"forbidden: {' | '.join(forb_texts)}")
            # Note: ignore constraints are omitted from P-RANK (Section I.5)

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_req = sum(len(s.required) for s in self.slots.values())
        n_forb = sum(len(s.forbidden) for s in self.slots.values())
        n_ign = sum(len(s.ignorable) for s in self.slots.values())
        return (
            f"EvidenceFrame(required={n_req}, forbidden={n_forb}, "
            f"ignorable={n_ign}, summary='{self.global_summary[:50]}...')"
        )
