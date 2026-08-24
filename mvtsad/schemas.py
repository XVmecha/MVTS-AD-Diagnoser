"""External data contracts: answer key, evidence bundle, diagnosis claims.

These pydantic models define the JSON artifacts that cross component
boundaries (generator to checker, extractor to diagnoser, diagnoser to
checker). The checker depends only on this module, never on mvtsad.gen or
mvtsad.extract, so the contracts can move into the standalone checker
library at split time.

Vocabulary is domain-neutral per design.md section 2: channel, system,
scene. The design doc's example claims JSON uses the key "sensor"; the
terminology section mandates "channel" throughout code and schema, so
"channel" wins here (deviation logged in fyi.md).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class FaultClass(str, Enum):
    DRIFT = "drift"
    SPIKE = "spike"
    LEVEL_SHIFT = "level_shift"
    STUCK_AT = "stuck_at"
    VARIANCE_CHANGE = "variance_change"
    OSCILLATION = "oscillation"
    CORRELATION_BREAK = "correlation_break"


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"


# ---------------------------------------------------------------------------
# Answer key (the injection record, written by the generator)
# ---------------------------------------------------------------------------


class CauseLink(BaseModel):
    """One causal edge through which a deviation arrived at a channel."""

    parent: int
    lag: int


class RootEvent(BaseModel):
    """An injected fault: the intervention itself.

    window is [start, end), timestep indices. magnitude_z is the injected
    magnitude in units of the channel's noise sigma; None for classes where
    magnitude is not additive (stuck_at, variance_change, correlation_break).
    """

    channel: int
    fault_class: FaultClass
    window: tuple[int, int]
    direction: Direction | None = None
    magnitude_z: float | None = None
    params: dict[str, float | int | str] = Field(default_factory=dict)


class InducedEvent(BaseModel):
    """A downstream deviation produced by propagation, never injected.

    Derived from the divergence between the faulted run and its
    counterfactual clean twin. causes lists every incoming edge from a
    deviated parent (or the root); a claim's scalar caused_by matches if it
    names any of them. depth is graph distance from the root.
    """

    channel: int
    window: tuple[int, int]
    causes: list[CauseLink]
    depth: int


class AnswerKey(BaseModel):
    scene_id: str
    split: str
    wiring_id: int
    n_channels: int
    n_timesteps: int
    roots: list[RootEvent]
    induced: list[InducedEvent]
    # Legal operating-mode changes (timestep indices). Not faults; must not
    # be claimed.
    regime_switches: list[int]
    clean_channels: list[int]
    difficulty: dict[str, float | int | str | bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evidence bundle (written by the extractor, read by the diagnoser)
# ---------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """One flag in the event stream: the neutral tuple from design.md
    section 3, (source, channels, window, severity, direction).

    id is the citable handle claims reference (e.g. "thr:s4@412").
    source identifies the emitting detector family (threshold, changepoint,
    slope, xcorr, detector). channels lists one channel for univariate flags,
    two for cross-channel flags. Point events use window (t, t+1).
    """

    id: str
    source: str
    channels: list[int]
    window: tuple[int, int]
    severity: float | None = None
    direction: Direction | None = None


class EvidenceBundle(BaseModel):
    scene_id: str
    n_channels: int
    items: list[EvidenceItem]


# ---------------------------------------------------------------------------
# Diagnosis (the model's structured output)
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """One structured claim. claim_class serializes as "class".

    For root claims: claim_class is a FaultClass value or "unknown" (the
    escape class, design.md section 5), role is "root".
    For induced claims: claim_class is "induced", caused_by and lag are set.
    """

    model_config = ConfigDict(populate_by_name=True)

    channel: int
    window: tuple[int, int]
    claim_class: str = Field(alias="class")
    role: str | None = None
    direction: Direction | None = None
    magnitude_z: float | None = None
    caused_by: int | None = None
    lag: int | None = None
    evidence: list[str] = Field(default_factory=list)


class Diagnosis(BaseModel):
    claims: list[Claim]
    clean: list[int]
