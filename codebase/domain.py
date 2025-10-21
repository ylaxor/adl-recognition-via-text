from dataclasses import dataclass
from datetime import datetime
from enum import Enum

type SensorName = str

type ActivityName = str


class TestBed(Enum):
    ORDA = "ordonez-a"
    ORDB = "ordonez-b"
    CASM = "casas-milan"
    CASA = "casas-aruba"


@dataclass(frozen=True)
class Event:
    identifier: str
    start: datetime
    end: datetime

    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp <= self.end

    def overlaps(self, other_start: datetime, other_end: datetime) -> bool:
        return max(self.start, other_start) < min(self.end, other_end)


class SensorActivation(Event):
    pass


class ActivityOccurrence(Event):
    pass


@dataclass(frozen=True)
class TriggerEvent:
    timestamp: datetime
    sensor_id: str
    sensor_value: bool


@dataclass
class Window:
    events: list[TriggerEvent]
    label: ActivityName


@dataclass(frozen=True)
class SensorMeta:
    sensor_type: str
    sensor_room: str
    sensor_object: str
