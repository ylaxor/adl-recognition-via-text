from collections.abc import Mapping, Sequence
from csv import DictReader
from datetime import datetime, timedelta
from pathlib import Path

from toml import load as load_toml

from codebase.domain import (
    ActivityName,
    ActivityOccurrence,
    Event,
    SensorActivation,
    SensorMeta,
    SensorName,
    TestBed,
)

config = load_toml(Path(__file__).parent.parent / "config" / "data.toml")


def _read_file(
    dataset: TestBed,
    what: str = "sensors",
) -> Sequence[Event]:
    assert what in {"sensors", "activities"}, "what must be 'sensors' or 'activities'"
    directory = Path(__file__).parent.parent / "testbed"
    path = directory / dataset.value / f"{what}.csv"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, mode="r", newline="", encoding="utf-8") as csvfile:
        return [
            Event(
                identifier=row["name"],
                start=datetime.fromisoformat(row["start"]),
                end=datetime.fromisoformat(row["end"]),
            )
            for row in DictReader(csvfile)
        ]


def _filter_events_by_days(
    events: Sequence[Event], days_to_include: int
) -> Sequence[Event]:
    earliest_start = min(event.start for event in events)
    cutoff_time = earliest_start + timedelta(days=days_to_include)
    return [event for event in events if event.end <= cutoff_time]


def read_dataset(
    dataset: TestBed,
    days_to_include: int | None = None,
    simplify_sensors: bool = False,
    simplify_activities: bool = False,
    rename_activities: bool = False,
) -> tuple[Sequence[SensorActivation], Sequence[ActivityOccurrence]]:
    return read_sensor_activations(
        dataset,
        days_to_include,
        simplify_sensors,
    ), read_activity_occurrences(
        dataset,
        days_to_include,
        simplify_activities,
        rename_activities,
    )


def read_sensor_activations(
    dataset: TestBed,
    days_to_include: int | None = None,
    simplify_sensors: bool = False,
) -> Sequence[SensorActivation]:
    events = _read_file(dataset, what="sensors")
    sensors = config["sensors"][str(dataset)]
    if sensors and simplify_sensors:
        events = [event for event in events if event.identifier in sensors]
    sensor_activations = [SensorActivation(**event.__dict__) for event in events]

    if days_to_include is not None:
        filtered_events = _filter_events_by_days(sensor_activations, days_to_include)
        return [SensorActivation(**event.__dict__) for event in filtered_events]

    return sensor_activations


def read_activity_occurrences(
    dataset: TestBed,
    days_to_include: int | None = None,
    simplify_activities: bool = False,
    rename_activities: bool = False,
) -> Sequence[ActivityOccurrence]:
    events = _read_file(dataset, what="activities")
    activities = config["activities"][str(dataset)]
    activity_renames = config.get("activity_renames", {})
    if activities and simplify_activities:
        events = [event for event in events if event.identifier in activities]
    if days_to_include is not None:
        events = _filter_events_by_days(events, days_to_include)
    for event in events:
        if event.identifier in activity_renames and rename_activities:
            object.__setattr__(event, "identifier", activity_renames[event.identifier])
        else:
            object.__setattr__(event, "identifier", event.identifier)
    return [ActivityOccurrence(**event.__dict__) for event in events]


def read_sensors_metadata(
    dataset: TestBed,
) -> Mapping[SensorName, SensorMeta]:
    directory = Path(__file__).parent.parent / "testbed"
    path = directory / dataset.value / "meta.toml"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    toml_data = load_toml(path)
    sensor_metadata = toml_data.get("sensors", {}).get("metadata", {})
    return {
        sensor_name: SensorMeta(
            sensor_type=metadata.get("type", ""),
            sensor_room=metadata.get("room", ""),
            sensor_object=metadata.get("object", ""),
        )
        for sensor_name, metadata in sensor_metadata.items()
    }


def read_activities_metadata(
    dataset: TestBed,
) -> Mapping[ActivityName, dict[str, str]]:
    directory = Path(__file__).parent.parent / "testbed"
    path = directory / dataset.value / "meta.toml"
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    toml_data = load_toml(path)
    activity_metadata = toml_data.get("activities", {}).get("metadata", {})
    return {
        activity_name: dict(metadata)
        for activity_name, metadata in activity_metadata.items()
    }
