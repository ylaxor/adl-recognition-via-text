from collections.abc import Sequence
from datetime import datetime

from codebase.domain import ActivityOccurrence, SensorActivation, TriggerEvent, Window


def intervals_overlap(
    start1: datetime,
    end1: datetime,
    start2: datetime,
    end2: datetime,
) -> float:
    latest_start = max(start1, start2)
    earliest_end = min(end1, end2)
    delta = (earliest_end - latest_start).total_seconds()
    return max(0, delta)


def belongs_to_interval(
    t: datetime,
    start: datetime,
    end: datetime,
) -> bool:
    return start < t <= end


def get_ground_truth_windows(
    sensor_activations: Sequence[SensorActivation],
    activity_occurrences: Sequence[ActivityOccurrence],
    use_idle: bool = False,
) -> list[Window]:
    windows = []
    sen_activations_sorted = sorted(sensor_activations, key=lambda s: (s.start, s.end))
    act_occurrences_sorted = sorted(
        activity_occurrences, key=lambda a: (a.start, a.end)
    )
    for idx, activity in enumerate(act_occurrences_sorted):
        win_start = activity.start
        win_end = activity.end
        if win_end <= win_start:
            continue
        win_events = []
        for sa in sen_activations_sorted:
            overlap = intervals_overlap(sa.start, sa.end, win_start, win_end)
            if overlap <= 0:
                continue
            clip_start = max(sa.start, win_start)
            clip_end = min(sa.end, win_end)
            assert clip_end > clip_start
            win_events.append(
                TriggerEvent(
                    timestamp=clip_start, sensor_id=sa.identifier, sensor_value=True
                )
            )
            win_events.append(
                TriggerEvent(
                    timestamp=clip_end, sensor_id=sa.identifier, sensor_value=False
                )
            )
        win_events.sort(key=lambda x: (x.timestamp, not x.sensor_value, x.sensor_id))
        if not win_events:
            continue
        windows.append(Window(events=win_events, label=activity.identifier))
    if not use_idle:
        windows = [w for w in windows if w.label != "OTHER"]
    return windows
    return windows
