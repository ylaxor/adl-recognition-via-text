from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Iterable, Mapping
from datetime import datetime
from enum import Enum

from num2words import num2words

from codebase.domain import SensorMeta, SensorName, Window


class TextualizerType(Enum):
    DTS = "dhekane_summary_single"
    CTS = "civitare_summary_single"
    TPB = "thukralb_summary_single"
    TPT = "thukralt_summary_single"
    CSS = "ncibi_summary_single"
    TSB = "thukralb_summary_sequence"
    TST = "thukralt_summary_sequence"
    CSS_NO_ROOM = "ncibi_summary_single_no_room"
    CSS_NO_TYPE = "ncibi_summary_single_no_type"
    CSS_NO_TIME = "ncibi_summary_single_no_time"
    CSS_NO_DURATION = "ncibi_summary_single_no_duration"
    CSS_NO_TOKENS = "ncibi_summary_single_no_tokens"


def delta_to_words(delta_seconds):
    if delta_seconds < 60:
        return f"{num2words(delta_seconds)} seconds"
    elif delta_seconds < 3600:
        minutes = delta_seconds // 60
        seconds = delta_seconds % 60
        if seconds == 0:
            return f"{num2words(minutes)} minutes"
        else:
            return f"{num2words(minutes)} minutes and {num2words(seconds)} seconds"
    else:
        hours = delta_seconds // 3600
        minutes = (delta_seconds % 3600) // 60
        seconds = delta_seconds % 60
        parts = []
        if hours > 0:
            parts.append(f"{num2words(hours)} {'hour' if hours == 1 else 'hours'}")
        if minutes > 0:
            parts.append(
                f"{num2words(minutes)} {'minute' if minutes == 1 else 'minutes'}"
            )
        if seconds > 0:
            parts.append(
                f"{num2words(seconds)} {'second' if seconds == 1 else 'seconds'}"
            )
        return " and ".join(parts)


class WindowTextualizer(ABC):
    def __init__(self):
        self.meta = None

    def setup(self, meta: Mapping[SensorName, SensorMeta]) -> "WindowTextualizer":
        self.meta = meta
        return self

    def describe(self, windows: Iterable[Window]) -> list[list[str]]:
        return [self.describe_window(win) for win in windows]

    @abstractmethod
    def describe_window(self, window: Window) -> list[str]: ...


class TDOSTSeqBasic(WindowTextualizer):
    def describe_window(self, window: Window) -> list[str]:
        assert self.meta is not None, "not set up with metadata."
        outputs = []
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if meta is None:
                raise ValueError(f"Unknown sensor ID: {event.sensor_id}")
            sensor_type = meta.sensor_type
            sensor_room = meta.sensor_room
            sensor_object = meta.sensor_object
            sen_value = "ON" if event.sensor_value else "OFF"
            outputs.append(
                f"A {sensor_type} sensor placed on the {sensor_object} in the {sensor_room} fired with value {sen_value}."
            )
        return outputs


class TDOSTSeqTemporal(WindowTextualizer):
    def describe_window(self, window: Window) -> list[str]:
        assert self.meta is not None, "not set up with metadata."
        outputs = []
        for i, event in enumerate(window.events):
            meta = self.meta.get(event.sensor_id)
            if meta is None:
                raise ValueError(f"Unknown sensor ID: {event.sensor_id}")
            sen_type = meta.sensor_type
            sensor_room = meta.sensor_room
            sen_value = "ON" if event.sensor_value else "OFF"
            sen_timestamp = event.timestamp
            sen_obect = meta.sensor_object
            if i == 0:
                info_first = self.time_to_words(sen_timestamp.strftime("%H:%M:%S"))
                s = f"A {sen_type} sensor placed on the {sen_obect} in the {sensor_room} fired with value {sen_value} {info_first}."
            else:
                prev_timestamp = window.events[i - 1].timestamp
                delta_seconds = int((sen_timestamp - prev_timestamp).total_seconds())
                info_next = f"After {delta_to_words(delta_seconds)}"
                s = f"{info_next} a {sen_type} sensor placed on the {sen_obect} in the {sensor_room} fired with value {sen_value}."
            outputs.append(s)
        return outputs

    def time_to_words(self, time_str):
        time_obj = datetime.strptime(time_str, "%H:%M:%S")
        hour = time_obj.hour
        minute = time_obj.minute
        second = time_obj.second
        if hour == 0:
            hour_12 = 12
            period = "AM"
        elif hour < 12:
            hour_12 = hour
            period = "AM"
        elif hour == 12:
            hour_12 = 12
            period = "PM"
        else:
            hour_12 = hour - 12
            period = "PM"
        parts = [f"at {num2words(hour_12)} {'hour' if hour_12 == 1 else 'hours'}"]
        if minute > 0:
            parts.append(
                f"{num2words(minute)} {'minute' if minute == 1 else 'minutes'}"
            )
        if second > 0:
            parts.append(
                f"{num2words(second)} {'second' if second == 1 else 'seconds'}"
            )
        if len(parts) == 1:
            result = parts[0]
        elif len(parts) == 2:
            result = f"{parts[0]} and {parts[1]}"
        else:
            result = f"{parts[0]} {parts[1]} and {parts[2]}"
        return f"{result} {period}"


class TDOSTPlainBasic(WindowTextualizer):
    def describe_window(self, window: Window) -> list[str]:
        assert self.meta is not None, "not set up with metadata."
        seqel_outputs = TDOSTSeqBasic().setup(self.meta).describe_window(window)
        concatenated = " ".join(seqel_outputs)
        outputs = [concatenated]
        return outputs


class TDOSTPlainTemporal(WindowTextualizer):
    def describe_window(self, window: Window) -> list[str]:
        assert self.meta is not None, "not set up with metadata."
        seqel_outputs = TDOSTSeqTemporal().setup(self.meta).describe_window(window)
        concatenated = " ".join(seqel_outputs)
        outputs = [concatenated]
        return outputs


class CivitareseTextualSummary(WindowTextualizer):
    def build_states(self, events) -> dict:
        events_by_sensor = {}
        for event in events:
            if event.sensor_id not in events_by_sensor:
                events_by_sensor[event.sensor_id] = []
            events_by_sensor[event.sensor_id].append(event)
        states_by_sensor = {}
        for sensor_id, events in events_by_sensor.items():
            states = []
            current_state = None
            current_start = None
            for event in events:
                if event.sensor_value:
                    if current_state is None:
                        current_state = "ON"
                        current_start = event.timestamp
                else:
                    if current_state == "ON":
                        states.append(
                            {
                                "state": "ON",
                                "start": current_start,
                                "end": event.timestamp,
                            }
                        )
                        current_state = None
                        current_start = None
            if current_state == "ON" and current_start is not None:
                states.append(
                    {
                        "state": "ON",
                        "start": current_start,
                        "end": events[-1].timestamp,
                    }
                )
            states_by_sensor[sensor_id] = states
        return states_by_sensor

    def build_temporal(self, window: Window) -> str:
        start_time = window.events[0].timestamp
        end_time = window.events[-1].timestamp
        duration = int((end_time - start_time).total_seconds())
        time_str = start_time.strftime("%-I:%M %p").lower()
        output = f"During a {duration}-second time window (around {time_str}), the system observed the following. "
        return output

    def describe_action(self, meta: SensorMeta) -> str:
        sensor_type = meta.sensor_type.lower()
        sensor_object = meta.sensor_object.lower()
        if sensor_type == "motion":
            return f"get around the {sensor_object}"
        elif sensor_type == "contact":
            return f"open the {sensor_object}"
        elif sensor_type == "flush":
            return f"flushe the {sensor_object}"
        elif sensor_type == "pressure":
            return f"stay on the {sensor_object}"
        elif sensor_type == "electric":
            return f"turn on the {sensor_object}"
        else:
            return f"interact with the {sensor_type} sensor"

    def describe_window(self, window: Window) -> list[str]:
        assert self.meta is not None, "not set up with metadata."
        if not window.events:
            return ["No activity detected"]
        output = self.build_temporal(window)
        seg_by_room = self.segment_by_room(window)
        for j, room_seg in enumerate(seg_by_room):
            if j == 0 and len(seg_by_room) > 1:
                outer_link = "First, the subject is in the"
            elif j == 0 and len(seg_by_room) == 1:
                outer_link = "The subject is in the"
            else:
                outer_link = "Then, they go to the"
            room_action_desc = f"{outer_link} {room_seg['room']}."
            room_actions = [
                (action, sensor)
                for sensor, sensor_actions in self.build_states(
                    room_seg["events"]
                ).items()
                for action in sensor_actions
            ]
            room_actions = sorted(room_actions, key=lambda x: x[0]["start"])
            for i, f in enumerate(room_actions):
                action = f[0]
                sensor = f[1]
                if i == 0 and len(room_actions) > 1:
                    innder_link = "Here, they"
                elif i == 0 and len(room_actions) == 1:
                    innder_link = "They"
                else:
                    innder_link = "Then, they"
                action_duration = int((action["end"] - action["start"]).total_seconds())
                action_description = self.describe_action(self.meta[sensor])
                room_action_desc += (
                    " "
                    + innder_link
                    + " "
                    + f" {action_description} for {action_duration} seconds. "
                )
            output += room_action_desc + " "
        return [output.strip()]

    def segment_by_room(self, window: Window) -> list[dict]:
        assert self.meta is not None, "not set up with metadata."
        segments = []
        current_room = None
        current_start_time = None
        current_events = []
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if not meta:
                continue
            room = meta.sensor_room.lower()
            if room != current_room:
                if current_room is not None:
                    assert current_start_time is not None
                    duration = int(
                        (event.timestamp - current_start_time).total_seconds()
                    )
                    segments.append(
                        {
                            "room": current_room,
                            "duration": duration,
                            "events": current_events,
                        }
                    )
                current_room = room
                current_start_time = event.timestamp
                current_events = []
            current_events.append(event)
        if current_room is not None:
            assert current_start_time is not None
            duration = int(
                (window.events[-1].timestamp - current_start_time).total_seconds()
            )
            segments.append(
                {
                    "room": current_room,
                    "duration": duration,
                    "events": current_events,
                }
            )
        return segments


class DhekaneTextualSummary(WindowTextualizer):
    def __init__(self, attributes: list[str] | None = None):
        super().__init__()
        if attributes is None:
            attributes = [
                "time_of_occurrence",
                "duration",
                "top_k_rooms",
                "top_k_sensors",
            ]
        valid_attributes = {
            "time_of_occurrence",
            "duration",
            "top_k_rooms",
            "top_k_sensors",
        }
        for attr in attributes:
            if attr not in valid_attributes:
                raise ValueError(
                    f"Invalid attribute '{attr}'. Choose from: {valid_attributes}"
                )
        self.attributes = attributes

    def describe_window(self, window: Window) -> list[str]:
        descriptions = []
        for attr in self.attributes:
            method = getattr(self, attr)
            descriptions.append(method(window))
        outputs = [". ".join(descriptions) + "."]
        return outputs

    def time_of_occurrence(self, window: Window) -> str:
        def period_of_day(hour: int) -> str:
            if 5 <= hour < 12:
                return "in the morning"
            elif 12 <= hour < 17:
                return "in the afternoon"
            elif 17 <= hour < 21:
                return "in the evening"
            else:
                return "at night"

        start_time = window.events[0].timestamp
        end_time = window.events[-1].timestamp
        start_hour_part = start_time.strftime("%I:%M %p").lstrip("0")
        end_hour_part = end_time.strftime("%I:%M %p").lstrip("0")
        start_period_part = period_of_day(start_time.hour)
        end_period_part = period_of_day(end_time.hour)
        return f"The activity started at {start_hour_part} {start_period_part} and ended at {end_hour_part} {end_period_part}"

    def duration(self, window: Window) -> str:
        duration_seconds = (
            window.events[-1].timestamp - window.events[0].timestamp
        ).total_seconds()
        if duration_seconds < 60:
            return f"The activity was performed for {int(duration_seconds)} seconds"
        elif duration_seconds < 3600:
            minutes = duration_seconds // 60
            return f"The activity was performed for {int(minutes)} minutes"
        else:
            hours = duration_seconds // 3600
            return f"The activity was performed for {int(hours)} hours"

    def top_k_rooms(self, window: Window, k: int = 3) -> str:
        assert self.meta is not None, "not set up with metadata."
        room_counter = Counter()
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if meta:
                room_counter[meta.sensor_room.lower()] += 1
        most_common = room_counter.most_common(k)
        if not most_common:
            return "No rooms detected"
        rooms = [room for room, _ in most_common]
        if len(rooms) == 1:
            return f"The activity is taking place mainly in the {rooms[0]}"
        else:
            return f"The activity is taking place mainly in the {rooms[0]} and parts of it in {', '.join(rooms[1:])}"

    def top_k_sensors(self, window: Window, k: int = 3) -> str:
        assert self.meta is not None, "not set up with metadata."
        sensor_counter = Counter()
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if meta:
                sensor_counter[
                    (
                        meta.sensor_type,
                        meta.sensor_room.lower(),
                        meta.sensor_object.lower(),
                    )
                ] += 1
        most_common = sensor_counter.most_common(k)
        if not most_common:
            return "No sensors detected"
        sensors = [
            f"the {stype} sensor placed on the {sobj} in the {sroom}"
            for (stype, sroom, sobj), _ in most_common
        ]
        if len(sensors) == 1:
            return f"The most commonly fired sensor in this activity is {sensors[0]}"
        else:
            return f"The two most commonly fired sensors in this activity are {sensors[0]} and {sensors[1]}"


class CompoundTextualSummary(WindowTextualizer):
    def __init__(self, attributes: list[str] | None = None):
        super().__init__()
        if attributes is None:
            attributes = [
                "unique_types",
                "unique_rooms",
                "token_sequence",
                "duration",
                "time_of_occurrence",
            ]
        valid_attributes = {
            "unique_types",
            "unique_rooms",
            "token_sequence",
            "duration",
            "time_of_occurrence",
        }
        for attr in attributes:
            if attr not in valid_attributes:
                raise ValueError(
                    f"Invalid attribute '{attr}'. Choose from: {valid_attributes}"
                )
        self.attributes = attributes

    def setup(self, meta: Mapping[SensorName, SensorMeta]) -> "CompoundTextualSummary":
        self.meta = meta
        return self

    def describe_window(self, window: Window) -> list[str]:
        descriptions = []
        for attr in self.attributes:
            method = getattr(self, attr)
            descriptions.append(method(window))
        outputs = [" :: ".join(descriptions) + "."]
        return outputs

    def time_of_occurrence(self, window: Window) -> str:
        def period_of_day(hour: int) -> str:
            if 5 <= hour < 12:
                return "in the morning"
            elif 12 <= hour < 17:
                return "in the afternoon"
            elif 17 <= hour < 21:
                return "in the evening"
            else:
                return "at night"

        start_time = window.events[0].timestamp
        end_time = window.events[-1].timestamp
        start_hour_part = start_time.strftime("%I:%M %p").lstrip("0")
        end_hour_part = end_time.strftime("%I:%M %p").lstrip("0")
        start_period_part = period_of_day(start_time.hour)
        end_period_part = period_of_day(end_time.hour)
        return f"OCCURRENCE: from {start_hour_part} {start_period_part} to {end_hour_part} {end_period_part}"

    def duration(self, window: Window) -> str:
        duration_seconds = window.events[-1].timestamp - window.events[0].timestamp
        return f"DURATION: {delta_to_words(int(duration_seconds.total_seconds()))}"

    def token_sequence(self, window: Window) -> str:
        assert self.meta is not None, "not set up with metadata."
        tokens = []
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if meta:
                sensor_object = meta.sensor_object.lower()
                sensor_room = meta.sensor_room.lower()
                sensor_type = meta.sensor_type.lower()
                sensor_value = "ON" if event.sensor_value else "OFF"
                token = f"{sensor_object}@{sensor_room}({sensor_type})={sensor_value}"
                tokens.append(token)
        if not tokens:
            return "EVENTS: None"
        else:
            return f"EVENTS: {', '.join(tokens)}"

    def unique_types(self, window: Window) -> str:
        assert self.meta is not None, "not set up with metadata."
        types = set()
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if meta:
                types.add(meta.sensor_type.lower())
        if not types:
            return "TYPES: None"
        else:
            return f"TYPES: {'|'.join(sorted(types))}"

    def unique_rooms(self, window: Window) -> str:
        assert self.meta is not None, "not set up with metadata."
        rooms = set()
        for event in window.events:
            meta = self.meta.get(event.sensor_id)
            if meta:
                rooms.add(meta.sensor_room.lower())
        if not rooms:
            return "ROOMS: None"
        else:
            return f"ROOMS: {'|'.join(sorted(rooms))}"


class CompoundTextualSummaryNoRoom(CompoundTextualSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "token_sequence",
                "duration",
                "time_of_occurrence",
            ]
        )


class CompoundTextualSummaryNoType(CompoundTextualSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_rooms",
                "token_sequence",
                "duration",
                "time_of_occurrence",
            ]
        )


class CompoundTextualSummaryNoTime(CompoundTextualSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "unique_rooms",
                "token_sequence",
                "duration",
            ]
        )


class CompoundTextualSummaryNoDuration(CompoundTextualSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "unique_rooms",
                "token_sequence",
                "time_of_occurrence",
            ]
        )


class CompoundTextualSummaryNoTokens(CompoundTextualSummary):
    def __init__(self):
        super().__init__(
            attributes=[
                "unique_types",
                "unique_rooms",
                "duration",
                "time_of_occurrence",
            ]
        )


class WindowTextualizerFactory:
    _registry = {
        "thukralb_summary_sequence": TDOSTSeqBasic,
        "thukralt_summary_sequence": TDOSTSeqTemporal,
        "thukralb_summary_single": TDOSTPlainBasic,
        "thukralt_summary_single": TDOSTPlainTemporal,
        "civitare_summary_single": CivitareseTextualSummary,
        "dhekane_summary_single": DhekaneTextualSummary,
        "ncibi_summary_single": CompoundTextualSummary,
        "ncibi_summary_single_no_room": CompoundTextualSummaryNoRoom,
        "ncibi_summary_single_no_type": CompoundTextualSummaryNoType,
        "ncibi_summary_single_no_time": CompoundTextualSummaryNoTime,
        "ncibi_summary_single_no_duration": CompoundTextualSummaryNoDuration,
        "ncibi_summary_single_no_tokens": CompoundTextualSummaryNoTokens,
    }

    @classmethod
    def create(cls, name: str, **kwargs) -> WindowTextualizer:
        key = name.strip().lower()
        if key not in cls._registry:
            raise ValueError(
                f"Unknown  '{name}'. Choose from: {', '.join(cls._registry)}"
            )
        return cls._registry[key](**kwargs)
