from __future__ import annotations

import json
import random
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol

from energy_grid.domain import GridEvent

TOPICS = {
    "demand": "grid.demand.v1",
    "generation": "grid.generation.v1",
    "price": "market.prices.v1",
    "weather": "weather.forecasts.v1",
}
DEAD_LETTER_TOPIC = "energy.dead-letter.v1"


class Producer(Protocol):
    def send(self, topic: str, key: bytes, value: bytes) -> None: ...

    def flush(self) -> None: ...


class MemoryProducer:
    def __init__(self) -> None:
        self.messages: list[tuple[str, bytes, bytes]] = []

    def send(self, topic: str, key: bytes, value: bytes) -> None:
        self.messages.append((topic, key, value))

    def flush(self) -> None:
        return None


class KafkaProducerAdapter:
    def __init__(self, bootstrap_servers: str) -> None:
        try:
            from confluent_kafka import Producer as ConfluentProducer
        except ImportError as exc:
            raise RuntimeError("install the 'streaming' extra to publish to Kafka") from exc
        self._producer = ConfluentProducer({"bootstrap.servers": bootstrap_servers})

    def send(self, topic: str, key: bytes, value: bytes) -> None:
        self._producer.produce(topic, key=key, value=value)
        self._producer.poll(0)

    def flush(self) -> None:
        self._producer.flush()


@dataclass(frozen=True)
class ReplayProfile:
    seed: int = 42
    duplicate_rate: float = 0.0
    missing_rate: float = 0.0
    late_rate: float = 0.0
    corrupt_rate: float = 0.0
    max_delay_minutes: int = 240
    speed: float = 0.0


def load_fixture(path: Path) -> list[GridEvent]:
    events: list[GridEvent] = []
    with path.open(encoding="utf-8") as fixture:
        for line_number, line in enumerate(fixture, start=1):
            if not line.strip():
                continue
            try:
                events.append(GridEvent.model_validate(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid fixture at line {line_number}: {exc}") from exc
    return events


def mutate_events(events: Iterable[GridEvent], profile: ReplayProfile) -> list[GridEvent | bytes]:
    rng = random.Random(profile.seed)
    output: list[GridEvent | bytes] = []
    for event in events:
        if rng.random() < profile.missing_rate:
            continue
        changed = event
        if rng.random() < profile.late_rate:
            changed = event.model_copy(
                update={
                    "ingested_at": event.ingested_at
                    + timedelta(minutes=rng.randint(121, profile.max_delay_minutes))
                }
            )
        output.append(changed)
        if rng.random() < profile.duplicate_rate:
            output.append(changed)
        if rng.random() < profile.corrupt_rate:
            output.append(b'{"schema_version":"broken"')
    rng.shuffle(output)
    return output


def replay(
    events: Iterable[GridEvent],
    producer: Producer,
    profile: ReplayProfile,
    on_publish: Callable[[str], None] | None = None,
) -> int:
    published = 0
    for item in mutate_events(events, profile):
        if isinstance(item, bytes):
            producer.send(DEAD_LETTER_TOPIC, b"corrupt", item)
            topic = DEAD_LETTER_TOPIC
        else:
            topic = TOPICS[item.event_type.value]
            producer.send(topic, item.business_key.encode(), item.to_message())
        published += 1
        if on_publish:
            on_publish(topic)
        if profile.speed > 0:
            time.sleep(1 / profile.speed)
    producer.flush()
    return published
