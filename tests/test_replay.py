from pathlib import Path

from energy_grid.replay import (
    DEAD_LETTER_TOPIC,
    MemoryProducer,
    ReplayProfile,
    load_fixture,
    replay,
)

FIXTURE = Path("data/fixtures/events.jsonl")


def test_fixture_is_valid_and_contains_revision() -> None:
    events = load_fixture(FIXTURE)
    assert len(events) == 6
    assert any(event.revision == 2 for event in events)


def test_fault_profile_is_deterministic() -> None:
    events = load_fixture(FIXTURE)
    first = MemoryProducer()
    second = MemoryProducer()
    profile = ReplayProfile(seed=7, duplicate_rate=1, late_rate=1, corrupt_rate=1)
    assert replay(events, first, profile) == len(events) * 3
    assert replay(events, second, profile) == len(events) * 3
    assert first.messages == second.messages
    assert sum(topic == DEAD_LETTER_TOPIC for topic, _, _ in first.messages) == len(events)


def test_missing_profile_can_drop_every_event() -> None:
    producer = MemoryProducer()
    count = replay(load_fixture(FIXTURE), producer, ReplayProfile(missing_rate=1))
    assert count == 0
    assert producer.messages == []

