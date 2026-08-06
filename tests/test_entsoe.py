from datetime import UTC, datetime

from energy_grid.domain import EventType
from energy_grid.sources.entsoe import parse_entsoe_document

DOCUMENT = b"""<?xml version="1.0" encoding="UTF-8"?>
<GL_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0">
  <TimeSeries>
    <MktPSRType><psrType>B19</psrType></MktPSRType>
    <Period>
      <timeInterval><start>2025-01-01T00:00Z</start><end>2025-01-01T00:30Z</end></timeInterval>
      <resolution>PT15M</resolution>
      <Point><position>1</position><quantity>123.5</quantity></Point>
      <Point><position>2</position><quantity>124.0</quantity></Point>
    </Period>
  </TimeSeries>
</GL_MarketDocument>
"""


def test_entsoe_xml_is_converted_to_canonical_events() -> None:
    retrieved = datetime(2025, 1, 1, 1, tzinfo=UTC)
    events = list(parse_entsoe_document(DOCUMENT, EventType.GENERATION, retrieved_at=retrieved))
    assert len(events) == 2
    assert events[0].value == 123.5
    assert events[0].dimension == "B19"
    assert events[1].interval_start == datetime(2025, 1, 1, 0, 15, tzinfo=UTC)
    assert events[0].payload_checksum == events[1].payload_checksum

