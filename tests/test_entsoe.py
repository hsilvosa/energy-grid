from datetime import UTC, datetime

from energy_grid.domain import EventType
from energy_grid.sources.entsoe import EntsoeClient, parse_entsoe_document

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


def test_price_request_uses_both_market_domains(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class Response:
        content = DOCUMENT.replace(b"<quantity>", b"<price.amount>").replace(
            b"</quantity>", b"</price.amount>"
        )

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, *, params: dict[str, str], timeout: float):
        captured.update(params)
        return Response()

    monkeypatch.setattr("energy_grid.sources.entsoe.httpx.get", fake_get)
    client = EntsoeClient("secret")
    client.download(
        EventType.PRICE,
        datetime(2025, 1, 1, tzinfo=UTC),
        datetime(2025, 1, 2, tzinfo=UTC),
    )
    assert captured["in_Domain"] == "10YES-REE------0"
    assert captured["out_Domain"] == "10YES-REE------0"
    assert "outBiddingZone_Domain" not in captured
