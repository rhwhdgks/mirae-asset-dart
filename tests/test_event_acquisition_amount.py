"""Keep early-acquisition consideration separate from the bond face value."""

from types import SimpleNamespace

from app.tools.events import EventTool


class _FieldIndex:
    def __init__(self, rows):
        self._rows = rows

    def rows(self, _corp_code, *, as_of, rcept_no):
        assert as_of == "20260619"
        assert rcept_no == "20260310002968"
        return self._rows

    @staticmethod
    def verified(row):
        return row

    def lookup(self, *_args, **_kwargs):
        raise AssertionError("The exact acquisition path must resolve first")


def test_acquisition_amount_is_not_bond_face_value():
    parent = "6. 매도 대상 사채의 만기전 취득내역 > "
    rows = [
        SimpleNamespace(
            path=parent + "매도 대상 사채의 권면(전자등록)금액(원)",
            value="4,800,000,000", occurrence=1, order=1,
            evidence_status="verified"),
        SimpleNamespace(
            path=parent + "취득금액(원)",
            value="5,007,633,600", occurrence=1, order=2,
            evidence_status="verified"),
    ]
    tool = EventTool(object(), fidx=_FieldIndex(rows))
    row, status = tool.field_value(
        "00231044", "20260310002968", "취득금액",
        as_of="20260619", trace=[])
    assert status == "verified"
    assert row is rows[1]
