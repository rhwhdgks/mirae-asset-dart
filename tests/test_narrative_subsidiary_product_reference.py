"""Offline regressions for explicitly referenced subsidiary product details."""

from types import SimpleNamespace

import pytest

from app.tools.narrative import (
    _authority_roundtripped_text,
    _business_axis_score,
    _exact_document_authority_hits,
    _referenced_holding_product_role_chunk_ids,
    _referenced_subsidiary_product_chunk_ids,
    _strip_prompt_boundaries,
)


_REFERENCE = (
    "가. 당사 주요 제품 및 서비스 당사는 자회사 등에 대하여 지배와 투자, "
    "경영자문 등을 제공하고 있으며, 임대사업 등을 진행하고 있습니다.\n"
    "나. 종속기업의 주요 제품 및 서비스 주요종속회사의 주요 제품 및 서비스에 관한 사항은 "
    "'II. 사업의 내용의 7. 기타 참고사항'을 참고하시기 바랍니다."
)

_SUBSIDIARY_PROFILE = """가. 시장 전망
시장 전망에 관한 설명은 사업 설명과 구분합니다.

다. 주요종속회사 사업의 내용

(주)예시금속

1. 사업의 개요
당사는 특수강을 생산하고 판매하는 철강 전문회사입니다.

2. 주요 제품 및 서비스
| 제품 | 설명 |
|---|---|
| 특수강봉강 | 특수 용도의 탄소강과 합금강을 봉 형태로 만든 제품 |

3. 원재료 및 생산설비
이 내용은 제품 설명에 포함하지 않습니다.
"""


def _chunks(reference_text: str):
    return (
        SimpleNamespace(
            chunk_id="issuer-products", doc_id="periodic_20260515002787",
            source_file_id="source-file", rcept_dt="20260601",
            section_id="section-products",
            path="II. 사업의 내용 > 2. 주요 제품 및 서비스",
            locator="part[0]", text_prompt_safe=reference_text,
        ),
        SimpleNamespace(
            chunk_id="subsidiary-products", doc_id="periodic_20260515002787",
            source_file_id="source-file", rcept_dt="20260601",
            section_id="section-other", path="II. 사업의 내용 > 7. 기타 참고사항",
            locator="part[1]", text_prompt_safe=_SUBSIDIARY_PROFILE,
        ),
    )


def test_broad_business_query_can_follow_only_an_explicit_same_report_reference():
    rows = _chunks(_REFERENCE)
    product_row = rows[1]
    task = SimpleNamespace(
        # Stage1 gives the narrative cell its axis topic; the issuer name is
        # already carried separately as expected_corp_name.
        retrieval_query="주요 사업 내용",
        expected_corp_name="예시지주회사",
    )

    assert _referenced_subsidiary_product_chunk_ids(rows) == {
        "subsidiary-products"
    }
    assert _referenced_holding_product_role_chunk_ids(rows) == {
        "issuer-products"
    }

    exact_task = SimpleNamespace(
        retrieval_query="주요 사업 내용", expected_corp_name="예시지주회사",
        corp_codes=("00138731",), as_of="20260619",
        document_selector=SimpleNamespace(
            rcept_no="20260515002787", doc_id="periodic_20260515002787",
            doc_group="periodic"),
    )
    hits = _exact_document_authority_hits(
        SimpleNamespace(chunks=lambda **_kwargs: rows),
        exact_task, "business_axis", rows=rows)
    hit_map = {hit.chunk_id: hit for hit in hits}
    assert hit_map["issuer-products"].holding_product_role_reference_verified
    assert hit_map["subsidiary-products"].subsidiary_products_reference_verified

    assert _business_axis_score(task, product_row) == 0
    assert _business_axis_score(
        task, product_row, allow_subsidiary_products=True) > 0
    assert _business_axis_score(
        task, rows[0], allow_holding_product_role=True) > 0

    role_hit = SimpleNamespace(
        chunk_id="issuer-products", section_id="section-products",
        path=rows[0].path, text_prompt_safe=rows[0].text_prompt_safe,
        holding_product_role_reference_verified=True,
    )
    role_text = _authority_roundtripped_text(
        task, role_hit, rows[0], rows[0].text_prompt_safe, "business_axis")
    assert role_text is not None
    role_body = _strip_prompt_boundaries(role_text)
    assert "지배와 투자, 경영자문" in role_body
    assert "임대사업" in role_body
    assert "참고하시기 바랍니다" not in role_body

    product_hit = SimpleNamespace(
        chunk_id="subsidiary-products", section_id=product_row.section_id,
        path=product_row.path, text_prompt_safe=product_row.text_prompt_safe,
        subsidiary_products_reference_verified=True,
    )
    product_text = _authority_roundtripped_text(
        task, product_hit, product_row, product_row.text_prompt_safe,
        "business_axis")
    assert product_text is not None
    body = _strip_prompt_boundaries(product_text)
    assert body.startswith("(주)예시금속\n\n1. 사업의 개요")
    assert "2. 주요 제품 및 서비스" in body
    assert "특수강봉강" in body
    assert "시장 전망" not in body
    assert "3. 원재료 및 생산설비" not in body


def test_subsidiary_appendix_is_not_authorized_without_the_report_reference():
    rows = _chunks("당사는 사업을 영위하고 있습니다.")

    assert _referenced_subsidiary_product_chunk_ids(rows) == frozenset()
    assert _referenced_holding_product_role_chunk_ids(rows) == frozenset()


@pytest.mark.parametrize("topic", ["주요 사업 내용", "주요 제품 및 서비스"])
def test_verified_subsidiary_profile_excludes_unrelated_appendix_sections(topic):
    row = _chunks(_REFERENCE)[1]
    task = SimpleNamespace(retrieval_query=topic)
    hit = SimpleNamespace(
        section_id=row.section_id,
        path=row.path, text_prompt_safe=row.text_prompt_safe,
        subsidiary_products_reference_verified=True,
    )

    text = _authority_roundtripped_text(
        task, hit, row, row.text_prompt_safe, "business_axis")

    assert text is not None
    body = _strip_prompt_boundaries(text)
    assert body.startswith("(주)예시금속\n\n1. 사업의 개요")
    assert "특수강봉강" in body
    assert "시장 전망" not in body
    assert "원재료 및 생산설비" not in body
