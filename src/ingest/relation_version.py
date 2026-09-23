"""정정 관계 판별기 버전.

상류 relation JSONL의 도장과 canonical ``Relation.resolver_version``이 같은
판별 규칙을 가리키게 한 곳에서 관리한다. 날짜 표기·PDF 최초제출일 판별처럼
endpoint를 바꾸는 수정은 이 값을 반드시 올린다.
"""

RELATION_RESOLVER_VERSION = "relation/1.2+source-date"

__all__ = ["RELATION_RESOLVER_VERSION"]
