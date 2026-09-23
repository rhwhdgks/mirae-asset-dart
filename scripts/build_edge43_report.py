#!/usr/bin/env python3
"""Build the human-review report for EDGE-002..EDGE-044.

The report deliberately separates the answer returned by the running system
from the semantic verdict assigned by the team.  It never treats HTTP 200 as
answer correctness.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


PASS_BASELINE = {2, 3, 4, 5, 8, 10, 20, 22, 24, 27, 30, 40}
PASS_FINAL = {21, 31, 36, 39, 42, 44}
PARTIAL_FINAL = {7, 11, 13, 14, 15, 16, 26, 29, 38, 43}
FAIL_FINAL = {6, 9, 12, 17, 18, 19, 23, 25, 28, 32, 33, 34, 35, 37, 41}
TARGETED_OVERRIDE = {13, 14, 16, 43}
CORRECTION_OVERRIDE = {32, 39}

BASELINE_SUMMARIES = {
    2: "두 회사 매출액, 대소 관계, 차이와 공시 근거를 모두 반환했다.",
    3: "두 회사 매출액 비교와 차이 계산을 맞게 반환했다.",
    4: "2025년 별도 매출액과 공시 근거를 반환했다.",
    5: "두 회사 연결 매출액 비교와 차이를 맞게 반환했다.",
    8: "두 로봇 회사의 매출액 비교와 차이를 맞게 반환했다.",
    10: "2025년 연결 매출액과 공시 근거를 반환했다.",
    20: "2025년 별도 매출액과 공시 근거를 반환했다.",
    22: "전년 대비 증감액과 증감률을 맞게 반환했다.",
    24: "연결·별도 매출액과 차이를 맞게 반환했다.",
    27: "서로 다른 원천 단위를 정규화한 뒤 비교와 차이를 맞게 반환했다.",
    30: "2025년 연결 매출액과 공시 근거를 반환했다.",
    40: "전년 대비 증감액과 증감률을 맞게 반환했다.",
}

VERDICT_NOTES = {
    6: "Stage1 의미 정규화가 실패해 실행 계획을 만들지 못했다.",
    7: "회사×기간의 cited 원문은 확보했지만 연도 간 변화 비교를 합성하지 못했다.",
    9: "지원 가능한 CAPEX 절댓값 비교를 지원 불가로 거절했다.",
    11: "회사×기간 원문과 근거는 확보했지만 비교 결론 대신 원문을 나열했고 구조 예산 limitation이 남았다.",
    12: "Stage1 compiler binding 실패로 실행 계획을 만들지 못했다.",
    13: "두 회사 근거는 확보했지만 수익구조 일부가 not_found이고 비교 합성이 부족했다.",
    14: "두 금융지주 근거는 확보했지만 핵심 자회사 coverage와 구조 예산 limitation이 남았다.",
    15: "양사 cited 원문은 확보했지만 주요 사업 차이를 직접 합성하지 못했다.",
    16: "근거는 확보했지만 수익원 일부가 not_found이고 증권사·금융지주 차이 합성이 부족했다.",
    17: "이번 실행에서는 지원 가능한 재무 비교를 거절해 요구를 충족하지 못했다.",
    18: "최신 정기보고서 주요 사업 정리를 거절해 요구를 충족하지 못했다.",
    19: "회사와 별칭이 이미 명시됐는데도 불필요한 역질문을 반환했다.",
    21: "두 게임사의 연결 매출액, 대소 관계, 차이와 근거를 반환했다.",
    23: "실행 응답은 있었지만 요청한 설비 투자계획 전체를 근거와 함께 충족하지 못했다.",
    25: "Stage1이 질문을 실행 가능한 사업·제품 비교 계획으로 해석하지 못했다.",
    26: "양사 cited 원문은 확보했지만 사업모델 차이를 비교 문장으로 합성하지 못했다.",
    28: "event 조회가 45초 timeout으로 끝나 답변을 만들지 못했다.",
    29: "양사 cited 원문은 확보했지만 원전 관련 공통점과 차이를 합성하지 못했다.",
    31: "투자 대상·목적·금액·기간을 표로 완전하게 반환하고 접수번호를 제시했다.",
    32: "Composer 수정 후 두 차례 재호출이 모두 45초 실행 제한에 걸려 최신 실측에서는 답변을 만들지 못했다.",
    33: "특정 계약의 상대방과 계약명 조회를 거절해 요구를 충족하지 못했다.",
    34: "사명 변경 전후 동일 법인 조회를 해결하지 못해 요구한 매출액을 반환하지 못했다.",
    35: "Stage1이 전년 대비 증감액·증감률 실행 계획을 만들지 못했다.",
    36: "3개 회사의 모든 원값, 순위, 접수번호를 반환했다.",
    37: "Stage1이 전년 대비 증감률 실행 계획을 만들지 못했다.",
    38: "양사 cited 원문은 확보했지만 주요 사업 차이를 비교 문장으로 합성하지 못했다.",
    39: "3단 정정과 6→8→10→16개월 변경, 접수번호와 provisional limitation을 모두 반환했다.",
    41: "국문·영문 회사 표기가 함께 있는 증감 질문을 Stage1이 실행 계획으로 만들지 못했다.",
    42: "두 회사 원값, 큰 기업, 절댓값 차이와 접수번호를 맞게 반환했다.",
    43: "3개 회사의 cited 원문은 확보했지만 신재생·연료전지 사업의 공통점과 차이를 합성하지 못했다.",
    44: "4개 회사의 모든 원값, 순위, 접수번호를 반환했고 괄호 복합 회사명도 안전하게 병합했다.",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def edge_number(value: str) -> int:
    return int(value.rsplit("-", 1)[-1])


def verdict(number: int) -> str:
    if number in PASS_BASELINE or number in PASS_FINAL:
        return "통과"
    if number in PARTIAL_FINAL:
        return "부분 통과"
    if number in FAIL_FINAL:
        return "실패"
    raise KeyError(number)


def expected_answer(question: str) -> str:
    return (
        "질문에 명시된 회사·기간·재무제표/공시 범위를 지키고, 요청한 값·비교·변화·목록을 "
        "빠짐없이 제시하며 각 핵심 주장에 접수번호 또는 동등한 공시 근거를 연결해야 한다. "
        "확정할 수 없는 부분은 추정하지 않고 확인된 내용과 typed limitation을 분리해야 한다."
    )


def stage_summary(row: dict) -> str:
    parts: list[str] = []
    for key, label in (
        ("status", "request"),
        ("handoff_status", "handoff"),
        ("final_status", "final"),
    ):
        if row.get(key) is not None:
            parts.append(f"{label}={row[key]}")
    if row.get("http_status") is not None:
        parts.append(f"HTTP={row['http_status']}")
    if row.get("elapsed") is not None:
        parts.append(f"{float(row['elapsed']):.3f}s")
    if row.get("error"):
        parts.append(f"error={row['error']}")
    return ", ".join(parts) or "상세 실행 메타데이터 없음"


def build(repo: Path) -> str:
    requests = repo / "out" / "requests"
    baseline_rows = read_jsonl(requests / "answers_20260825.jsonl")
    final_rows = read_jsonl(requests / "edge31_live_final_20260825.jsonl")
    targeted_rows = read_jsonl(requests / "edge31_targeted_followup_20260825.jsonl")
    correction_rows = read_jsonl(requests / "correction_limitation_live_20260825.jsonl")
    final_server_rows = read_jsonl(requests / "edge31_final_server" / "answers_20260825.jsonl")
    targeted_server_rows = read_jsonl(requests / "edge31_targeted_followup_server" / "answers_20260825.jsonl")

    baseline_by_number: dict[int, dict] = {}
    for row in baseline_rows:
        qid = row.get("question_id", "")
        if qid == "EDGE-024-RERUN":
            baseline_by_number[24] = row
        elif qid.startswith("EDGE-") and qid[5:].isdigit():
            number = edge_number(qid)
            if number != 24:
                baseline_by_number.setdefault(number, row)

    final_by_number = {edge_number(row["source_question_id"]): row for row in final_rows}
    targeted_by_number = {edge_number(row["source_question_id"]): row for row in targeted_rows}
    correction_by_number = {edge_number(row["source_question_id"]): row for row in correction_rows}
    final_server = {row["question_id"]: row for row in final_server_rows}
    targeted_server = {row["question_id"]: row for row in targeted_server_rows}

    records: list[dict] = []
    for number in range(2, 45):
        if number in CORRECTION_OVERRIDE:
            live = correction_by_number[number]
            records.append(
                {
                    "number": number,
                    "question": live["question"],
                    "verdict": verdict(number),
                    "source": "정정 limitation 수정 후 live 재호출",
                    "source_file": "out/requests/correction_limitation_live_20260825.jsonl",
                    "run": {**live, **live.get("server_meta", {})},
                    "answer": live.get("response", {}).get("answer", ""),
                    "answer_kind": "전문",
                }
            )
        elif number in TARGETED_OVERRIDE:
            live = targeted_by_number[number]
            server = targeted_server.get(live["question_id"], {})
            records.append(
                {
                    "number": number,
                    "question": live["question"],
                    "verdict": verdict(number),
                    "source": "표적 재호출",
                    "source_file": "out/requests/edge31_targeted_followup_20260825.jsonl",
                    "run": {**live, **server},
                    "answer": live.get("response", {}).get("answer", ""),
                    "answer_kind": "전문",
                }
            )
        elif number in PASS_BASELINE:
            row = baseline_by_number[number]
            records.append(
                {
                    "number": number,
                    "question": row["question"],
                    "verdict": verdict(number),
                    "source": "기존 통과 기준선",
                    "source_file": "out/requests/answers_20260825.jsonl",
                    "run": row,
                    "answer": BASELINE_SUMMARIES[number],
                    "answer_kind": "요약",
                }
            )
        else:
            live = final_by_number[number]
            server = final_server.get(live["question_id"], {})
            records.append(
                {
                    "number": number,
                    "question": live["question"],
                    "verdict": verdict(number),
                    "source": "31문항 최종 재호출",
                    "source_file": "out/requests/edge31_live_final_20260825.jsonl",
                    "run": {**live, **server},
                    "answer": live.get("response", {}).get("answer", ""),
                    "answer_kind": "전문",
                }
            )

    if [r["number"] for r in records] != list(range(2, 45)):
        raise RuntimeError("EDGE-002..044 배정이 완전하지 않습니다")
    counts = {name: sum(r["verdict"] == name for r in records) for name in ("통과", "부분 통과", "실패")}
    if counts != {"통과": 18, "부분 통과": 10, "실패": 15}:
        raise RuntimeError(f"판정 합계가 합의값과 다릅니다: {counts}")

    out: list[str] = [
        "# E2E 43문항 질문·응답·판정 기록",
        "",
        "## 문서 범위와 읽는 법",
        "",
        "이 문서는 `EDGE-001`을 제외하고 `EDGE-002`~`EDGE-044` 43문항의 질문, 실제 시스템 응답, 의미 판정과 실패 사유를 한곳에 모은 인계 자료다. HTTP 200 여부와 정답 여부를 분리하며, 숫자 정답이 별도 검증되지 않은 문항에는 임의의 값을 만들지 않는다.",
        "",
        "결과 선택 우선순위는 다음과 같다.",
        "",
        "1. 13·14·16·43: 표적 재호출 결과",
        "2. 32·39: correction limitation 수정 후 live 재호출 결과",
        "3. 그 외 최신 수정 대상: 31문항 최종 재호출 결과",
        "4. 재호출 대상이 아니었던 기존 통과 12문항: 당시 실행 요약",
        "",
        f"- 최종 의미 판정: 통과 {counts['통과']} / 부분 통과 {counts['부분 통과']} / 실패 {counts['실패']} / 합계 43",
        "- 기존 통과 12문항은 당시 raw 응답 전문을 저장하지 않았으므로 실행 로그의 응답 요약을 싣는다.",
        "- 나머지 31문항은 저장된 live 응답 전문을 그대로 싣는다.",
        "- 32·39는 correction limitation Composer 수정 후 다시 호출한 최신 실측을 반영한다.",
        "",
        "## 전체 판정표",
        "",
        "| 번호 | 의미 판정 | 결과 출처 | 핵심 판정 사유 |",
        "|---:|---|---|---|",
    ]
    for record in records:
        number = record["number"]
        note = BASELINE_SUMMARIES[number] if number in BASELINE_SUMMARIES else VERDICT_NOTES[number]
        out.append(f"| {record['number']} | {record['verdict']} | {record['source']} | {note.replace('|', '&#124;')} |")

    out += ["", "## 문항별 질문과 실제 응답", ""]
    for record in records:
        number = record["number"]
        note = BASELINE_SUMMARIES[number] if number in BASELINE_SUMMARIES else VERDICT_NOTES[number]
        escaped_answer = html.escape(record["answer"] or "응답 본문 없음")
        out += [
            f"### EDGE-{number:03d}",
            "",
            f"- 질문: {record['question']}",
            f"- 기대 답변/정답 기준: {expected_answer(record['question'])}",
            f"- 의미 판정: **{record['verdict']}**",
            f"- 판정 사유: {note}",
            f"- 실행 상태: `{stage_summary(record['run'])}`",
            f"- 권위 결과: `{record['source_file']}` ({record['source']})",
            "",
        ]
        if record["answer_kind"] == "요약":
            out += [
                "실제 응답 기록: 응답 전문은 보존되지 않았고, 당시 검수자가 남긴 요약은 다음과 같다.",
                "",
                f"> {record['answer']}",
                "",
            ]
        else:
            out += [
                "<details>",
                "<summary>실제 응답 전문 펼치기</summary>",
                "",
                f"<pre>{escaped_answer}</pre>",
                "",
                "</details>",
                "",
            ]

    out += [
        "## 수정 후 판정 영향 메모",
        "",
        "- `EDGE-039`: 수정 후 live 응답에서 3단 변경 이력·접수번호와 `correction_identity_provisional` 안전 문구가 함께 확인되어 통과로 승격했다.",
        "- `EDGE-032`: 수정 후 재호출 두 번 모두 45초 timeout이 재현되어 최신 실측은 실패다. limitation 렌더링보다 앞선 collection/실행시간 문제를 별도로 해결해야 한다.",
        "- `EDGE-028`: 실패 원인이 timeout과 event 검색 경로이므로 이번 Composer limitation 수정의 해결 대상이 아니다.",
        "- narrative 문항의 공통 잔여 문제는 Stage1 matrix 생성이 아니라 Stage4 비교 합성이다. 셀별 회사·기간·주제·citation을 보존한 채 차이를 합성하고, 빠진 셀은 limitation으로 남겨야 한다.",
        "",
    ]
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/stage1/43문항_질문_응답_판정.md"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output if args.output.is_absolute() else repo / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build(repo), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
