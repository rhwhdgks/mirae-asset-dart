"""코퍼스 경로 해석기.

제공 코퍼스의 `raw/` 하위 한글 법인 폴더명은 유니코드 NFD(자모 분해형)로 저장돼 있고,
`manifest.jsonl`의 `file_path`와 `universe.csv`의 `corp_name`은 NFC(완성형)다.
따라서 문자열로 경로를 재조합하면 4,204건 중 4,054건이 예외 없이 빈 결과로 실패한다.

해결 원칙(주최 측 공지 및 전처리 설계명세 v0.2):

1. 정규화한 문자열로 경로를 만들지 않는다. 실제 파일시스템을 훑어 얻은 Path만 사용한다.
2. 조회 키만 NFC로 정규화한다.
3. 정규화 후 충돌하는 항목은 자동 선택하지 않고 오류로 올린다.

폴더명을 NFC로 일괄 변경하는 방법도 공지에 안내돼 있으나 채택하지 않는다.
배포 환경에 원본 코퍼스를 새로 복사하면 NFD로 돌아와 같은 실패가 재현되고,
그 시점에는 원인 추적이 어렵기 때문이다.
"""

from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

__all__ = [
    "nfc",
    "normalized_path_key",
    "PathBinding",
    "CorpusIndex",
    "PathCollisionError",
    "CorpusPathNotFound",
]


def nfc(text: str) -> str:
    """비교·조회용 정규화. 실제 경로 조립에는 절대 쓰지 않는다."""
    return unicodedata.normalize("NFC", text)


def normalized_path_key(relative_path: str | os.PathLike[str]) -> str:
    """코퍼스 상대경로의 NFC 비교키.

    이 문자열은 조회와 ID 계산에만 쓴다. 반환값으로 파일 경로를 재조립하면 NFD로
    저장된 디렉터리에서 다시 조용히 실패하므로, 실제 접근에는
    :class:`PathBinding` 의 ``path`` 또는 ``actual_relpath`` 를 사용한다.
    """
    return nfc(os.fspath(relative_path)).rstrip("/")


class PathCollisionError(RuntimeError):
    """서로 다른 실제 경로가 같은 NFC 키로 정규화된 경우."""


class CorpusPathNotFound(KeyError):
    """manifest의 경로에 대응하는 실제 경로가 없는 경우."""


@dataclass(frozen=True)
class PathBinding:
    """정규화 조회키와 실제 디스크 경로의 명시적 결합.

    ``normalized_path_key`` 는 NFC 비교·조인용이고, ``actual_relpath`` 는 파일시스템을
    훑어 얻은 표기를 그대로 보존한다. 둘은 한글이 포함된 경로에서 서로 다를 수 있다.
    """

    normalized_path_key: str
    actual_relpath: str
    path: Path


class CorpusIndex:
    """`raw/` 전체를 한 번 훑어 만든 `NFC 상대경로 -> 실제 Path` 맵.

    사용::

        index = CorpusIndex.build("data/corpus")
        folder = index.resolve("raw/periodic/삼성전자/20260310002820_annual_2025_12")
        xmls = index.xml_files(folder_rel)
    """

    def __init__(self, root: Path, entries: dict[str, Path]) -> None:
        self.root = root
        self._entries = entries

    # ------------------------------------------------------------------ build

    @classmethod
    def build(cls, root: str | os.PathLike[str], scan_dir: str = "raw") -> "CorpusIndex":
        root = Path(root)
        target = root / scan_dir
        if not target.is_dir():
            raise FileNotFoundError(f"스캔 대상이 없습니다: {target}")

        entries: dict[str, Path] = {}
        collisions: dict[str, list[Path]] = {}

        for dirpath, dirnames, filenames in os.walk(target):
            here = Path(dirpath)
            for name in list(dirnames) + list(filenames):
                real = here / name
                actual_relpath = real.relative_to(root).as_posix()
                key = normalized_path_key(actual_relpath)
                previous = entries.get(key)
                if previous is not None and previous != real:
                    collisions.setdefault(key, [previous]).append(real)
                    continue
                entries[key] = real

        if collisions:
            detail = "\n".join(
                f"  {key} <- {[str(p) for p in paths]}" for key, paths in collisions.items()
            )
            raise PathCollisionError(
                f"NFC 정규화 후 충돌한 경로 {len(collisions)}건. 자동 선택하지 않습니다.\n{detail}"
            )

        return cls(root, entries)

    # --------------------------------------------------------------- resolving

    def bind(self, relative_path: str | os.PathLike[str]) -> PathBinding:
        """NFC/NFD 어느 표기로 받은 상대경로든 실제 inventory 항목에 결합한다.

        반환된 ``normalized_path_key`` 로 비교하고, 파일을 열 때는 ``path`` 를 쓴다.
        ``actual_relpath`` 는 canonical 산출물에 실제 표기를 보존할 때 사용한다.
        """
        key = normalized_path_key(relative_path)
        try:
            actual = self._entries[key]
        except KeyError:
            raise CorpusPathNotFound(
                f"실제 경로를 찾을 수 없습니다: {os.fspath(relative_path)!r} "
                f"(root={self.root})"
            ) from None
        return PathBinding(
            normalized_path_key=key,
            actual_relpath=actual.relative_to(self.root).as_posix(),
            path=actual,
        )

    def resolve(self, relative_path: str) -> Path:
        """manifest의 `file_path`(NFC)를 실제 Path로 바꾼다."""
        return self.bind(relative_path).path

    def try_resolve(self, relative_path: str) -> Path | None:
        try:
            return self.resolve(relative_path)
        except CorpusPathNotFound:
            return None

    def files(self, relative_path: str) -> list[Path]:
        """문서 폴더 안의 파일을 이름순으로 돌려준다."""
        folder = self.resolve(relative_path)
        if not folder.is_dir():
            raise NotADirectoryError(f"폴더가 아닙니다: {folder}")
        return sorted((p for p in folder.iterdir() if p.is_file()), key=lambda p: p.name)

    def xml_files(self, relative_path: str) -> list[Path]:
        return [p for p in self.files(relative_path) if p.suffix.lower() == ".xml"]

    def main_xml(self, relative_path: str, rcept_no: str) -> Path | None:
        """본문 XML(`{접수번호}.xml`). 첨부(`_00760` 등)와 구분한다."""
        for path in self.xml_files(relative_path):
            if path.stem == rcept_no:
                return path
        return None

    def attachment_xmls(self, relative_path: str, rcept_no: str) -> list[Path]:
        """첨부 XML(`{접수번호}_00760.xml` 등). 실측 415건 전부 감사보고서·연결감사보고서다.

        본문과 같은 DART XML 이라 같은 파서를 쓴다. 정기공시 210개 문서에만 붙어 있고
        문서당 최대 2개다.
        """
        return [p for p in self.xml_files(relative_path) if p.stem != rcept_no]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, relative_path: object) -> bool:
        return (isinstance(relative_path, (str, os.PathLike))
                and normalized_path_key(relative_path) in self._entries)


# ----------------------------------------------------------------- manifest 검증


def iter_manifest(root: str | os.PathLike[str]) -> Iterator[dict]:
    path = Path(root) / "manifest.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def verify(root: str | os.PathLike[str] = "data/corpus") -> dict:
    """manifest 전건에 대해 경로 해석과 파일 수 일치를 검사한다.

    `naive_hits`는 정규화 없이 문자열로 경로를 조립했을 때 우연히 성공하는 건수다.
    이 값이 전체보다 작다는 사실 자체가 해석기가 필요한 이유의 증거다.
    """
    root = Path(root)
    index = CorpusIndex.build(root)

    total = 0
    naive_hits = 0
    unresolved: list[str] = []
    file_count_mismatch: list[tuple[str, int, int]] = []
    main_xml_missing: list[str] = []
    source_files = 0
    source_files_resolved = 0
    source_file_contract_mismatch: list[str] = []

    for record in iter_manifest(root):
        total += 1
        rel = record["file_path"]

        if (root / rel).is_dir():
            naive_hits += 1

        folder = index.try_resolve(rel)
        if folder is None:
            unresolved.append(record["doc_id"])
            continue

        files = index.files(rel)
        actual = len(files)
        expected = int(record["n_files"])
        if actual != expected:
            file_count_mismatch.append((record["doc_id"], expected, actual))

        # 파일 단위 계약도 검증한다. NFC 키로 다시 찾았을 때 inventory에서 얻은 실제
        # Path와 같아야 하며, actual_relpath로 연 파일도 같은 항목이어야 한다.
        for actual_path in files:
            source_files += 1
            actual_relpath = actual_path.relative_to(root).as_posix()
            try:
                binding = index.bind(actual_relpath)
            except CorpusPathNotFound:
                source_file_contract_mismatch.append(actual_relpath)
                continue
            if (binding.path != actual_path
                    or root / binding.actual_relpath != actual_path
                    or binding.normalized_path_key != normalized_path_key(actual_relpath)):
                source_file_contract_mismatch.append(actual_relpath)
                continue
            source_files_resolved += 1

        if record["file_format"] == "xml" and index.main_xml(rel, record["rcept_no"]) is None:
            main_xml_missing.append(record["doc_id"])

    return {
        "indexed_entries": len(index),
        "manifest_records": total,
        "naive_hits": naive_hits,
        "resolved": total - len(unresolved),
        "unresolved": unresolved,
        "file_count_mismatch": file_count_mismatch,
        "main_xml_missing": main_xml_missing,
        "source_files": source_files,
        "source_files_resolved": source_files_resolved,
        "source_file_contract_mismatch": source_file_contract_mismatch,
    }


def _main() -> int:
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/corpus"
    report = verify(root)

    total = report["manifest_records"]
    naive = report["naive_hits"]
    print(f"인덱싱된 항목      : {report['indexed_entries']:,}")
    print(f"manifest 레코드    : {total:,}")
    print(f"정규화 없이 성공   : {naive:,} / {total:,}  ({naive / total:.1%})  <- 조용한 실패 구간")
    print(f"해석기로 성공      : {report['resolved']:,} / {total:,}")
    print(f"미해결             : {len(report['unresolved'])}")
    print(f"n_files 불일치     : {len(report['file_count_mismatch'])}")
    print(f"본문 XML 없음      : {len(report['main_xml_missing'])}")
    print(f"원문 파일 경로 계약 : {report['source_files_resolved']:,} / "
          f"{report['source_files']:,}")

    for doc_id in report["unresolved"][:10]:
        print(f"  미해결: {doc_id}")
    for doc_id, expected, actual in report["file_count_mismatch"][:10]:
        print(f"  파일수 불일치: {doc_id} manifest={expected} 실제={actual}")
    for doc_id in report["main_xml_missing"][:10]:
        print(f"  본문 XML 없음: {doc_id}")
    for relpath in report["source_file_contract_mismatch"][:10]:
        print(f"  원문 경로 계약 불일치: {relpath}")

    ok = not (
        report["unresolved"] or report["file_count_mismatch"] or report["main_xml_missing"]
        or report["source_file_contract_mismatch"]
    )
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_main())
