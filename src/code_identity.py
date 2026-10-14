"""Best-effort, content-free Git identity; no environment/credential reads."""
from hashlib import sha256
from pathlib import Path
import subprocess


def code_identity(root: Path) -> dict:
    def git_bytes(*args: str) -> bytes:
        return subprocess.check_output(
            ["git", *args], cwd=root, stderr=subprocess.DEVNULL,
            timeout=5)
    def git(*args: str) -> str:
        return git_bytes(*args).decode("utf-8").strip()
    try:
        diff = git_bytes("diff", "HEAD", "--binary")
        return {"git_head": git("rev-parse", "HEAD"),
                "branch": git("branch", "--show-current"),
                "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                "tracked_diff_sha256": sha256(diff).hexdigest()}
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return {"git_head": None, "availability": "unavailable"}
