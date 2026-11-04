"""Recorded tool outcomes, separate from model-generated segments."""
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    execution_id: str
    status: str
    stdout: str = ""
    stderr: str = ""
    error: str | None = None
    elapsed_seconds: float = 0.0
    searches: list[dict] = field(default_factory=list)
    execution_mode: str = "trusted_subprocess"

    @property
    def observation(self) -> str:
        if self.status == "success":
            return self.stdout
        return f"{self.stdout}\nERROR [{self.status}]: {self.error or self.stderr}".lstrip("\n")
