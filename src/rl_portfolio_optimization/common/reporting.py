from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd


@dataclass
class ReportArtifact:
    title: str
    filename: str
    caption: str | None = None


@dataclass
class RunReport:
    workflow_name: str
    root_dir: Path
    sections: list[tuple[str, str]] = field(default_factory=list)
    artifacts: list[ReportArtifact] = field(default_factory=list)

    @classmethod
    def create(cls, reports_root: Path, workflow_name: str) -> "RunReport":
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root_dir = reports_root / workflow_name / timestamp
        root_dir.mkdir(parents=True, exist_ok=True)
        return cls(workflow_name=workflow_name, root_dir=root_dir)

    def add_text(self, heading: str, body: str) -> None:
        self.sections.append((heading, body))

    def save_dataframe(self, frame: pd.DataFrame, filename: str) -> Path:
        path = self.root_dir / filename
        frame.to_csv(path)
        return path

    def add_artifact(self, title: str, filename: str, caption: str | None = None) -> None:
        self.artifacts.append(ReportArtifact(title=title, filename=filename, caption=caption))

    def write_markdown(self, extra_lines: list[str] | None = None) -> Path:
        lines = [f"# {self.workflow_name} Run Report", ""]
        lines.append(f"Generated at: `{datetime.now().isoformat(timespec='seconds')}`")
        lines.append("")

        for heading, body in self.sections:
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(body)
            lines.append("")

        if self.artifacts:
            lines.append("## Figures")
            lines.append("")
            for artifact in self.artifacts:
                lines.append(f"### {artifact.title}")
                lines.append("")
                lines.append(f"![{artifact.title}]({artifact.filename})")
                lines.append("")
                if artifact.caption:
                    lines.append(artifact.caption)
                    lines.append("")

        if extra_lines:
            lines.extend(extra_lines)

        path = self.root_dir / "report.md"
        path.write_text("\n".join(lines))
        return path
