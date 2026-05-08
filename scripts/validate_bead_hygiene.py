#!/usr/bin/env python3
"""Validate Beads tracker hygiene for high-churn parity work.

This verifier reads `.beads/issues.jsonl` directly and emits a deterministic
report suitable for `br`/`bv` triage. It does not update tracker rows. Instead,
it proposes exact append-only parent progress notes that a human or agent can
review before applying with `br update`.

Usage:
    python3 scripts/validate_bead_hygiene.py
    python3 scripts/validate_bead_hygiene.py --json-out artifacts/bead-hygiene-report.json
    python3 scripts/validate_bead_hygiene.py --markdown-out artifacts/bead-hygiene-report.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ISSUES = REPO_ROOT / ".beads" / "issues.jsonl"

CURRENT_RE = re.compile(
    r"Current:\s*(?P<pct>\d+)%\s*\((?P<implemented>\d+)/(?P<total>\d+)\s+pandas methods\)",
    re.IGNORECASE,
)
GAP_RE = re.compile(r"Gap:\s*(?P<missing>\d+)\s+missing methods", re.IGNORECASE)
CROSS_LIST_RE = re.compile(
    r"Cross-listed with\s+[`'\"]?(?P<id>[A-Za-z0-9_.-]+)[`'\"]?",
    re.IGNORECASE,
)
BACKTICK_RE = re.compile(r"`([^`]+)`")
METHOD_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
COMMON_FALSE_POSITIVE_METHODS = {"a", "an", "as", "at", "by", "if", "in", "is", "no", "of", "on", "or", "to"}


def parse_timestamp(value: str | None) -> datetime | None:
    """Parse Beads timestamps, including nanosecond fractions, as UTC datetimes."""
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        prefix, suffix = text.split(".", 1)
        match = re.match(r"(?P<fraction>\d+)(?P<tail>.*)", suffix)
        if match:
            fraction = match.group("fraction")[:6].ljust(6, "0")
            text = f"{prefix}.{fraction}{match.group('tail')}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_issues(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    issues: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                issue = decoder.decode(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            issue["_line_no"] = line_no
            issue_id = str(issue.get("id") or "")
            if not issue_id:
                raise ValueError(f"{path}:{line_no}: issue missing id")
            if issue_id in issues:
                raise ValueError(f"{path}:{line_no}: duplicate issue id {issue_id}")
            issues[issue_id] = issue
            ordered.append(issue)
    return issues, ordered


def issue_timestamp(issue: dict[str, Any], key: str) -> datetime | None:
    return parse_timestamp(issue.get(key))


def max_issue_timestamp(issues: Iterable[dict[str, Any]]) -> datetime:
    candidates: list[datetime] = []
    for issue in issues:
        for key in ("updated_at", "closed_at", "created_at"):
            parsed = issue_timestamp(issue, key)
            if parsed is not None:
                candidates.append(parsed)
    if not candidates:
        return datetime.now(timezone.utc).replace(microsecond=0)
    return max(candidates)


def compact_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": issue.get("id"),
        "title": issue.get("title"),
        "status": issue.get("status"),
        "priority": issue.get("priority"),
        "updated_at": issue.get("updated_at"),
        "closed_at": issue.get("closed_at"),
    }


def dependency_edges(ordered: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for issue in ordered:
        for dep in issue.get("dependencies") or []:
            edge = dict(dep)
            edge["_container_issue_id"] = issue["id"]
            edges.append(edge)
    return sorted(
        edges,
        key=lambda dep: (
            str(dep.get("issue_id") or ""),
            str(dep.get("depends_on_id") or ""),
            str(dep.get("type") or ""),
        ),
    )


def parent_child_index(edges: list[dict[str, Any]]) -> dict[str, list[str]]:
    children_by_parent: dict[str, list[str]] = defaultdict(list)
    for dep in edges:
        if dep.get("type") != "parent-child":
            continue
        parent_id = dep.get("depends_on_id")
        child_id = dep.get("issue_id")
        if parent_id and child_id:
            children_by_parent[str(parent_id)].append(str(child_id))
    return {parent: sorted(set(children)) for parent, children in sorted(children_by_parent.items())}


def detect_stale_in_progress(
    ordered: list[dict[str, Any]], as_of: datetime, stale_hours: float
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for issue in sorted(ordered, key=lambda item: item["id"]):
        if issue.get("status") != "in_progress":
            continue
        updated = issue_timestamp(issue, "updated_at") or issue_timestamp(issue, "created_at")
        if updated is None:
            continue
        age_hours = (as_of - updated).total_seconds() / 3600.0
        if age_hours >= stale_hours:
            findings.append(
                {
                    **compact_issue(issue),
                    "age_hours": round(age_hours, 2),
                    "threshold_hours": stale_hours,
                }
            )
    return findings


def detect_cross_listed_duplicates(
    issues: dict[str, dict[str, Any]], ordered: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    pairs: dict[tuple[str, str], dict[str, Any]] = {}
    for issue in ordered:
        description = issue.get("description") or ""
        for match in CROSS_LIST_RE.finditer(description):
            other_id = match.group("id").rstrip(".,;")
            if other_id not in issues:
                continue
            pair_key = tuple(sorted((issue["id"], other_id)))
            first, second = pair_key
            pairs[pair_key] = {
                "ids": [first, second],
                "statuses": {
                    first: issues[first].get("status"),
                    second: issues[second].get("status"),
                },
                "titles": {
                    first: issues[first].get("title"),
                    second: issues[second].get("title"),
                },
                "source": issue["id"],
                "kind": "explicit_cross_list",
            }
    return [pairs[key] for key in sorted(pairs)]


def extract_missing_method_tokens(description: str) -> list[str]:
    tokens: set[str] = set()
    for span in BACKTICK_RE.findall(description or ""):
        for raw in re.split(r"[,\s/]+", span):
            token = raw.strip()
            if len(token) < 2:
                continue
            if token in COMMON_FALSE_POSITIVE_METHODS:
                continue
            if METHOD_TOKEN_RE.match(token):
                tokens.add(token)
    return sorted(tokens)


def method_is_mentioned(text: str, method: str) -> bool:
    lowered = text.lower()
    escaped = re.escape(method.lower())
    patterns = [
        rf"::{escaped}(?![A-Za-z0-9_])",
        rf"(?<![A-Za-z0-9_]){escaped}\s*\(",
        rf"(?<![A-Za-z0-9_-]){escaped}(?![A-Za-z0-9_-])",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def text_for_child_scan(issue: dict[str, Any]) -> str:
    # Child titles are the tracker convention for narrow method-slice claims
    # ("SeriesGroupBy quantile sem skew parity", "MultiIndex rename parity").
    # Descriptions and close reasons contain prose like "empty target" or
    # "all-missing groups" that can look like API methods but are only test
    # scenarios, so keep this signal intentionally narrow.
    return issue.get("title") or ""


def parse_progress_counts(description: str) -> dict[str, int | None]:
    current = CURRENT_RE.search(description or "")
    gap = GAP_RE.search(description or "")
    return {
        "reported_percent": int(current.group("pct")) if current else None,
        "reported_implemented": int(current.group("implemented")) if current else None,
        "reported_total": int(current.group("total")) if current else None,
        "reported_missing": int(gap.group("missing")) if gap else None,
    }


def detect_umbrella_progress_contradictions(
    issues: dict[str, dict[str, Any]],
    children_by_parent: dict[str, list[str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for parent_id in sorted(children_by_parent):
        parent = issues.get(parent_id)
        if not parent or parent.get("status") == "closed":
            continue
        description = parent.get("description") or ""
        counts = parse_progress_counts(description)
        missing_methods = extract_missing_method_tokens(description)
        if not missing_methods:
            continue
        method_children: dict[str, list[str]] = defaultdict(list)
        closed_children = [
            issues[child_id]
            for child_id in children_by_parent[parent_id]
            if child_id in issues and issues[child_id].get("status") == "closed"
        ]
        for child in closed_children:
            text = text_for_child_scan(child)
            for method in missing_methods:
                if method_is_mentioned(text, method):
                    method_children[method].append(child["id"])
        implemented_methods = [
            {"method": method, "closed_children": sorted(set(children))}
            for method, children in sorted(method_children.items())
        ]
        if not implemented_methods:
            continue
        implemented_count = len(implemented_methods)
        reported_missing = counts["reported_missing"]
        adjusted_missing_floor = (
            max(0, reported_missing - implemented_count)
            if reported_missing is not None
            else None
        )
        findings.append(
            {
                "parent": compact_issue(parent),
                "progress_counts": counts,
                "closed_child_count": len(closed_children),
                "listed_missing_methods_with_closed_children": implemented_methods,
                "reported_missing_minus_closed_child_methods": adjusted_missing_floor,
            }
        )
    return findings


def detect_missing_parent_progress_notes(
    issues: dict[str, dict[str, Any]],
    children_by_parent: dict[str, list[str]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for parent_id in sorted(children_by_parent):
        parent = issues.get(parent_id)
        if not parent or parent.get("status") == "closed":
            continue
        parent_updated = issue_timestamp(parent, "updated_at") or issue_timestamp(parent, "created_at")
        if parent_updated is None:
            continue
        stale_children: list[dict[str, Any]] = []
        for child_id in children_by_parent[parent_id]:
            child = issues.get(child_id)
            if not child or child.get("status") != "closed":
                continue
            closed_at = issue_timestamp(child, "closed_at") or issue_timestamp(child, "updated_at")
            if closed_at is None or closed_at <= parent_updated:
                continue
            parent_text = f"{parent.get('description') or ''}\n{parent.get('close_reason') or ''}"
            child_mentioned = child_id in parent_text
            stale_children.append(
                {
                    **compact_issue(child),
                    "child_id_mentioned_in_parent_text": child_mentioned,
                }
            )
        if stale_children:
            latest = max(
                (issue_timestamp(child, "closed_at") or issue_timestamp(child, "updated_at"))
                for child in stale_children
            )
            findings.append(
                {
                    "parent": compact_issue(parent),
                    "parent_updated_at": parent.get("updated_at"),
                    "closed_children_after_parent_update": sorted(
                        stale_children, key=lambda child: child["id"]
                    ),
                    "latest_child_closed_at": format_timestamp(latest),
                }
            )
    return findings


def detect_parent_child_integrity(
    issues: dict[str, dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: Counter[tuple[str, str, str]] = Counter()
    for dep in edges:
        child_id = str(dep.get("issue_id") or "")
        parent_id = str(dep.get("depends_on_id") or "")
        dep_type = str(dep.get("type") or "")
        seen[(child_id, parent_id, dep_type)] += 1
        problems: list[str] = []
        if not child_id:
            problems.append("missing issue_id")
        if not parent_id:
            problems.append("missing depends_on_id")
        if child_id and child_id not in issues:
            problems.append("issue_id does not exist")
        if parent_id and parent_id not in issues:
            problems.append("depends_on_id does not exist")
        if child_id and dep.get("_container_issue_id") != child_id:
            problems.append("dependency stored under a different issue row")
        if dep_type == "parent-child" and child_id and parent_id and child_id == parent_id:
            problems.append("parent-child self dependency")
        if dep_type == "parent-child" and child_id in issues and parent_id in issues:
            child_status = issues[child_id].get("status")
            parent_status = issues[parent_id].get("status")
            if child_status != "closed" and parent_status == "closed":
                problems.append("open child depends on closed parent")
        if problems:
            findings.append(
                {
                    "dependency": {
                        "issue_id": child_id or None,
                        "depends_on_id": parent_id or None,
                        "type": dep_type or None,
                        "container_issue_id": dep.get("_container_issue_id"),
                    },
                    "problems": sorted(problems),
                }
            )
    for (child_id, parent_id, dep_type), count in sorted(seen.items()):
        if count > 1:
            findings.append(
                {
                    "dependency": {
                        "issue_id": child_id,
                        "depends_on_id": parent_id,
                        "type": dep_type,
                    },
                    "problems": [f"duplicate dependency edge appears {count} times"],
                }
            )
    return findings


def detect_dependency_cycles(issues: dict[str, dict[str, Any]], edges: list[dict[str, Any]]) -> list[list[str]]:
    graph: dict[str, list[str]] = {issue_id: [] for issue_id in issues}
    for dep in edges:
        child_id = dep.get("issue_id")
        depends_on_id = dep.get("depends_on_id")
        if child_id in issues and depends_on_id in issues:
            graph[str(child_id)].append(str(depends_on_id))
    for deps in graph.values():
        deps.sort()

    cycles: set[tuple[str, ...]] = set()
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
        if not cycle:
            return tuple()
        rotations = [tuple(cycle[i:] + cycle[:i]) for i in range(len(cycle))]
        return min(rotations)

    def dfs(node: str) -> None:
        visiting.add(node)
        stack.append(node)
        for next_node in graph[node]:
            if next_node in visiting:
                start = stack.index(next_node)
                cycles.add(canonical_cycle(stack[start:] + [next_node]))
            elif next_node not in visited:
                dfs(next_node)
        stack.pop()
        visiting.remove(node)
        visited.add(node)

    for issue_id in sorted(graph):
        if issue_id not in visited:
            dfs(issue_id)
    return [list(cycle) for cycle in sorted(cycles)]


def make_progress_note(
    parent: dict[str, Any],
    contradiction: dict[str, Any] | None,
    stale_note: dict[str, Any] | None,
    as_of: datetime,
) -> str:
    method_lines: list[str] = []
    if contradiction:
        for item in contradiction["listed_missing_methods_with_closed_children"]:
            children = ", ".join(item["closed_children"])
            method_lines.append(f"- `{item['method']}`: closed child bead(s) {children}")
    child_lines: list[str] = []
    if stale_note:
        for child in stale_note["closed_children_after_parent_update"]:
            child_lines.append(f"- `{child['id']}`: {child['title']}")

    lines = [
        "",
        "",
        "### Tracker hygiene progress note",
        "",
        f"Generated by `scripts/validate_bead_hygiene.py` at {format_timestamp(as_of)}.",
    ]
    if method_lines:
        lines.extend(
            [
                "",
                "Closed child beads now cover methods still listed as missing:",
                *method_lines,
            ]
        )
    if child_lines:
        lines.extend(
            [
                "",
                "Closed child beads landed after this umbrella was last updated:",
                *child_lines,
            ]
        )
    if contradiction:
        counts = contradiction["progress_counts"]
        reported_missing = counts.get("reported_missing")
        adjusted = contradiction.get("reported_missing_minus_closed_child_methods")
        if reported_missing is not None and adjusted is not None:
            lines.extend(
                [
                    "",
                    (
                        f"Recount prompt: review the reported gap of {reported_missing} missing "
                        f"methods; the closed-child floor is now at most {adjusted} before any "
                        "new coverage discovered outside child close reasons."
                    ),
                ]
            )
    lines.extend(
        [
            "",
            "Apply this only as an append-only progress note or use it to make a reviewed parent",
            "description update; do not overwrite unrelated tracker edits.",
        ]
    )
    return "\n".join(lines)


def build_proposals(
    issues: dict[str, dict[str, Any]],
    contradictions: list[dict[str, Any]],
    stale_notes: list[dict[str, Any]],
    as_of: datetime,
) -> list[dict[str, Any]]:
    by_parent_contradiction = {
        entry["parent"]["id"]: entry for entry in contradictions
    }
    by_parent_stale = {entry["parent"]["id"]: entry for entry in stale_notes}
    proposals: list[dict[str, Any]] = []
    for parent_id in sorted(set(by_parent_contradiction) | set(by_parent_stale)):
        parent = issues[parent_id]
        proposals.append(
            {
                "parent_id": parent_id,
                "parent_title": parent.get("title"),
                "workflow": "append_only_progress_note",
                "exact_append_text": make_progress_note(
                    parent,
                    by_parent_contradiction.get(parent_id),
                    by_parent_stale.get(parent_id),
                    as_of,
                ),
            }
        )
    return proposals


def build_report(path: Path, as_of: datetime, stale_hours: float) -> dict[str, Any]:
    issues, ordered = load_issues(path)
    edges = dependency_edges(ordered)
    children_by_parent = parent_child_index(edges)

    status_counts = Counter(str(issue.get("status")) for issue in ordered)
    type_counts = Counter(str(issue.get("issue_type")) for issue in ordered)

    stale_in_progress = detect_stale_in_progress(ordered, as_of, stale_hours)
    duplicates = detect_cross_listed_duplicates(issues, ordered)
    contradictions = detect_umbrella_progress_contradictions(issues, children_by_parent)
    missing_parent_notes = detect_missing_parent_progress_notes(issues, children_by_parent)
    integrity = detect_parent_child_integrity(issues, edges)
    cycles = detect_dependency_cycles(issues, edges)
    proposals = build_proposals(issues, contradictions, missing_parent_notes, as_of)

    return {
        "generated_by": "scripts/validate_bead_hygiene.py",
        "issues_path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
        "as_of": format_timestamp(as_of),
        "stale_in_progress_threshold_hours": stale_hours,
        "summary": {
            "issue_count": len(ordered),
            "dependency_edge_count": len(edges),
            "parent_child_parent_count": len(children_by_parent),
            "status_counts": dict(sorted(status_counts.items())),
            "type_counts": dict(sorted(type_counts.items())),
            "finding_counts": {
                "stale_in_progress": len(stale_in_progress),
                "duplicate_cross_listed_epics": len(duplicates),
                "umbrella_progress_contradictions": len(contradictions),
                "missing_parent_progress_notes": len(missing_parent_notes),
                "dependency_cycles": len(cycles),
                "parent_child_integrity": len(integrity),
                "proposed_updates": len(proposals),
            },
        },
        "findings": {
            "stale_in_progress": stale_in_progress,
            "duplicate_cross_listed_epics": duplicates,
            "umbrella_progress_contradictions": contradictions,
            "missing_parent_progress_notes": missing_parent_notes,
            "dependency_cycles": cycles,
            "parent_child_integrity": integrity,
        },
        "proposed_updates": proposals,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    finding_counts = summary["finding_counts"]
    lines = [
        "# Bead Hygiene Report",
        "",
        f"Generated by `scripts/validate_bead_hygiene.py` from `{report['issues_path']}`.",
        "",
        f"- As of: `{report['as_of']}`",
        f"- Issues: `{summary['issue_count']}`",
        f"- Dependency edges: `{summary['dependency_edge_count']}`",
        f"- Stale in-progress threshold: `{report['stale_in_progress_threshold_hours']}` hours",
        "",
        "## Finding Counts",
        "",
        "| Finding | Count |",
        "|---|---:|",
    ]
    for name, count in sorted(finding_counts.items()):
        lines.append(f"| `{name}` | {count} |")

    findings = report["findings"]

    lines.extend(["", "## Structural Findings", ""])
    cycles = findings["dependency_cycles"]
    integrity = findings["parent_child_integrity"]
    if not cycles and not integrity:
        lines.append("- No dependency cycles or parent-child integrity problems detected.")
    else:
        for cycle in cycles:
            lines.append(f"- Cycle: {' -> '.join(cycle)}")
        for item in integrity:
            problems = "; ".join(item["problems"])
            dep = item["dependency"]
            lines.append(
                f"- Parent-child integrity: `{dep.get('issue_id')}` -> "
                f"`{dep.get('depends_on_id')}` ({problems})"
            )

    lines.extend(["", "## Tracker Hygiene Findings", ""])
    stale = findings["stale_in_progress"]
    if stale:
        lines.append("### Stale in-progress beads")
        lines.append("")
        for item in stale:
            lines.append(
                f"- `{item['id']}` age={item['age_hours']}h title={item['title']}"
            )
        lines.append("")
    else:
        lines.append("- No stale in-progress beads detected.")
        lines.append("")

    duplicates = findings["duplicate_cross_listed_epics"]
    if duplicates:
        lines.append("### Cross-listed duplicate epics")
        lines.append("")
        for item in duplicates:
            ids = ", ".join(f"`{issue_id}`" for issue_id in item["ids"])
            lines.append(f"- {ids} ({item['kind']})")
        lines.append("")

    contradictions = findings["umbrella_progress_contradictions"]
    if contradictions:
        lines.append("### Umbrella progress contradictions")
        lines.append("")
        for item in contradictions:
            parent = item["parent"]
            methods = ", ".join(
                f"`{entry['method']}`" for entry in item["listed_missing_methods_with_closed_children"]
            )
            lines.append(f"- `{parent['id']}`: listed-missing methods with closed children: {methods}")
        lines.append("")

    parent_notes = findings["missing_parent_progress_notes"]
    if parent_notes:
        lines.append("### Parent progress notes missing")
        lines.append("")
        for item in parent_notes:
            parent = item["parent"]
            count = len(item["closed_children_after_parent_update"])
            latest = item["latest_child_closed_at"]
            lines.append(
                f"- `{parent['id']}`: {count} child closeout(s) after parent update; latest `{latest}`"
            )
        lines.append("")

    proposals = report["proposed_updates"]
    lines.extend(["## Proposed Parent Updates", ""])
    if not proposals:
        lines.append("- No parent update proposals.")
    for proposal in proposals:
        lines.extend(
            [
                f"### `{proposal['parent_id']}`",
                "",
                "Append the following reviewed progress note:",
                "",
                "```markdown",
                proposal["exact_append_text"].strip("\n"),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issues",
        type=Path,
        default=DEFAULT_ISSUES,
        help="Path to issues JSONL (default: .beads/issues.jsonl)",
    )
    parser.add_argument(
        "--as-of",
        help="UTC-ish ISO timestamp for stale-age checks; default is max tracker timestamp.",
    )
    parser.add_argument(
        "--stale-hours",
        type=float,
        default=24.0,
        help="Age threshold for in_progress stale findings (default: 24).",
    )
    parser.add_argument("--json-out", type=Path, help="Write JSON report to this path.")
    parser.add_argument("--markdown-out", type=Path, help="Write Markdown report to this path.")
    parser.add_argument(
        "--fail-on-structural",
        action="store_true",
        help="Exit 1 if dependency cycles or parent-child integrity findings are present.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    issues_path = args.issues
    if not issues_path.is_absolute():
        issues_path = REPO_ROOT / issues_path
    try:
        if args.as_of:
            as_of = parse_timestamp(args.as_of)
            if as_of is None:
                raise ValueError("--as-of must not be empty")
        else:
            _, ordered = load_issues(issues_path)
            as_of = max_issue_timestamp(ordered)
        report = build_report(issues_path, as_of, args.stale_hours)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rendered_json = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        out = args.json_out if args.json_out.is_absolute() else REPO_ROOT / args.json_out
        write_text(out, rendered_json)
    else:
        print(rendered_json, end="")

    if args.markdown_out:
        out = (
            args.markdown_out
            if args.markdown_out.is_absolute()
            else REPO_ROOT / args.markdown_out
        )
        write_text(out, render_markdown(report))

    structural_failed = (
        report["summary"]["finding_counts"]["dependency_cycles"] > 0
        or report["summary"]["finding_counts"]["parent_child_integrity"] > 0
    )
    if args.fail_on_structural and structural_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
