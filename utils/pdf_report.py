from __future__ import annotations

import os
from typing import Iterable

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _paragraphs(items: Iterable[str], style):
    return [ListItem(Paragraph(str(item), style)) for item in items if str(item).strip()]


def build_posture_pdf(output_path: str, payload: dict) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    panel = payload.get("panel", {})
    session_summary = payload.get("session_summary", {})
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodySmall", parent=styles["BodyText"], fontSize=10, leading=14))
    styles.add(ParagraphStyle(name="SectionTitle", parent=styles["Heading2"], fontSize=15, textColor=colors.HexColor("#1d2a2f")))

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Flucture AI Posture Report",
    )

    story = []
    story.append(Paragraph("Flucture AI Posture Report", styles["Title"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(panel.get("overall_assessment", "No assessment available."), styles["BodyText"]))
    story.append(Spacer(1, 10))

    summary_table = Table(
        [
            ["Risk Level", str(panel.get("risk_level", "unknown")).title()],
            ["Confidence", str(session_summary.get("session_confidence", "low")).title()],
            ["Frames Reviewed", str(session_summary.get("total_frames", 0))],
            ["Progress Score", str(panel.get("progress_score", {}).get("current_score", 0))],
        ],
        colWidths=[1.8 * inch, 3.8 * inch],
    )
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f2ea")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#1d2a2f")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d8d2c9")),
                ("PADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 14))

    story.append(Paragraph("What Is Wrong", styles["SectionTitle"]))
    for issue in panel.get("what_is_wrong", []):
        story.append(
            Paragraph(
                f"<b>{issue.get('issue', 'Issue')}</b> ({issue.get('severity', 'unknown')})",
                styles["BodyText"],
            )
        )
        evidence = issue.get("evidence", [])
        if evidence:
            story.append(ListFlowable(_paragraphs(evidence, styles["BodySmall"]), bulletType="bullet"))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Possible Consequences", styles["SectionTitle"]))
    for item in panel.get("possible_consequences", []):
        story.append(Paragraph(f"<b>{item.get('issue', 'Issue')}</b>", styles["BodyText"]))
        story.append(ListFlowable(_paragraphs(item.get("risks", []), styles["BodySmall"]), bulletType="bullet"))
        story.append(Spacer(1, 6))

    story.append(Paragraph("Improvement Plan", styles["SectionTitle"]))
    for item in panel.get("improvement_plan", []):
        story.append(
            Paragraph(
                f"<b>Priority {item.get('priority', '?')}:</b> {item.get('action', '')}",
                styles["BodyText"],
            )
        )
        story.append(Paragraph(item.get("reason", ""), styles["BodySmall"]))
        story.append(Spacer(1, 6))

    remedies = panel.get("remedies", {})
    remedy_sections = [
        ("Stretches", remedies.get("stretches", [])),
        ("Strengthening", remedies.get("strengthening", [])),
        ("Daily Habits", remedies.get("daily_habits", [])),
        ("Ergonomic Corrections", remedies.get("ergonomic_corrections", [])),
    ]
    story.append(Paragraph("Remedies", styles["SectionTitle"]))
    for title, items in remedy_sections:
        story.append(Paragraph(f"<b>{title}</b>", styles["BodyText"]))
        story.append(ListFlowable(_paragraphs(items, styles["BodySmall"]), bulletType="bullet"))
        story.append(Spacer(1, 6))

    red_flags = panel.get("red_flags", [])
    if red_flags:
        story.append(Paragraph("When To Seek Extra Help", styles["SectionTitle"]))
        story.append(ListFlowable(_paragraphs(red_flags, styles["BodySmall"]), bulletType="bullet"))

    doc.build(story)
    return output_path
