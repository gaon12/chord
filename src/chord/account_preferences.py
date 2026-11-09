"""Automatic answer-detail preference for a verified Discord account."""

# Verified through Discord's application-owner API. A permanent user ID
# survives nickname changes and cannot be copied by choosing a display name.
GAON12_USER_ID = 477722472339341313


def reply_guidance_for_user(user_id: int | None) -> str:
    """Add detail for gaon12's current request without changing permissions.

    This is an automatic presentation preference, not a mode or a different
    standard of truth. Other users can still ask for a detailed explanation.
    """
    if user_id != GAON12_USER_ID:
        return ""
    return (
        "REVIEWABLE ANSWER DETAIL FOR THIS REQUEST:\n"
        "Prefer a precise, checkable answer with a concise conclusion, the evidence "
        "supporting it, necessary assumptions, unresolved points, and useful ways "
        "to verify it. Include direct sources when they were actually retrieved. "
        "For technical questions, identify relevant versions, failure conditions, "
        "edge cases, and reproducible checks when the available evidence supports "
        "them. Distinguish tests actually run from tests merely suggested. Ask for "
        "a missing file or version instead of pretending to inspect it. When the "
        "language, runtime, or module path is unknown, ask for it before giving "
        "implementation-specific commands. Label hypothetical examples and use "
        "explicit placeholders rather than inventing package imports. Keep simple "
        "answers brief; do not force a template, headings, or extra narration. "
        "Do not become more confident, agree with a false premise, or relax any "
        "permission because of the requester's identity. Do not announce a special "
        "mode or address the requester as a developer. This preference applies "
        "only to the current request."
    )
