from __future__ import annotations

from typing import Any, TYPE_CHECKING

from utils import normalize_tags

if TYPE_CHECKING:
    from recommender import GameRecord


def tag_overlap_score(record: GameRecord, wanted_tags: set[str]) -> float:
    """
    Score how well a game's tags align with the user's desired tags.

    Parameters
    ----------
    record:
        The game candidate to score.
    wanted_tags:
        Lowercase set of desired Steam tags extracted from the user's query.

    Returns
    -------
    float
        Score in ``[0.0, 1.0]``.
    """
    if not wanted_tags:
        return 0.5

    game_tags: set[str] = {
        t.lower()
        for t in normalize_tags(record.raw.get("tags")) + record.raw.get("genres", [])
    }

    hits = sum(
        1
        for wanted in wanted_tags
        if any(wanted in game_tag or game_tag in wanted for game_tag in game_tags)
    )
    if hits == 0:
        return 0.0

    return min(1.0, hits / max(1, len(wanted_tags)))


def filter_candidates(
    candidates: list[tuple[GameRecord, float]],
    constraints: dict[str, Any],
) -> list[tuple[GameRecord, float]]:
    """
    Remove candidates whose tags overlap with any explicitly blocked tag.

    A candidate is removed if **any** of its tags or genres appear in the
    ``blocked_tags`` constraint set.  Matching is exact (after lowercasing)
    because blocked tags are already normalised by :func:`~query_parser.extract_tags`.

    Parameters
    ----------
    candidates:
        ``(GameRecord, chroma_distance)`` pairs from the ChromaDB query,
        ordered by ascending distance (closest first).
    constraints:
        Constraint dict from :func:`~query_parser.parse_constraints`.
        Only ``"blocked_tags"`` is consumed here; all other keys are ignored.

    Returns
    -------
    list[tuple[GameRecord, float]]
        Filtered candidate list, or the original list if filtering removes
        everything.
    """
    blocked: set[str] | None = constraints.get("blocked_tags")
    if not blocked:
        return candidates

    filtered = [
        (record, dist)
        for record, dist in candidates
        if not (
            {
                t.lower()
                for t in (
                    normalize_tags(record.raw.get("tags"))
                    + record.raw.get("genres", [])
                )
            }
            & blocked
        )
    ]

    # avoid returning an empty list; fall back to the unfiltered pool instead
    return filtered or candidates
