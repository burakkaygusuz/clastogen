from typing import TypedDict


class BaselinePayload(TypedDict):
    test_id: str
    target: str
    successes: int
    runs: int
    p0: float
    stable: bool
    error: str | None


class ExecutionPayload(TypedDict):
    test_id: str
    target: str
    mutant_id: str
    description: str
    status: str
    sample_count: int | None
    llr: float | None
    error: str | None


class RecordsPayload(TypedDict):
    """Records of one marked test; execnet serializes only exact builtin types, so statuses travel as plain str."""

    baselines: list[BaselinePayload]
    results: list[ExecutionPayload]
