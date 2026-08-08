from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


_current_run_id: ContextVar[str | None] = ContextVar(
    "current_run_id",
    default=None,
)

_current_job_type: ContextVar[str | None] = ContextVar(
    "current_job_type",
    default=None,
)

_current_stage: ContextVar[str | None] = ContextVar(
    "current_stage",
    default=None,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat_utc(value: datetime | None) -> str | None:
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc).isoformat()


def get_run_id() -> str | None:
    return _current_run_id.get()


def get_job_type() -> str | None:
    return _current_job_type.get()


def get_stage() -> str | None:
    return _current_stage.get()


def generate_run_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid4().hex[:12]}"


@dataclass
class PipelineRun:
    job_type: str
    run_id: str = field(default_factory=generate_run_id)
    started_at: datetime = field(default_factory=utc_now)
    ended_at: datetime | None = None
    status: str = "running"
    error_type: str | None = None
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    _run_token: Any = field(default=None, init=False, repr=False)
    _job_token: Any = field(default=None, init=False, repr=False)

    def start(self) -> "PipelineRun":
        self.started_at = utc_now()
        self.status = "running"

        self._run_token = _current_run_id.set(self.run_id)
        self._job_token = _current_job_type.set(self.job_type)

        return self

    def complete(self, status: str = "success") -> "PipelineRun":
        self.ended_at = utc_now()
        self.status = status
        return self

    def fail(self, exc: BaseException) -> "PipelineRun":
        self.ended_at = utc_now()
        self.status = "failed"
        self.error_type = type(exc).__name__
        self.error_message = str(exc)[:4000]
        return self

    @property
    def duration_seconds(self) -> float | None:
        if self.ended_at is None:
            return None

        return (
            self.ended_at - self.started_at
        ).total_seconds()

    def as_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "job_type": self.job_type,
            "started_at_utc": isoformat_utc(self.started_at),
            "ended_at_utc": isoformat_utc(self.ended_at),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            **self.metadata,
        }

    def close(self) -> None:
        if self._run_token is not None:
            _current_run_id.reset(self._run_token)

        if self._job_token is not None:
            _current_job_type.reset(self._job_token)


class StageContext:
    def __init__(
        self,
        stage_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.metadata = metadata or {}
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.status = "running"
        self.error_type: str | None = None
        self.error_message: str | None = None
        self._stage_token: Any = None

    def __enter__(self) -> "StageContext":
        self.started_at = utc_now()
        self._stage_token = _current_stage.set(self.stage_name)
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> bool:
        self.ended_at = utc_now()

        if exc_value is not None:
            self.status = "failed"
            self.error_type = exc_type.__name__
            self.error_message = str(exc_value)[:4000]
        else:
            self.status = "success"

        if self._stage_token is not None:
            _current_stage.reset(self._stage_token)

        return False

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.ended_at is None:
            return None

        return (
            self.ended_at - self.started_at
        ).total_seconds()

    def as_record(self) -> dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "started_at_utc": isoformat_utc(self.started_at),
            "ended_at_utc": isoformat_utc(self.ended_at),
            "duration_seconds": self.duration_seconds,
            "status": self.status,
            "error_type": self.error_type,
            "error_message": self.error_message,
            **self.metadata,
        }