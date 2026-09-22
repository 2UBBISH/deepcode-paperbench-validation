"""Bounded PostgreSQL connection pools used by the V2 runtime.

The V2 repositories pre-date pooling and therefore use two slightly different
patterns: some call ``connection.close()`` explicitly while others use
``with connection``.  ``PoolRegistry.factory()`` returns a small lease proxy
which supports both patterns and always returns the underlying connection to
the pool.  This keeps the migration incremental while making an unbounded
``psycopg.connect()`` impossible for runtime code that uses the registry.

Per-session governance (``statement_timeout`` and friends) is deliberately
*not* applied from here.  A pool that issues ``SET``/``set_config`` at
checkout binds the value to whichever backend a transaction-pooling PgBouncer
happened to assign, so the setting both leaks to unrelated clients of the same
PgBouncer pool and is absent from every other backend.  The values live as
role defaults instead (``ALTER ROLE ... SET``, applied by PostgreSQL at backend
start and therefore invisible to any pooler); see
``apps.v2.database.runtime_principals.provision_runtime_principal``.  What the
pool does here is *verify* that the governance actually reached the session, so
a role that was never provisioned surfaces as a metric instead of as an
unbounded query.
"""

from __future__ import annotations

import os
import math
import logging
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable, Iterable, Mapping

from psycopg import Connection
from psycopg.rows import tuple_row
from psycopg_pool import ConnectionPool, PoolTimeout


logger = logging.getLogger(__name__)


def _env_int(source: Mapping[str, str], key: str, default: int, *, minimum: int = 0) -> int:
    raw = source.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _env_float(
    source: Mapping[str, str], key: str, default: float, *, minimum: float = 0.0
) -> float:
    raw = source.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


@dataclass(frozen=True, slots=True)
class PoolSettings:
    """Safe defaults and bounded knobs shared by all V2 database roles."""

    min_size: int = 0
    # Two connections per role keeps the aggregate budget safe when Product
    # runs alongside Asset, cleaners, scanners and schedulers on one database.
    # Operators can raise this only after the cross-process budget check.
    max_size: int = 2
    timeout_seconds: float = 5.0
    connect_timeout_seconds: float = 5.0
    max_lifetime_seconds: float = 1800.0
    max_idle_seconds: float = 300.0
    max_waiting: int = 100
    statement_timeout_ms: int = 30_000
    lock_timeout_ms: int = 5_000
    idle_in_transaction_timeout_ms: int = 60_000
    slow_query_log_ms: int = 1_000
    leak_threshold_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.min_size < 0 or self.max_size < 1 or self.min_size > self.max_size:
            raise ValueError("pool sizes must satisfy 0 <= min_size <= max_size")
        for name in (
            "timeout_seconds",
            "connect_timeout_seconds",
            "max_lifetime_seconds",
            "max_idle_seconds",
            "leak_threshold_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in (
            "max_waiting",
            "statement_timeout_ms",
            "lock_timeout_ms",
            "idle_in_transaction_timeout_ms",
            "slow_query_log_ms",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def role_session_defaults(self) -> tuple[tuple[str, int], ...]:
        """Return the ``(guc, milliseconds)`` pairs every runtime LOGIN carries.

        Milliseconds are the base unit PostgreSQL reports for these settings in
        ``pg_settings.setting``, so the same pairs drive both the provisioning
        statements and the pool's verification without a unit conversion.

        These are provisioned as role defaults rather than issued by the pool,
        because a pooled ``SET`` does not survive transaction pooling.  Only
        settings a non-superuser runtime role may hold belong here;
        ``log_min_duration_statement`` is superuser-only and is configured at
        the server level by the V2 database Compose file instead.  A configured
        ``0`` means "deliberately unbounded" and verifies as satisfied.
        """

        return (
            ("statement_timeout", self.statement_timeout_ms),
            ("lock_timeout", self.lock_timeout_ms),
            ("idle_in_transaction_session_timeout", self.idle_in_transaction_timeout_ms),
        )

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, prefix: str = "DEEPEVOL_V2_DB_POOL_"
    ) -> "PoolSettings":
        source = os.environ if environ is None else environ
        defaults = cls()
        return cls(
            min_size=_env_int(source, prefix + "MIN_SIZE", defaults.min_size),
            max_size=_env_int(source, prefix + "MAX_SIZE", defaults.max_size, minimum=1),
            timeout_seconds=_env_float(source, prefix + "WAIT_TIMEOUT_SECONDS", defaults.timeout_seconds, minimum=0.001),
            connect_timeout_seconds=_env_float(source, prefix + "CONNECT_TIMEOUT_SECONDS", defaults.connect_timeout_seconds, minimum=0.001),
            max_lifetime_seconds=_env_float(source, prefix + "MAX_LIFETIME_SECONDS", defaults.max_lifetime_seconds, minimum=1.0),
            max_idle_seconds=_env_float(source, prefix + "MAX_IDLE_SECONDS", defaults.max_idle_seconds, minimum=1.0),
            max_waiting=_env_int(source, prefix + "MAX_WAITING", defaults.max_waiting),
            statement_timeout_ms=_env_int(source, prefix + "STATEMENT_TIMEOUT_MS", defaults.statement_timeout_ms),
            lock_timeout_ms=_env_int(source, prefix + "LOCK_TIMEOUT_MS", defaults.lock_timeout_ms),
            idle_in_transaction_timeout_ms=_env_int(
                source,
                prefix + "IDLE_IN_TRANSACTION_TIMEOUT_MS",
                defaults.idle_in_transaction_timeout_ms,
            ),
            slow_query_log_ms=_env_int(
                source,
                prefix + "SLOW_QUERY_LOG_MS",
                defaults.slow_query_log_ms,
            ),
            leak_threshold_seconds=_env_float(
                source,
                prefix + "LEAK_THRESHOLD_SECONDS",
                defaults.leak_threshold_seconds,
                minimum=0.001,
            ),
        )


@dataclass(frozen=True, slots=True)
class PoolBudget:
    """A deploy-time connection budget for one PostgreSQL server."""

    database_max_connections: int
    reserved_for_superuser: int = 3
    reserved_for_operations: int = 5

    @classmethod
    def from_env(
        cls, environ: Mapping[str, str] | None = None, *, prefix: str = "DEEPEVOL_V2_DB_"
    ) -> "PoolBudget":
        source = os.environ if environ is None else environ
        defaults = cls(100)
        return cls(
            database_max_connections=_env_int(source, prefix + "MAX_CONNECTIONS", defaults.database_max_connections, minimum=1),
            reserved_for_superuser=_env_int(source, prefix + "RESERVED_SUPERUSER", defaults.reserved_for_superuser),
            reserved_for_operations=_env_int(source, prefix + "RESERVED_OPERATIONS", defaults.reserved_for_operations),
        )

    def validate(self, pools: Mapping[str, PoolSettings]) -> None:
        if self.database_max_connections < 1:
            raise ValueError("database_max_connections must be positive")
        if self.reserved_for_superuser < 0 or self.reserved_for_operations < 0:
            raise ValueError("connection reservations must be non-negative")
        total = sum(pool.max_size for pool in pools.values())
        budget = self.database_max_connections - self.reserved_for_superuser - self.reserved_for_operations
        if total > budget:
            raise ValueError(
                f"configured pool budget {total} exceeds database budget {budget} "
                f"({self.database_max_connections} max_connections)"
            )


def verify_session_governance(
    connection: Connection[Any], expected: Iterable[tuple[str, int]]
) -> tuple[str, ...]:
    """Report which contracted role defaults did not reach this session.

    The pool reads the governance back rather than writing it: the values are
    role defaults applied by PostgreSQL at backend start, so a mismatch here
    means the LOGIN was never provisioned (or was reset) and its queries are
    running unbounded.

    ``pg_settings.setting`` reports these timeouts in milliseconds, matching
    ``PoolSettings.role_session_defaults()``, so no unit conversion is needed.
    An explicit ``tuple_row`` cursor is required: several pools register
    ``connect_kwargs={"row_factory": dict_row}``, and positional row access
    would silently read the wrong column on those.
    """

    wanted = tuple(expected)
    with connection.cursor(row_factory=tuple_row) as cursor:
        cursor.execute(
            "SELECT name, setting FROM pg_settings WHERE name = ANY(%s)",
            ([name for name, _ in wanted],),
        )
        observed = {str(row[0]): str(row[1]) for row in cursor.fetchall()}
    return tuple(
        f"{name}={observed.get(name, 'missing')} (expected {value}ms)"
        for name, value in wanted
        if observed.get(name) != str(value)
    )


@dataclass(frozen=True, slots=True)
class LeaseInfo:
    pool: str
    acquired_at: float
    age_seconds: float


class _ConnectionLease:
    """Connection-compatible proxy which returns a checked-out connection."""

    __slots__ = ("_acquired_at", "_name", "_pool", "_raw", "_registry", "_released")

    def __init__(
        self,
        raw: Connection[Any],
        pool: ConnectionPool[Any],
        registry: "PoolRegistry",
        name: str,
    ) -> None:
        self._raw = raw
        self._pool = pool
        self._registry = registry
        self._name = name
        self._acquired_at = monotonic()
        self._released = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)

    @property
    def row_factory(self) -> Any:
        """Expose psycopg's connection-level row factory through the lease.

        A few incremental adapters still set ``connection.row_factory``
        before issuing a cursor.  The lease deliberately proxies the
        connection API, so dropping this attribute at the pool boundary would
        turn an otherwise valid request into ``AttributeError`` at runtime.
        New code should prefer ``connection.cursor(row_factory=...)``; this
        compatibility property keeps the pooled and direct connection paths
        behaviorally equivalent while those adapters are migrated.
        """

        return self._raw.row_factory

    @row_factory.setter
    def row_factory(self, value: Any) -> None:
        self._raw.row_factory = value

    def __enter__(self) -> "_ConnectionLease":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool | None:
        try:
            # Preserve psycopg's normal commit/rollback semantics for callers
            # using ``with connection``.
            return self._raw.__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()

    def close(self) -> None:
        if self._released:
            return
        self._released = True
        self._registry._release(self._name, self._pool, self._raw, self._acquired_at)


class PoolRegistry:
    """Own all V2 pools and expose factories for legacy repository APIs."""

    def __init__(self, settings: PoolSettings | None = None) -> None:
        self.settings = settings or PoolSettings.from_env()
        self._pools: dict[str, ConnectionPool[Any]] = {}
        self._pool_timeouts: dict[str, float] = {}
        self._leases: dict[int, LeaseInfo] = {}
        self._session_governance: dict[str, tuple[str, ...]] = {}
        self._session_governance_reported: set[str] = set()
        self._lock = threading.Lock()

    def _record_session_governance(self, name: str, faults: tuple[str, ...]) -> None:
        """Record one pool's session-governance verdict, logging it once.

        ``configure`` runs on every new physical connection, so the log is
        emitted only when a pool's verdict changes.  Without that guard a
        misprovisioned role would emit one line per reconnect for the life of
        the process.
        """

        with self._lock:
            previous = self._session_governance.get(name)
            self._session_governance[name] = faults
            already_reported = name in self._session_governance_reported
            if faults:
                self._session_governance_reported.add(name)
            else:
                self._session_governance_reported.discard(name)
        if faults and (not already_reported or previous != faults):
            logger.error(
                "V2 database pool %r is running without its contracted session "
                "governance; the runtime LOGIN needs re-provisioning: %s",
                name,
                "; ".join(faults),
            )
        elif previous:
            logger.info("V2 database pool %r regained its session governance", name)

    def register(
        self,
        name: str,
        dsn: str,
        *,
        application_name: str | None = None,
        settings: PoolSettings | None = None,
        connect_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not name or name in self._pools:
            raise ValueError(f"pool name must be unique: {name!r}")
        config = settings or self.settings
        kwargs: dict[str, Any] = {"connect_timeout": config.connect_timeout_seconds}
        if connect_kwargs:
            kwargs.update(connect_kwargs)
        if application_name:
            kwargs["application_name"] = application_name

        expected = config.role_session_defaults()
        pool_name = name

        def configure(connection: Connection[Any]) -> None:
            faults = verify_session_governance(connection, expected)
            connection.rollback()
            self._record_session_governance(pool_name, faults)

        self._pools[name] = ConnectionPool(
            conninfo=dsn,
            kwargs=kwargs,
            min_size=config.min_size,
            max_size=config.max_size,
            timeout=config.timeout_seconds,
            max_waiting=config.max_waiting,
            max_lifetime=config.max_lifetime_seconds,
            max_idle=config.max_idle_seconds,
            configure=configure,
            check=ConnectionPool.check_connection,
            open=False,
            name=f"v2-{name}",
        )
        self._pool_timeouts[name] = config.timeout_seconds

    def factory(self, name: str, *, deadline: Callable[[], float | None] | None = None) -> Callable[[], _ConnectionLease]:
        if name not in self._pools:
            raise KeyError(f"unknown V2 database pool: {name}")
        pool = self._pools[name]
        pool_timeout = self._pool_timeouts[name]

        def acquire() -> _ConnectionLease:
            absolute = deadline() if deadline is not None else None
            if absolute is not None and (not isinstance(absolute, (int, float)) or isinstance(absolute, bool) or not math.isfinite(absolute)):
                raise ValueError("database acquisition deadline must be finite")

            def remaining() -> float:
                timeout = pool_timeout if absolute is None else min(pool_timeout, absolute - monotonic())
                if timeout <= 0:
                    raise PoolTimeout("database acquisition deadline exhausted")
                return timeout
            # Pools are registered with ``open=False`` so constructing a
            # runtime (or importing a repository in a CLI) never starts a
            # background connector.  Startup probes call the repository
            # factories directly, so open the specific pool on first use.
            # psycopg_pool.open() is idempotent while the pool is alive.
            pool.open(wait=False, timeout=remaining())
            try:
                raw = pool.getconn(timeout=remaining())
            except PoolTimeout:
                raise
            if absolute is not None and monotonic() >= absolute:
                pool.putconn(raw)
                raise PoolTimeout("database acquisition completed after deadline")
            lease = _ConnectionLease(raw, pool, self, name)
            with self._lock:
                self._leases[id(lease)] = LeaseInfo(name, lease._acquired_at, 0.0)
            return lease

        return acquire

    def _release(
        self,
        name: str,
        pool: ConnectionPool[Any],
        raw: Connection[Any],
        acquired_at: float,
    ) -> None:
        pool.putconn(raw)
        with self._lock:
            for key, info in tuple(self._leases.items()):
                if info.pool == name and info.acquired_at == acquired_at:
                    del self._leases[key]
                    break

    def open_all(self, *, wait: bool = False, timeout: float | None = None) -> None:
        for pool in self._pools.values():
            pool.open(wait=wait, timeout=timeout or self.settings.timeout_seconds)

    def close(self) -> None:
        # A pool closes checked-out connections as they are returned, so this is
        # safe during ASGI shutdown even if a request is winding down.
        for pool in self._pools.values():
            pool.close(timeout=self.settings.timeout_seconds)

    def validate_budget(self, budget: PoolBudget | None = None) -> None:
        (budget or PoolBudget.from_env()).validate(
            dict.fromkeys(self._pools, self.settings)
        )

    def stats(self) -> dict[str, dict[str, int]]:
        return {name: dict(pool.get_stats()) for name, pool in self._pools.items()}

    def active_leases(self) -> tuple[LeaseInfo, ...]:
        now = monotonic()
        with self._lock:
            return tuple(
                LeaseInfo(info.pool, info.acquired_at, now - info.acquired_at)
                for info in self._leases.values()
            )

    def leaked_leases(self) -> tuple[LeaseInfo, ...]:
        return tuple(
            lease
            for lease in self.active_leases()
            if lease.age_seconds >= self.settings.leak_threshold_seconds
        )

    def session_governance_faults(self) -> dict[str, tuple[str, ...]]:
        """Return the per-pool session-governance mismatches seen so far.

        A pool is absent until it has opened its first physical connection, and
        maps to an empty tuple once its contracted role defaults verified.
        """

        with self._lock:
            return dict(self._session_governance)

    def ungoverned_pools(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, faults in self.session_governance_faults().items() if faults)
        )

    def prometheus_metrics(self) -> str:
        """Return a dependency-free Prometheus text snapshot for scraping."""
        governance = self.session_governance_faults()
        lines = [
            "# HELP deepevol_v2_db_pool_connections Current pool connection count.",
            "# TYPE deepevol_v2_db_pool_connections gauge",
            "# HELP deepevol_v2_db_pool_requests_total Pool checkout requests.",
            "# TYPE deepevol_v2_db_pool_requests_total counter",
            "# HELP deepevol_v2_db_pool_waiting Current waiters.",
            "# TYPE deepevol_v2_db_pool_waiting gauge",
            "# HELP deepevol_v2_db_pool_leases Active connection leases.",
            "# TYPE deepevol_v2_db_pool_leases gauge",
            "# HELP deepevol_v2_db_pool_leaks Leases older than the configured threshold.",
            "# TYPE deepevol_v2_db_pool_leaks gauge",
            "# HELP deepevol_v2_db_pool_session_governance_faults Contracted role "
            "defaults missing from the pool's sessions.",
            "# TYPE deepevol_v2_db_pool_session_governance_faults gauge",
        ]
        for name, stats in self.stats().items():
            label = name.replace('\\', '\\\\').replace('"', '\\"')
            leases = tuple(lease for lease in self.active_leases() if lease.pool == name)
            lines.extend(
                (
                    f'deepevol_v2_db_pool_connections{{pool="{label}"}} {stats.get("pool_size", 0)}',
                    f'deepevol_v2_db_pool_requests_total{{pool="{label}"}} {stats.get("requests_num", 0)}',
                    # psycopg_pool exposes ``requests_queued`` as a cumulative
                    # counter.  The alerting gauge must use the instantaneous
                    # ``requests_waiting`` value so an old request does not
                    # look like a live queue forever.
                    f'deepevol_v2_db_pool_waiting{{pool="{label}"}} {stats.get("requests_waiting", 0)}',
                    f'deepevol_v2_db_pool_leases{{pool="{label}"}} {len(leases)}',
                    f'deepevol_v2_db_pool_leaks{{pool="{label}"}} {sum(lease.age_seconds >= self.settings.leak_threshold_seconds for lease in leases)}',
                    f'deepevol_v2_db_pool_session_governance_faults{{pool="{label}"}} {len(governance.get(name, ()))}',
                )
            )
        return "\n".join(lines) + "\n"


__all__ = [
    "LeaseInfo",
    "PoolBudget",
    "PoolRegistry",
    "PoolSettings",
    "verify_session_governance",
]
