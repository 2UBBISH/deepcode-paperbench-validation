"""Protect shared breaker coordination itself from repeated network failures."""

import hashlib

from redis.exceptions import ConnectionError, RedisError

from .circuit import CircuitPolicy, CircuitRegistry, DependencyUnavailable

_COORDINATORS = CircuitRegistry(CircuitPolicy(failure_threshold=2, recovery_seconds=5, max_in_flight=4, max_keys=64))


class ProtectedCoordinatorClient:
    def __init__(self, client, *, identity: str, circuits=None):
        self.client = client
        self.circuits = circuits if circuits is not None else _COORDINATORS
        self.key = hashlib.sha256(identity.encode()).hexdigest()

    def eval(self, *args):
        try:
            permit = self.circuits.acquire(self.key)
        except DependencyUnavailable:
            # Preserve RedisCircuitRegistry's existing fail-closed mapping.
            raise ConnectionError("Circuit coordinator temporarily unavailable") from None
        try:
            try:
                result = self.client.eval(*args)
            except RedisError:
                permit.finish("failure")
                raise
            permit.finish("success")
            return result
        finally:
            permit.finish()

    def close(self):
        self.client.close()

    def __getattr__(self, name):
        return getattr(self.client, name)
