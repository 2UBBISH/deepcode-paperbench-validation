"""Small process-local probe allowance during coordinator transport failure."""

import time
from threading import Lock

from .circuit import DependencyUnavailable


class EmergencyReadBudget:
    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.lock = Lock()
        self.states = {}
        self.admitted = 0
        self.rejected = 0

    def acquire(self, key):
        with self.lock:
            now = self.clock()
            self.states = {k: v for k, v in self.states.items() if v[1] or v[0] > now}
            state = self.states.get(key)
            if state is not None or len(self.states) >= 64:
                self.rejected += 1
                raise DependencyUnavailable("DEPENDENCY_EMERGENCY_READ_BUDGET_EXHAUSTED", 30)
            self.states[key] = (now + 30, True)
            self.admitted += 1
            return _Permit(self, key, now + 5)

    def metrics(self):
        with self.lock:
            return (f"deepevol_http_emergency_reads_total {self.admitted}\n"
                    f"deepevol_http_emergency_reads_rejected_total {self.rejected}\n"
                    f"deepevol_http_emergency_reads_in_flight {sum(v[1] for v in self.states.values())}\n")


class _Permit:
    def __init__(self, budget, key, deadline):
        self.budget, self.key = budget, key
        self.egress_deadline_monotonic = deadline
        self.finished = False

    def validate(self):
        if self.budget.clock() >= self.egress_deadline_monotonic:
            raise DependencyUnavailable("DEPENDENCY_PERMIT_EXPIRED", 1)
        return self.egress_deadline_monotonic

    def finish(self, outcome="neutral"):
        with self.budget.lock:
            if not self.finished:
                self.finished = True
                deadline, _ = self.budget.states[self.key]
                self.budget.states[self.key] = (deadline, False)


EMERGENCY_READS = EmergencyReadBudget()
