"""Atomic shared breaker backend with expiring permits.

Callers MUST bound the entire admitted operation below lease_seconds. Expiry is
crash recovery, not cancellation of a live operation. Runtime integration must
establish that invariant before enabling this backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from threading import Lock
from uuid import uuid4

from redis.exceptions import RedisError

from .circuit import CircuitPolicy, DependencyUnavailable


_SCRIPT = r'''
local root = KEYS[1]
local op, key, token, outcome = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local policy = ARGV[5]
local p = cjson.decode(policy)
local threshold, recovery, capacity, maxkeys, lifetime = p[1], p[2], p[3], p[4], p[5]
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
local server_id = string.match(redis.call('INFO', 'server'), 'run_id:([%w]+)')
if not server_id then return {'DEPENDENCY_COORDINATOR_STATE_INVALID', 1} end
local configured = redis.call('HGET', root, '__policy')
if op == 'initialize' and redis.call('EXISTS', root) == 1 then
  return {'DEPENDENCY_NAMESPACE_EXISTS', 1}
end
if configured and configured ~= policy then return {'DEPENDENCY_POLICY_MISMATCH', 1} end
if not configured then
  if redis.call('EXISTS', root) == 1 then return {'DEPENDENCY_COORDINATOR_STATE_INVALID', 1} end
  -- Runtime cannot distinguish a first start from data loss. Wait long enough
  -- for every old permit to expire before admitting work into empty state.
  redis.call('HSET', root, '__policy', policy, '__not_before', op == 'initialize' and 0 or now + lifetime, '__server_id', server_id)
end
local recorded_server = redis.call('HGET', root, '__server_id')
if not recorded_server then return {'DEPENDENCY_COORDINATOR_STATE_INVALID', 1} end
if recorded_server ~= server_id then
  -- A persisted snapshot can contain stale counters after restart/failover.
  redis.call('HSET', root, '__server_id', server_id, '__not_before', now + lifetime)
  local previous = redis.call('HGETALL', root)
  for i=1,#previous,2 do
    if string.sub(previous[i], 1, 2) ~= '__' then
      local old = cjson.decode(previous[i+1])
      old.leases = {}; old.probe = ''; old.epoch = old.epoch + 1
      redis.call('HSET', root, previous[i], cjson.encode(old))
    end
  end
end
local ready = tonumber(redis.call('HGET', root, '__not_before'))
if not ready then return {'DEPENDENCY_COORDINATOR_STATE_INVALID', 1} end
local metrics = cjson.decode(redis.call('HGET', root, '__metrics') or '{"rejected_total":0,"opened_total":0}')
local function save(k, s) redis.call('HSET', root, k, cjson.encode(s)) end
local function count(s) local n = 0; for _ in pairs(s.leases) do n = n + 1 end; return n end
local function clean(s)
  for id, lease in pairs(s.leases) do
    if lease.until_ms <= now then
      s.leases[id] = nil
      if s.probe == id then
        s.probe = ''; s.open_until = now + recovery; s.epoch = s.epoch + 1
      end
    end
  end
end
local function reject(reason, delay)
  if op == 'acquire' then
    metrics.rejected_total = metrics.rejected_total + 1
    redis.call('HSET', root, '__metrics', cjson.encode(metrics))
  end
  return {reason, math.max(1, math.ceil(delay / 1000))}
end
redis.call('HSET', root, '__metrics', cjson.encode(metrics))
if op == 'initialize' then return {'OK', 1} end
if (op == 'acquire' or op == 'hint' or op == 'validate' or op == 'key_snapshot') and ready > now then
  return reject('DEPENDENCY_COORDINATOR_RECOVERING', ready - now)
end
if op == 'snapshot' then
  local result = {keys=0, open=0, half_open=0, in_flight=0, recovering=ready > now and 1 or 0,
    latched=0, rejected_total=metrics.rejected_total, opened_total=metrics.opened_total}
  local all = redis.call('HGETALL', root)
  for i=1,#all,2 do
    if string.sub(all[i], 1, 2) ~= '__' then
      local s = cjson.decode(all[i+1]); clean(s); save(all[i], s)
      result.keys = result.keys + 1; result.in_flight = result.in_flight + count(s)
      if s.open_until > 0 or s.latched then result.open = result.open + 1 end
      if s.probe ~= '' then result.half_open = result.half_open + 1 end
      if s.latched then result.latched = result.latched + 1 end
    end
  end
  return {'OK', cjson.encode(result)}
end
local raw = redis.call('HGET', root, key)
local s = raw and cjson.decode(raw) or nil
if s then clean(s); save(key, s) end
if op == 'key_snapshot' then
  if not s then return {'OK', '{"open":0,"half_open":0,"in_flight":0,"failures":0,"latched":0}'} end
  local result = {open=(s.open_until > 0 or s.latched) and 1 or 0,
    half_open=s.probe ~= '' and 1 or 0, in_flight=count(s), failures=s.failures,
    latched=s.latched and 1 or 0}
  return {'OK', cjson.encode(result)}
end
if op == 'validate' then
  if not s or not s.leases[token] or s.leases[token].epoch ~= s.epoch then
    return {'DEPENDENCY_PERMIT_LOST', 1}
  end
  return {'OK', s.leases[token].until_ms - now}
end
if op == 'finish' then
  if not s or not s.leases[token] then return {'OK', 0} end
  local lease = s.leases[token]; s.leases[token] = nil
  if lease.epoch == s.epoch then
    if outcome == 'failure' then
      s.failures = s.failures + 1
      if lease.probe or s.failures >= threshold then
        s.open_until = now + recovery; s.probe = ''; s.epoch = s.epoch + 1
        metrics.opened_total = metrics.opened_total + 1
      end
    elseif outcome == 'success' then
      s.failures = 0
      if lease.probe then s.open_until = 0; s.probe = ''; s.epoch = s.epoch + 1 end
    elseif lease.probe then s.probe = ''; s.open_until = now + recovery end
  end
  save(key, s); redis.call('HSET', root, '__metrics', cjson.encode(metrics))
  return {'OK', 1}
end
if not s then
  if op == 'hint' then return {'OK', 0} end
  if redis.call('HLEN', root) - 4 >= maxkeys then
    local all = redis.call('HGETALL', root)
    for i=1,#all,2 do
      if string.sub(all[i], 1, 2) ~= '__' then
        local candidate = cjson.decode(all[i+1]); clean(candidate); save(all[i], candidate)
        if candidate.open_until == 0 and candidate.failures == 0 and count(candidate) == 0 then
          redis.call('HDEL', root, all[i]); break
        end
      end
    end
    if redis.call('HLEN', root) - 4 >= maxkeys then return reject('DEPENDENCY_REGISTRY_FULL', 1000) end
  end
  s = {failures=0, epoch=0, open_until=0, probe='', leases={}, latched=false}
end
if op == 'force' then
  s.failures = math.max(s.failures, threshold); s.open_until = now + recovery
  s.probe = ''; s.epoch = s.epoch + 1; s.latched = outcome == 'latched'
  metrics.opened_total = metrics.opened_total + 1
  save(key, s); redis.call('HSET', root, '__metrics', cjson.encode(metrics))
  return {'OK', 1}
end
if s.latched then return reject('DEPENDENCY_CIRCUIT_LATCHED', lifetime) end
if s.open_until > 0 and (now < s.open_until or s.probe ~= '') then
  return reject('DEPENDENCY_CIRCUIT_OPEN', s.open_until - now)
end
if count(s) >= capacity then return reject('DEPENDENCY_CAPACITY_EXHAUSTED', 1000) end
if op == 'hint' then return {'OK', 0} end
local probe = s.open_until > 0
if probe then s.probe = token end
s.leases[token] = {until_ms=now + lifetime, epoch=s.epoch, probe=probe}
save(key, s)
return {'OK', 1}
'''


class RedisCircuitRegistry:
    def __init__(self, client, *, namespace: str, policy: CircuitPolicy = CircuitPolicy(), lease_seconds: float = 900):
        if not namespace or len(namespace) > 128 or not all(c.isalnum() or c in '-_' for c in namespace):
            raise ValueError('invalid shared circuit namespace')
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError('invalid shared permit lifetime')
        self.client = client
        self.policy = policy
        self.lease_seconds = lease_seconds
        self._root = f'deepevol:circuits:{{{namespace}}}'
        self._policy = json.dumps([policy.failure_threshold, math.ceil(policy.recovery_seconds * 1000),
            policy.max_in_flight, policy.max_keys, math.ceil(lease_seconds * 1000)], separators=(',', ':'))
        self.coordinator_errors = 0
        self._lifecycle_lock = Lock()
        self._closed = False
        self._active_calls = 0
        self._client_closed = False

    @staticmethod
    def _key(key):
        if not isinstance(key, str) or not key or len(key) > 2048:
            raise ValueError('invalid dependency key')
        return hashlib.sha256(key.encode()).hexdigest()

    def _call(self, operation, key='', token='', outcome='neutral'):
        with self._lifecycle_lock:
            if self._closed:
                raise DependencyUnavailable('DEPENDENCY_COORDINATOR_CLOSED', 1)
            self._active_calls += 1
        try:
            try:
                result = self.client.eval(_SCRIPT, 1, self._root, operation, key, token, outcome, self._policy)
            except RedisError:
                self.coordinator_errors += 1
                raise DependencyUnavailable('DEPENDENCY_COORDINATOR_UNAVAILABLE', 1) from None
        finally:
            close_client = False
            with self._lifecycle_lock:
                self._active_calls -= 1
                if self._closed and self._active_calls == 0 and not self._client_closed:
                    self._client_closed = True
                    close_client = True
            if close_client:
                self.client.close()
        status = result[0].decode() if isinstance(result[0], bytes) else result[0]
        if status != 'OK':
            raise DependencyUnavailable(status, float(result[1]))
        return result[1]

    def initialize_new_namespace(self):
        """Operator-only: caller must prove the namespace has NEVER been used.

        Never call during runtime startup/recovery. A missing key may mean lost
        state with live requests; normal acquire enters quarantine in that case.
        """
        self._call('initialize')

    def acquire(self, key):
        digest, token = self._key(key), uuid4().hex
        started = time.monotonic()
        self._call('acquire', digest, token)
        return SharedPermit(self, digest, token, egress_deadline=started + self.lease_seconds * 0.9)

    def check_available(self, key):
        self._call('hint', self._key(key))

    def key_snapshot(self, key):
        return json.loads(self._call('key_snapshot', self._key(key)))

    def force_open(self, key, *, latched=False):
        self._call('force', self._key(key), outcome='latched' if latched else 'failure')

    def snapshot(self):
        return json.loads(self._call('snapshot'))

    def close(self):
        close_client = False
        with self._lifecycle_lock:
            self._closed = True
            if self._active_calls == 0 and not self._client_closed:
                self._client_closed = True
                close_client = True
        if close_client:
            self.client.close()

    def prometheus_metrics(self, *, component=None):
        if component not in {None, "gateway", "http", "s3", "stream"}:
            raise ValueError("unknown circuit metrics component")
        labels = "" if component is None else f'{{component="{component}"}}'
        try:
            state = self.snapshot()
        except DependencyUnavailable:
            return f"deepevol_dependency_coordinator_available{labels} 0\n"
        return f"deepevol_dependency_coordinator_available{labels} 1\n" + "".join(
            f"deepevol_dependency_circuit_{key}{labels} {value}\n" for key, value in state.items()
        )


class SharedPermit:
    def __init__(self, registry, key, token, *, egress_deadline=None):
        self.egress_deadline_monotonic = (
            time.monotonic() + registry.lease_seconds * 0.9 if egress_deadline is None else egress_deadline
        )
        self.registry, self.key, self.token = registry, key, token
        self.finished = False
        self._lock = Lock()

    def validate(self):
        if self.finished or time.monotonic() >= self.egress_deadline_monotonic:
            raise DependencyUnavailable('DEPENDENCY_PERMIT_EXPIRED', 1)
        remaining_ms = self.registry._call('validate', self.key, self.token)
        self.egress_deadline_monotonic = min(
            self.egress_deadline_monotonic, time.monotonic() + float(remaining_ms) * 0.0009,
        )
        if time.monotonic() >= self.egress_deadline_monotonic:
            raise DependencyUnavailable('DEPENDENCY_PERMIT_EXPIRED', 1)
        return self.egress_deadline_monotonic

    def finish(self, outcome='neutral'):
        if outcome not in {'success', 'failure', 'neutral'}:
            raise ValueError('invalid circuit outcome')
        with self._lock:
            if self.finished:
                return
            self.finished = True
        try:
            self.registry._call('finish', self.key, self.token, outcome)
        except DependencyUnavailable:
            # A coordination failure after a billable call cannot change its
            # business outcome. The orphan permit expires without being reused.
            pass
