"""Outbound network control.

The strongest single mitigation against data exfiltration by a compromised agent is that
it has nowhere to send the data. An agent that has been talked into embedding a customer
record in a URL still cannot reach the attacker's host if the host is not on the allowlist.

Enforcement is allowlist-only. A denylist assumes you can enumerate every bad destination,
which you cannot, and a single missed entry is a total bypass.

The checks below are the application-layer half. The other half belongs in the network:
a NetworkPolicy or egress firewall that stops a process bypassing this code entirely. This
module documents the intended set and gives the runtime a fast path; it is not a substitute
for the network control, and the deployment manifests carry the matching rules.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

from eap.platform.errors import PolicyViolation

# Ranges that must never be reachable from a tool call. A URL resolving into one of these
# is a server-side request forgery attempt: cloud metadata endpoints, loopback services and
# anything on the cluster's internal network.
BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",  # link-local; covers 169.254.169.254 cloud metadata
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.168.0.0/16",
        "198.18.0.0/15",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)


@dataclass(frozen=True, slots=True)
class EgressDecision:
    allowed: bool
    host: str
    reason: str | None = None


class EgressGuard:
    """Validates an outbound URL against an allowlist and the blocked-network set."""

    def __init__(self, allowlist: tuple[str, ...], *, resolve: bool = True) -> None:
        self._allowlist = tuple(host.lower().lstrip(".") for host in allowlist)
        self._resolve = resolve

    def check(self, url: str) -> EgressDecision:
        parsed = urlparse(url)

        if parsed.scheme not in ("https",):
            return EgressDecision(False, parsed.hostname or "", "only https egress is permitted")

        host = (parsed.hostname or "").lower()
        if not host:
            return EgressDecision(False, "", "url has no host")

        if not self._host_allowed(host):
            return EgressDecision(False, host, f"host '{host}' is not on the egress allowlist")

        if self._resolve:
            blocked = self._resolves_into_blocked_range(host)
            if blocked is not None:
                # An allowlisted name that resolves to a private address is DNS rebinding.
                return EgressDecision(False, host, f"host resolves to blocked address {blocked}")

        return EgressDecision(True, host)

    def enforce(self, url: str) -> EgressDecision:
        decision = self.check(url)
        if not decision.allowed:
            raise PolicyViolation(
                decision.reason or "egress denied",
                rule="netops.egress_allowlist",
                host=decision.host,
            )
        return decision

    def _host_allowed(self, host: str) -> bool:
        for allowed in self._allowlist:
            if host == allowed or host.endswith(f".{allowed}"):
                return True
        return False

    @staticmethod
    def _resolves_into_blocked_range(host: str) -> str | None:
        try:
            infos = socket.getaddrinfo(host, None)
        except socket.gaierror:
            return None  # Unresolvable: the connection will fail on its own merits.
        for info in infos:
            # sockaddr is (host, port) for IPv4 and (host, port, flowinfo, scope_id) for
            # IPv6, so the first element is a str in both cases even though the tuple type
            # is not narrow enough to say so.
            address = str(info[4][0])
            try:
                parsed = ipaddress.ip_address(address)
            except ValueError:
                continue
            if any(parsed in network for network in BLOCKED_NETWORKS):
                return address
        return None
