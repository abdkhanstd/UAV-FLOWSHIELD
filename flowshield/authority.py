"""Authority semantics of client tickets (freshness, single use, revocation, approval of the typed action).

auth.py authenticates where a document came from. This module defines what a verified ticket entitles the sender to, and
for how long:

  * a ticket is bound by its tag to the mission, the channel, an issue time `iat`, an expiry `exp` and a nonce `id`;
  * it is accepted only inside [iat - SKEW, exp) of the edge clock, with a lifetime of at most TTL_MAX;
  * it is single use: the first acceptance consumes the nonce, a rejected presentation does not;
  * it is void if the ticket or the client's credentials issued before a given time were revoked;
  * a ticket that asks for a spatial task additionally needs an approval token of the operator. The token is a tag over
    the ticket id and the typed action that the edge extracted (the resolved request, not its text), so an approval of one
    action cannot be reused for another action or another ticket.

Keys are simulation constants; the attacker models never read this module."""
import hashlib
import hmac
import json

SKEW = 5.0                  # seconds of clock difference tolerated between the portal and the edge
TTL_MAX = 900.0             # longest lifetime a ticket may declare (seconds)
_PORTAL = b"flowshield-simulation-portal-key"
_OPERATOR = b"flowshield-simulation-operator-key"


def _canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _mac(key, payload):
    return hmac.new(key, payload, hashlib.sha256).hexdigest()[:32]


def issue(client, mission, channel, text, now, ttl=300.0, nonce="0"):
    body = {"id": nonce, "client": client, "mission": mission, "channel": channel, "text": text, "iat": float(now), "exp": float(now + ttl)}
    return dict(body, mac=_mac(_PORTAL, _canon(body)))


def approve(ticket_id, action):
    """Operator approval of a typed action (a dict such as {'kind': 'inspection_request', 'site': 'S3', 'xy': [x, y]})."""
    return _mac(_OPERATOR, ticket_id.encode("utf-8") + b"|" + _canon(action))


class AuthorityState:
    """Edge-side state: consumed nonces, revoked tickets, and per-client revocation times."""

    def __init__(self):
        self.consumed = set()
        self.revoked = set()
        self.client_cutoff = {}

    def revoke_ticket(self, ticket_id):
        self.revoked.add(ticket_id)

    def revoke_client(self, client, t):
        """All tickets of `client` issued at or before t are void (a credential reset)."""
        self.client_cutoff[client] = max(t, self.client_cutoff.get(client, float("-inf")))


def check(state, ticket, mission, now, spatial=False, action=None, approval=None, mutants=()):
    """Return (accepted, reason). `mutants` names deliberately broken safeguards, used only by the test of the test."""
    body = {k: ticket.get(k) for k in ("id", "client", "mission", "channel", "text", "iat", "exp")}
    if not isinstance(ticket.get("mac"), str) or not hmac.compare_digest(ticket["mac"], _mac(_PORTAL, _canon(body))):
        return False, "bad_tag"
    if "no_mission" not in mutants and ticket["mission"] != mission:
        return False, "wrong_mission"
    if ticket["channel"] != "portal":
        return False, "wrong_channel"
    if "no_expiry" not in mutants and not (ticket["iat"] - SKEW <= now < ticket["exp"]):
        return False, "outside_window"
    if "no_ttl_cap" not in mutants and ticket["exp"] - ticket["iat"] > TTL_MAX:
        return False, "lifetime_too_long"
    if "no_revocation" not in mutants:
        if ticket["id"] in state.revoked or ticket["iat"] <= state.client_cutoff.get(ticket["client"], float("-inf")):
            return False, "revoked"
    if "early_consume" in mutants:
        was = ticket["id"] in state.consumed
        state.consumed.add(ticket["id"])
        if was:
            return False, "replayed"
    elif "no_single_use" not in mutants and ticket["id"] in state.consumed:
        return False, "replayed"
    if spatial:
        if approval is None or action is None:
            return False, "no_approval"
        if "approval_unbound" in mutants:      # a well-formed token counts, whatever action it was issued for
            ok = isinstance(approval, str) and len(approval) == 32
        else:
            ok = hmac.compare_digest(approval, approve(ticket["id"], action))
        if not ok:
            return False, "approval_mismatch"
    state.consumed.add(ticket["id"])
    return True, "ok"
