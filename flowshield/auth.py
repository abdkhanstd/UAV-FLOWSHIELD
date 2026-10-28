"""Origin authentication of client-portal documents.

The client portal attaches a tag to a ticket after the submitting user has authenticated:
    tag = HMAC-SHA256(client key, mission id | channel | ticket body)
The tag travels beside the text (a transport field, not part of the content), so an author of the text cannot produce
it. The trusted edge recomputes the tag; only a verified document may add a spatial task. This authenticates where a
request came from, not whether the request is wise: a stolen client credential defeats it, which is outside the
threat model (Section 3).

The key material below is a simulation constant. The attacker models of the benchmark never read this module."""
import hashlib
import hmac

_MASTER = b"flowshield-simulation-master-key"


def client_key(client):
    return hmac.new(_MASTER, client.encode("utf-8"), hashlib.sha256).digest()


def _msg(ep_id, channel, text):
    return f"{ep_id}|{channel}|{text}".encode("utf-8")


def sign(client, ep_id, channel, text):
    return hmac.new(client_key(client), _msg(ep_id, channel, text), hashlib.sha256).hexdigest()[:32]


def verify(ep, doc):
    tag = doc.get("sig")
    if not isinstance(tag, str) or not tag:
        return False
    good = sign(ep["operator"]["client"], ep["id"], doc["channel"], doc["text"])
    return hmac.compare_digest(tag, good)
