"""Proof certificates for Guarantee Bricks.

A certificate is the tamper-evident evidence that one brick's guarantee
held for one generation. It binds three things together with hashes:

  * the contract  (contract_hash) -- the certificate cannot be replayed
    against a different contract;
  * the output    (output_hash)   -- the certificate cannot be detached
    from the exact token sequence it certifies;
  * the chain     (prev_hash)     -- bricks stack; each certificate
    commits to the previous one, so a stack cannot be reordered or
    have a brick silently dropped.

Certificates are plain JSON dicts. The verifier needs no tokenizer and
no model: only token ids, the contract dict, and this module's hash
functions. That is what makes the verifier tiny and independently
auditable.
"""
import hashlib
import json

CERT_VERSION = 1
GENESIS_HASH = "0" * 64


def canonical(obj) -> bytes:
    """Deterministic JSON encoding for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def sha_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract_hash(contract: dict) -> str:
    """Hash of the canonical contract this certificate is bound to."""
    return sha_hex(canonical(contract))


def output_hash(token_ids) -> str:
    """Hash of the exact certified token sequence."""
    ids = [int(t) for t in token_ids]
    return sha_hex(canonical(ids))


def seal(cert: dict) -> dict:
    """Attach the certificate's self-hash (tamper-evidence).

    Any modification of any field invalidates cert_hash. The verifier
    recomputes it independently.
    """
    body = {k: v for k, v in cert.items() if k != "cert_hash"}
    cert = dict(body)
    cert["cert_hash"] = sha_hex(canonical(body))
    return cert


def check_seal(cert: dict) -> bool:
    """True iff the certificate's self-hash is intact."""
    if "cert_hash" not in cert:
        return False
    body = {k: v for k, v in cert.items() if k != "cert_hash"}
    return cert["cert_hash"] == sha_hex(canonical(body))
