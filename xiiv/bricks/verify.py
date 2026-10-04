"""Independent verifier for Guarantee Brick certificates.

This module is deliberately independent of the bricks' enforcement code:
it imports nothing from must_include (or any future brick). It re-derives
every claim from (certificate, output token ids, contract) using only the
hash primitives in certificate.py.

Verification is fail-closed: any check that cannot be positively
confirmed rejects the certificate. An unknown brick type, a broken seal,
a hash mismatch, or a positional check failure all reject.

The verifier needs no tokenizer and no model. It is small enough to
audit in full and cheap enough to run on-device: that is what makes it
a trust anchor instead of another thing to trust.
"""
from . import certificate as certmod

KNOWN_BRICKS = {"must-include"}


class VerificationResult:
    def __init__(self, ok: bool, reason: str):
        self.ok = ok
        self.reason = reason

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return f"VerificationResult(ok={self.ok}, reason={self.reason!r})"


def _reject(reason: str) -> VerificationResult:
    return VerificationResult(False, reason)


def verify_certificate(cert: dict, output_token_ids, contract: dict,
                       prev_hash=None) -> VerificationResult:
    """Verify one brick certificate against an output and a contract."""
    if not isinstance(cert, dict):
        return _reject("certificate is not a dict")
    if cert.get("brick") not in KNOWN_BRICKS:
        return _reject(f"unknown brick type: {cert.get('brick')!r}")
    if cert.get("version") != certmod.CERT_VERSION:
        return _reject(f"unsupported certificate version: "
                       f"{cert.get('version')!r}")

    # 1. Tamper-evidence: the seal must be intact before trusting any field.
    if not certmod.check_seal(cert):
        return _reject("certificate seal broken (fields tampered)")

    # 2. The certificate must be bound to THIS contract, not another one.
    if cert.get("contract_hash") != certmod.contract_hash(contract):
        return _reject("contract_hash mismatch: certificate was issued "
                       "for a different contract")

    # 3. The certificate must be bound to THIS exact output.
    output_ids = [int(t) for t in output_token_ids]
    if cert.get("output_hash") != certmod.output_hash(output_ids):
        return _reject("output_hash mismatch: output differs from the "
                       "certified sequence")
    if cert.get("output_length") != len(output_ids):
        return _reject("output_length mismatch")

    # 4. Chain integrity: the brick stack cannot be reordered or cut.
    expected_prev = prev_hash or certmod.GENESIS_HASH
    if cert.get("prev_hash") != expected_prev:
        return _reject("prev_hash mismatch: certificate chain broken")

    # 5. Positional proof: the phrase tokens really sit at the claimed
    #    positions in the certified output. Re-derived, not trusted.
    phrase_ids = cert.get("phrase_token_ids")
    positions = cert.get("positions")
    if (not isinstance(phrase_ids, list) or not isinstance(positions, list)
            or len(phrase_ids) != len(positions) or not phrase_ids):
        return _reject("malformed phrase/positions in certificate")
    for pos, tid in zip(positions, phrase_ids):
        if not (0 <= int(pos) < len(output_ids)):
            return _reject(f"certified position {pos} out of range")
        if output_ids[int(pos)] != int(tid):
            return _reject(
                f"phrase token {tid} not found at certified position {pos}")
    # Contiguity: a phrase is one block, not scattered tokens.
    if positions != list(range(positions[0], positions[0] + len(positions))):
        return _reject("certified positions are not contiguous")

    return VerificationResult(True, "must-include guarantee verified: "
                                    "phrase present at certified positions, "
                                    "certificate bound to contract+output")
