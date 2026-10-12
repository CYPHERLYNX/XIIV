"""Claim-level deterministic verification for XIIV proof certificates.

Extends the guarantee-native certificate with factual claim checks:
each claim in the generated output is verified deterministically against
a provided fact store — no model in the loop, microseconds per claim.

This makes the proof certificate strictly stronger:
  structural compliance (contracts) + factual compliance (claims)

Inspired by ProveML (Deconinck, 2026). First integration of deterministic
claim verification into a diffusion LM's proof-carrying output pipeline.
"""

import re
import time


class ClaimCheck:
    """A single factual claim to verify."""

    def __init__(self, text_span: str, key: str, op: str = "eq",
                 expected: str | None = None):
        """
        Args:
            text_span: the claim text as it appears in the output
            key: lookup key in the fact store
            op: "eq" (exact match), "contains", "gt", "lt", "gte", "lte"
            expected: override the fact-store value (for testing)
        """
        self.text_span = text_span
        self.key = key
        self.op = op
        self.expected = expected

    def to_dict(self):
        return {"text_span": self.text_span, "key": self.key,
                "op": self.op, "expected": self.expected}


def verify_claims(text: str, claims: list[ClaimCheck],
                  fact_store: dict[str, str]) -> list[dict]:
    """Deterministically verify each claim against the fact store.

    No model calls. Pure string/numeric comparison. ~microseconds per claim.

    Returns list of {claim, key, expected, actual, passed, latency_us}.
    """
    results = []
    for claim in claims:
        t0 = time.perf_counter_ns()
        actual = fact_store.get(claim.key)
        expected = claim.expected if claim.expected is not None else actual

        # Extract the claimed value from the text span
        # (simple: the span itself is the claimed value)
        claimed = claim.text_span.strip()

        passed = False
        if actual is not None:
            if claim.op == "eq":
                passed = claimed == actual
            elif claim.op == "contains":
                passed = actual in claimed or claimed in actual
            elif claim.op in ("gt", "lt", "gte", "lte"):
                try:
                    c_num = float(re.sub(r"[^\d.\-]", "", claimed))
                    a_num = float(re.sub(r"[^\d.\-]", "", actual))
                    if claim.op == "gt":
                        passed = c_num > a_num
                    elif claim.op == "lt":
                        passed = c_num < a_num
                    elif claim.op == "gte":
                        passed = c_num >= a_num
                    elif claim.op == "lte":
                        passed = c_num <= a_num
                except ValueError:
                    passed = False

        latency_us = (time.perf_counter_ns() - t0) / 1000
        results.append({
            "claim": claim.text_span,
            "key": claim.key,
            "op": claim.op,
            "expected": expected,
            "actual": actual,
            "passed": passed,
            "latency_us": round(latency_us, 1),
        })
    return results


def extract_claims(text: str, pattern: str = r"\[\[claim:(.*?)\|(.*?)\]\]") -> list[ClaimCheck]:
    """Extract inline-marked claims from text.

    Format: [[claim:<key>|<op>]]<text span>[[/claim]]
    Example: The model has [[claim:params|eq]]916M[[/claim]] parameters.

    Returns list of ClaimCheck objects.
    """
    claims = []
    # Match [[claim:key|op]]span[[/claim]]
    full_pattern = r"\[\[claim:(.*?)\|(.*?)\]\](.*?)\[\[/claim\]\]"
    for m in re.finditer(full_pattern, text):
        key, op, span = m.group(1), m.group(2), m.group(3)
        claims.append(ClaimCheck(text_span=span, key=key, op=op))
    return claims


def certificate_with_claims(base_certificate: dict,
                            text: str,
                            fact_store: dict[str, str]) -> dict:
    """Extend a proof certificate with claim-level verification.

    Takes the existing certificate dict (with structural checks) and adds
    a "claim_checks" section with deterministic factual verification.
    """
    claims = extract_claims(text)
    claim_results = verify_claims(text, claims, fact_store)

    all_passed = all(r["passed"] for r in claim_results)

    cert = dict(base_certificate)  # don't mutate the original
    cert["claim_checks"] = claim_results
    cert["claim_summary"] = {
        "total": len(claim_results),
        "passed": sum(1 for r in claim_results if r["passed"]),
        "failed": sum(1 for r in claim_results if not r["passed"]),
        "all_passed": all_passed,
    }
    # The certificate is only fully valid if BOTH structural and
    # claim-level checks pass
    cert["fully_verified"] = (
        cert.get("structurally_verified", False) and all_passed
    )
    return cert
