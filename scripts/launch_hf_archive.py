#!/usr/bin/env python3
"""Launch the HF archive upload with a vault surrogate as HF_TOKEN.

The surrogate (hsurr:...) is not the real secret — authd/sentinel swaps it
for the real HF token on egress to huggingface.co. Safe to pass via env.
"""
import os
import subprocess
import sys

sys.path.insert(0, "/opt/muse/skills/custom-connector-setup/bin")
from dynamic_credentials import dynamic_credential_entry

entry = dynamic_credential_entry("custom.huggingface")
surrogate = entry["surrogate"]
assert surrogate.startswith("hsurr:"), "bad surrogate"

env = dict(os.environ)
env["HF_TOKEN"] = surrogate
# httpx (used by huggingface_hub) chokes on bracketed IPv6 entries in
# no_proxy ("Invalid port: ':1]'"). We only talk to huggingface.co, so
# a minimal no_proxy is fine.
env["no_proxy"] = "localhost,127.0.0.1"
env["NO_PROXY"] = "localhost,127.0.0.1"

cmd = [
    sys.executable,
    "/home/hatch/workspace/ai-tools/corsac/scripts/archive_to_hf.py",
    "--datasets", "fineweb_edu", "papers", "reasoning", "math",
] + sys.argv[1:]

print(f"[launcher] running: {' '.join(cmd[1:])}", flush=True)
proc = subprocess.run(cmd, env=env, cwd="/home/hatch/workspace/ai-tools/corsac")
sys.exit(proc.returncode)
