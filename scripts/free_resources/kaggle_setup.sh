#!/bin/bash
# Kaggle setup for XIIV — run once per machine
# Prereq (Fahad): kaggle.com account + phone verification + API token at ~/.kaggle/kaggle.json
set -e

echo "=== XIIV Kaggle Setup ==="

# 1. Install kaggle CLI
pip install -q kaggle 2>/dev/null || pip3 install -q kaggle
echo "[ok] kaggle CLI installed: $(kaggle --version 2>/dev/null || echo 'installed')"

# 2. Check API token
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "[ACTION NEEDED] No ~/.kaggle/kaggle.json found."
    echo "  1. Go to https://www.kaggle.com/account"
    echo "  2. Click 'Create New API Token' (downloads kaggle.json)"
    echo "  3. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    echo "  5. Phone-verify at kaggle.com/settings (required for GPU + internet)"
    exit 1
fi
chmod 600 ~/.kaggle/kaggle.json
echo "[ok] API token found at ~/.kaggle/kaggle.json"

# 3. Verify auth works
if kaggle kernels list --mine --page-size 1 >/dev/null 2>&1; then
    echo "[ok] Kaggle API auth working"
else
    echo "[WARN] API call failed — check token and network"
    exit 1
fi

echo ""
echo "=== Setup complete. Next: ==="
echo "  cd scripts/free_resources"
echo "  python3 kaggle_sgbd_block.py --block 0 --dry-run   # preview"
echo "  python3 kaggle_sgbd_block.py --block 0             # launch"
