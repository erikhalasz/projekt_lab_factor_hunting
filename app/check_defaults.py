"""Consistency check for the factor explorer (issue #12).

Re-fits the default setup (notebook 04's primary model, static, daily, default windows) for every
target cached in factor_lens_local/models/ and compares fit statistics and coefficients.

    python app/check_defaults.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

if __name__ == "__main__":
    chk = E.check_against_cache(E.load_data())
    print(chk.to_string(index=False))
    bad = chk[~chk["ok"]]
    print(f"\n{len(chk) - len(bad)}/{len(chk)} targets reproduce the cached results "
          f"(max relative difference {chk['max_abs_diff'].max():.2e}).")
    sys.exit(1 if len(bad) else 0)
