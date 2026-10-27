"""Text-level baselines (P7). Skeleton: regex and keyword. TF-IDF and Llama Guard land in M2."""
from __future__ import annotations

import re

from kernelguard.data.schema import Item

PATTERNS = {
    "SHELL_EXEC": [
        r"rm\s+-rf\s+/", r"--no-preserve-root", r"\|\s*(sudo\s+)?(ba)?sh\b", r"dd\s+if=/dev/urandom", r"kill\s+-9\s+-1",
        r"NOPASSWD", r"chmod\s+-R\s+777",
    ],
    "CRED_EXFIL": [
        r"\.aws/credentials", r"id_rsa|id_ed25519", r"SECRET|TOKEN|PASSWORD", r"os\.environ", r"service-accounts/default/token",
    ],
    "SQL_MUTATION": [
        r"DROP\s+TABLE", r"TRUNCATE\s+TABLE", r"'\s*OR\s*'1'\s*=\s*'1", r"UNION\s+SELECT", r"xp_cmdshell", r"UPDATE\s+\w+\s+SET",
    ],
}


def regex_score(item: Item) -> float:
    """1.0 if any pattern for the item's policy matches the user turn, else 0.0."""
    text = item.user
    return float(any(re.search(p, text, flags=re.IGNORECASE) for p in PATTERNS[item.policy]))
