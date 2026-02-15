import re
import pandas as pd

# Minimal example mappings – tune for your environment
OFFICE = {"winword.exe":"WORD","excel.exe":"EXCEL","powerpnt.exe":"PPT","outlook.exe":"OUTLOOK"}
SCRIPTS = {"powershell.exe":"POWERSHELL","pwsh.exe":"POWERSHELL","wscript.exe":"WSCRIPT","cscript.exe":"CSCRIPT"}
BROWSERS = {"chrome.exe":"CHROME","msedge.exe":"EDGE","firefox.exe":"FIREFOX"}

ENCODED_RE = re.compile(r"(?i)(-enc|-encodedcommand)\b")
DOWNLOAD_CRADLE_RE = re.compile(r"(?i)(iwr|invoke-webrequest|curl|wget)\b")
LOLBINS = {"regsvr32.exe","rundll32.exe","mshta.exe","wmic.exe","certutil.exe","bitsadmin.exe"}

def _basename(path: str) -> str:
    return (path or "").split("\\")[-1].lower()

def build_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """    Create multi-resolution behavior tokens and context flags.

    Produces
    --------
    - token_coarse: low-sparsity class (OFFICE/SCRIPT/BROWSER/LOLBIN/PROC)
    - token_medium: balanced token (e.g., OFFICE:WORD, SCRIPT:POWERSHELL)
    - token_fine: compact context token (adds parent + key cmdline flags + integrity + signed)

    Also produces context flags:
    - has_encoded, has_download_cradle, is_lolbin

    Why this matters
    ----------------
    Addresses critique #2 (granularity vs sparsity): transition matrices use token_medium (and
    optionally token_coarse as backoff), while token_fine feeds the *context channel* rather
    than exploding the state space.

    TODO
    ----
    - Add MITRE technique tagging rules (mitre_techniques).
    - Add severity scoring consistent with your analytic.
    - Add allowlist/suppression per role for common-but-weird tooling (e.g., java->powershell).
    """
    out = df.copy()
    img = out["image"].map(_basename)
    pimg = out["parent_image"].map(_basename)
    cmd = out["cmdline"].fillna("")

    coarse = []
    medium = []
    for i in img:
        if i in OFFICE:
            coarse.append("OFFICE"); medium.append(f"OFFICE:{OFFICE[i]}")
        elif i in SCRIPTS:
            coarse.append("SCRIPT"); medium.append(f"SCRIPT:{SCRIPTS[i]}")
        elif i in BROWSERS:
            coarse.append("BROWSER"); medium.append(f"BROWSER:{BROWSERS[i]}")
        elif i in LOLBINS:
            coarse.append("LOLBIN"); medium.append(f"LOLBIN:{i.upper()}")
        else:
            coarse.append("PROC"); medium.append(f"PROC:{i.upper() if i else 'UNKNOWN'}")

    out["token_coarse"] = coarse
    out["token_medium"] = medium

    out["has_encoded"] = cmd.str.contains(ENCODED_RE)
    out["has_download_cradle"] = cmd.str.contains(DOWNLOAD_CRADLE_RE)
    out["is_lolbin"] = img.isin({x.lower() for x in LOLBINS})

    il = out["integrity_level"].fillna("UNK").astype(str).str.upper()
    signed = out["signed"].fillna(False).astype(bool).astype(int)
    parent = pimg.replace("", "UNKNOWN").str.upper()

    out["token_fine"] = (
        out["token_medium"]
        + "|PARENT:" + parent
        + "|ENC:" + out["has_encoded"].astype(int).astype(str)
        + "|DL:" + out["has_download_cradle"].astype(int).astype(str)
        + "|IL:" + il
        + "|SIG:" + signed.astype(str)
    )

    # Placeholders: wire into your mapping logic
    if "mitre_techniques" not in out.columns:
        out["mitre_techniques"] = [[] for _ in range(len(out))]
    if "severity" not in out.columns:
        out["severity"] = 0.0

    return out
