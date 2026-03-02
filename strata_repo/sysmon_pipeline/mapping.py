"""
Token mapping and severity grading.
=====================================
Best of:
  - pipeline_updated: multi-resolution tokens (coarse/medium/fine), context flags, MITRE stubs
  - v12_modular:      severity grading (score_to_label, grade_events)

The multi-resolution design is the key architectural decision from the README:
  - token_coarse  -> low sparsity, used as backoff in transition modeling
  - token_medium  -> primary transition state space
  - token_fine    -> context channel only (NOT in transition matrix to avoid sparsity)
"""
from __future__ import annotations

import re
from typing import Dict
import pandas as pd


# ---------------------------------------------------------------------------
# Token mapping tables (extend these for your environment)
# ---------------------------------------------------------------------------

OFFICE_MAP: Dict[str, str] = {
    "winword.exe":   "WORD",
    "excel.exe":     "EXCEL",
    "powerpnt.exe":  "PPT",
    "outlook.exe":   "OUTLOOK",
    "onenote.exe":   "ONENOTE",
    "visio.exe":     "VISIO",
    "msaccess.exe":  "ACCESS",
}

SCRIPT_MAP: Dict[str, str] = {
    "powershell.exe":  "POWERSHELL",
    "pwsh.exe":        "POWERSHELL",
    "wscript.exe":     "WSCRIPT",
    "cscript.exe":     "CSCRIPT",
    "python.exe":      "PYTHON",
    "python3.exe":     "PYTHON",
    "cmd.exe":         "CMD",
}

BROWSER_MAP: Dict[str, str] = {
    "chrome.exe":   "CHROME",
    "msedge.exe":   "EDGE",
    "firefox.exe":  "FIREFOX",
    "iexplore.exe": "IE",
}

LOLBINS: set = {
    "regsvr32.exe", "rundll32.exe", "mshta.exe", "wmic.exe",
    "certutil.exe", "bitsadmin.exe", "msiexec.exe", "odbcconf.exe",
    "regasm.exe", "regsvcs.exe", "installutil.exe", "cmstp.exe",
    "xwizard.exe", "pcalua.exe", "syncappvpublishingserver.exe",
}

# Regex patterns for context flags
ENCODED_RE       = re.compile(r"(?i)(-enc\b|-encodedcommand\b|-e\s+[A-Za-z0-9+/]{20})")
DOWNLOAD_CRADLE_RE = re.compile(r"(?i)(iwr|invoke-webrequest|curl|wget|downloadstring|downloadfile)\b")
BYPASS_RE        = re.compile(r"(?i)(-bypass|-nop\b|-noprofile|-executionpolicy\s+bypass)")
REFLECTION_RE    = re.compile(r"(?i)(reflection\.assembly|loadwithpartialname|load\()")

# ---------------------------------------------------------------------------
# EVENT_SEVERITY
# ---------------------------------------------------------------------------
# Numeric severity scores normalized to [0,1] from original domain-expert
# integer scores (divide by 100). Tune these to match your environment and
# threat model. Higher = more interesting for threat hunting.
#
# Key anchors:
#   1.00 = Event 10 (ProcessAccess / LSASS) — almost always malicious
#   0.95 = Event 8 (CreateRemoteThread) and Event 4104 (PS script block)
#   0.20 = Application crashes — low signal, rarely adversary-driven
# ---------------------------------------------------------------------------
EVENT_SEVERITY: Dict[int, float] = {
    # ---- Sysmon High-Value Events ----
    1:   0.70,  # Sysmon Process Create
    3:   0.85,  # Sysmon Network Connection
    5:   0.40,  # Sysmon Process Terminate
    6:   0.70,  # Sysmon Driver Loaded
    7:   0.80,  # Sysmon Image Load (DLL)
    8:   0.95,  # Sysmon CreateRemoteThread (code injection)
    10:  1.00,  # Sysmon Process Access (e.g., LSASS access)
    11:  0.85,  # Sysmon File Create (payload drop)
    12:  0.75,  # Sysmon Registry Object Added/Deleted
    13:  0.75,  # Sysmon Registry Value Set
    15:  0.60,  # Sysmon FileCreateStreamHash
    17:  0.70,  # Sysmon Named Pipe Events
    22:  0.80,  # Sysmon DNS Query (C2 hunting)

    # ---- Windows Security Log ----
    4624: 0.70,  # Successful Logon
    4634: 0.40,  # Logoff
    4648: 0.85,  # Logon w/ Explicit Credentials
    4672: 0.90,  # Special Privileges Assigned
    4688: 0.80,  # Process Creation (Security)
    4768: 0.80,  # Kerberos TGT
    4769: 0.80,  # Kerberos Service Ticket
    4776: 0.70,  # NTLM Auth
    4798: 0.75,  # User Group Membership Enumeration

    # ---- PowerShell ----
    4103: 0.75,  # PowerShell Module Logging
    4104: 0.95,  # PowerShell Script Block Logging

    # ---- Persistence / Services ----
    7045: 0.90,  # Service Installed

    # ---- Crypto / AD ----
    5058: 0.85,  # Key File Operations
    5061: 0.80,  # Cryptographic Operation
    1109: 0.90,  # Critical Directory Services Event (on DCs)

    # ---- Low Contextual Events ----
    7031: 0.40,  # Service Crashed
    7036: 0.30,  # Service Start/Stop
    1014: 0.30,  # DNS Failure
    1000: 0.25,  # App Crash
    1001: 0.25,  # Bugcheck
    1003: 0.25,  # App Hang
    600:  0.20,  # OS Startup/Shutdown
}


def _basename(path) -> str:
    """Extract lowercase filename from a Windows path."""
    if path is None: return ""
    try:
        s = str(path)
        if s in ("nan", "<NA>", "None", ""): return ""
        return s.replace("/", "\\").split("\\")[-1].lower().strip()
    except:
        return ""


def build_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign multi-resolution behavior tokens and context flags to each event row.

    Produces:
      token_coarse  : OFFICE | SCRIPT | BROWSER | LOLBIN | PROC
      token_medium  : OFFICE:WORD | SCRIPT:POWERSHELL | LOLBIN:RUNDLL32 | PROC:CMD.EXE
      token_fine    : token_medium + parent context + cmdline flags + integrity + signed
                      (used in context channel only, NOT in transition state space)

    Context flags:
      has_encoded           : -enc / -encodedcommand detected
      has_download_cradle   : invoke-webrequest / curl / wget etc.
      has_bypass            : -bypass / -noprofile etc.
      has_reflection        : reflection.assembly / load()
      is_lolbin             : known LOLBin binary

    Severity:
      severity_score  : float [0,1]
      severity_label  : critical / high / medium / low
    """
    out = df.copy()

    img   = out["image"].map(_basename)
    pimg  = out["parent_image"].map(_basename)
    cmd   = out["cmdline"].fillna("").astype(str)
    il    = out["integrity_level"].fillna("UNK").astype(str).str.upper()
    signed = out["signed"].fillna(False).astype(bool)

    # --- Coarse + Medium tokens ---
    coarse_list, medium_list = [], []
    for i in img:
        if i in OFFICE_MAP:
            coarse_list.append("OFFICE")
            medium_list.append(f"OFFICE:{OFFICE_MAP[i]}")
        elif i in SCRIPT_MAP:
            coarse_list.append("SCRIPT")
            medium_list.append(f"SCRIPT:{SCRIPT_MAP[i]}")
        elif i in BROWSER_MAP:
            coarse_list.append("BROWSER")
            medium_list.append(f"BROWSER:{BROWSER_MAP[i]}")
        elif i in LOLBINS:
            coarse_list.append("LOLBIN")
            medium_list.append(f"LOLBIN:{i.upper().replace('.EXE','')}")
        else:
            name = i.upper() if i else "UNKNOWN"
            coarse_list.append("PROC")
            medium_list.append(f"PROC:{name}")

    out["token_coarse"] = coarse_list
    out["token_medium"] = medium_list

    # --- Context flags ---
    out["has_encoded"]         = cmd.str.contains(ENCODED_RE).astype(bool)
    out["has_download_cradle"] = cmd.str.contains(DOWNLOAD_CRADLE_RE).astype(bool)
    out["has_bypass"]          = cmd.str.contains(BYPASS_RE).astype(bool)
    out["has_reflection"]      = cmd.str.contains(REFLECTION_RE).astype(bool)
    out["is_lolbin"]           = img.isin(LOLBINS)

    # --- Fine token (context channel only) ---
    parent = pimg.replace("", "UNKNOWN").str.upper()
    enc_flag     = out["has_encoded"].astype(int).astype(str)
    dl_flag      = out["has_download_cradle"].astype(int).astype(str)
    bypass_flag  = out["has_bypass"].astype(int).astype(str)
    sig_flag     = signed.astype(int).astype(str)

    out["token_fine"] = (
        out["token_medium"]
        + "|PAR:" + parent
        + "|ENC:" + enc_flag
        + "|DL:"  + dl_flag
        + "|BP:"  + bypass_flag
        + "|IL:"  + il
        + "|SIG:" + sig_flag
    )

    # --- Severity grading (from v12_modular) ---
    out["severity_score"] = out["event_id"].apply(
        lambda x: EVENT_SEVERITY.get(int(x), 0.1) if pd.notna(x) else 0.1
    )
    out["severity_label"] = out["severity_score"].apply(_score_to_label)

    # MITRE stub (wire in your mapping)
    if "mitre_technique" not in out.columns:
        out["mitre_technique"] = None
    if "mitre_tactic" not in out.columns:
        out["mitre_tactic"] = None

    return out


def _score_to_label(score: float) -> str:
    if score >= 0.85: return "critical"
    if score >= 0.60: return "high"
    if score >= 0.35: return "medium"
    return "low"
