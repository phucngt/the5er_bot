# config.py

import os
import platform
import MetaTrader5 as mt5

# ── YOUR MT5 & ACCOUNT SETTINGS ─────────────────────────────────────
TERMINAL_PATH = r"C:\Program Files\MetaTrader 5 EXNESS\terminal64.exe"
ACCOUNT       = 183342154
PASSWORD      = "Ars!@#241201"
SERVER        = "Exness-MT5Real25"   # must match exactly your MT5 login dialog
# ────────────────────────────────────────────────────────────────────

# 1) DEBUG: confirm the path & Python bitness
print(" MT5 terminal exists:", os.path.isfile(TERMINAL_PATH))
print(" Python interpreter:", platform.architecture()[0])

# 2) Initialize AND login in one shot
#    This bypasses separate mt5.login() calls and avoids the "authorization failed" on init.
ok = mt5.initialize(
    path     = TERMINAL_PATH,
    login    = ACCOUNT,
    password = PASSWORD,
    server   = SERVER
)

if not ok:
    code, msg = mt5.last_error()
    raise RuntimeError(f"mt5.initialize(login…) failed: ({code}) {msg!r}")

# 3) Success: show account info
info = mt5.account_info()
print(f" ✅ MT5 connected: Login={info.login}, Balance={info.balance}, FreeMargin={info.margin_free}")
