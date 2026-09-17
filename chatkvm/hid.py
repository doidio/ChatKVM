"""US-keyboard HID chords for AweSun `desktop_press_keys` / `desktop_typing_keys`.

OEM names and routing stay here; do not put them in model-visible tool docs.
"""

from __future__ import annotations

_ALIASES = {
    "esc": "ESCAPE",
    "escape": "ESCAPE",
    "backspace": "BACK",
    "back": "BACK",
    "return": "ENTER",
    "enter": "ENTER",
    "del": "DELETE",
    "spacebar": "SPACE",
    "space": "SPACE",
    "ctrl": "control",
}

_MODIFIERS = frozenset(
    {
        "shift",
        "control",
        "alt",
        "win",
        "command",
        "lcontrol",
        "rcontrol",
        "lalt",
        "ralt",
        "lwin",
        "rwin",
        "lcommand",
        "rcommand",
        "menu",
    }
)

_NAMED = frozenset(
    {
        "SHIFT",
        "CONTROL",
        "ALT",
        "LCONTROL",
        "RCONTROL",
        "LALT",
        "RALT",
        "MENU",
        "WIN",
        "LWIN",
        "RWIN",
        "COMMAND",
        "LCOMMAND",
        "RCOMMAND",
        "CAPSLOCK",
        "NUMLOCK",
        "SCROLL",
        "BACK",
        "TAB",
        "RETURN",
        "ENTER",
        "ESCAPE",
        "SPACE",
        "PAUSE",
        "PRIOR",
        "NEXT",
        "HOME",
        "END",
        "INSERT",
        "DELETE",
        "LEFT",
        "UP",
        "RIGHT",
        "DOWN",
        "MULTIPLY",
        "ADD",
        "SEPARATOR",
        "SUBTRACT",
        "DECIMAL",
        "DIVIDE",
        "OEM_1",
        "OEM_PLUS",
        "OEM_COMMA",
        "OEM_MINUS",
        "OEM_PERIOD",
        "OEM_2",
        "OEM_3",
        "OEM_4",
        "OEM_5",
        "OEM_6",
        "OEM_7",
        "OEM_8",
        "EQUAL",
        "TILDE",
        "SELECT",
        "PRINT",
        "EXECUTE",
        "SNAPSHOT",
        "HELP",
        "APPS",
        "SLEEP",
        "ENHANCED",
        "CAPITAL",
        *map(str, range(10)),
        *(f"F{n}" for n in range(1, 25)),
        *(f"NUMPAD{n}" for n in range(10)),
        *"ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    }
)

# character → HID chord. Caps Lock assumed off; uppercase is SHIFT.
_CHARS: dict[str, tuple[str, ...]] = {
    " ": ("SPACE",),
    "\t": ("TAB",),
    "\n": ("ENTER",),
    "\r": ("ENTER",),
    "/": ("OEM_2",),
    "?": ("SHIFT", "OEM_2"),
    "-": ("OEM_MINUS",),
    "_": ("SHIFT", "OEM_MINUS"),
    "=": ("OEM_PLUS",),
    "+": ("SHIFT", "OEM_PLUS"),
    ",": ("OEM_COMMA",),
    "<": ("SHIFT", "OEM_COMMA"),
    ".": ("OEM_PERIOD",),
    ">": ("SHIFT", "OEM_PERIOD"),
    ";": ("OEM_1",),
    ":": ("SHIFT", "OEM_1"),
    "'": ("OEM_7",),
    '"': ("SHIFT", "OEM_7"),
    "[": ("OEM_4",),
    "{": ("SHIFT", "OEM_4"),
    "]": ("OEM_6",),
    "}": ("SHIFT", "OEM_6"),
    "\\": ("OEM_5",),
    "|": ("SHIFT", "OEM_5"),
    "`": ("OEM_3",),
    "~": ("SHIFT", "OEM_3"),
    "!": ("SHIFT", "1"),
    "@": ("SHIFT", "2"),
    "#": ("SHIFT", "3"),
    "$": ("SHIFT", "4"),
    "%": ("SHIFT", "5"),
    "^": ("SHIFT", "6"),
    "&": ("SHIFT", "7"),
    "*": ("SHIFT", "8"),
    "(": ("SHIFT", "9"),
    ")": ("SHIFT", "0"),
}
_CHARS.update({digit: (digit,) for digit in "0123456789"})
_CHARS.update({ch: (ch.upper(),) for ch in "abcdefghijklmnopqrstuvwxyz"})
_CHARS.update({ch: ("SHIFT", ch) for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})


def _listed(items: list[str]) -> str:
    return ", ".join(repr(item) for item in dict.fromkeys(items))


def chords_for_text(text: str) -> list[list[str]]:
    chords: list[list[str]] = []
    unknown: list[str] = []
    for char in text:
        keys = _CHARS.get(char)
        if keys is None:
            unknown.append(char)
        else:
            chords.append(list(keys))
    if unknown:
        raise ValueError(
            f"unsupported characters: {_listed(unknown)}. "
            "Use ASCII letters, digits, space, and US-keyboard punctuation."
        )
    return chords


def _resolve_token(token: str) -> list[str] | None:
    raw = str(token).strip()
    if not raw:
        return None
    raw = _ALIASES.get(raw.lower(), raw)
    if raw.lower() in _MODIFIERS:
        return [raw.lower()]
    upper = raw.upper()
    if upper in _NAMED:
        return [upper]
    if len(raw) == 1 and raw in _CHARS:
        return list(_CHARS[raw])
    return None


def resolve_press(keys: list[str]) -> list[str]:
    resolved: list[str] = []
    unknown: list[str] = []
    for token in keys:
        mapped = _resolve_token(token)
        if mapped is None:
            unknown.append(str(token))
        else:
            resolved.extend(mapped)
    if unknown:
        raise ValueError(
            f"unsupported keys: {_listed(unknown)}. "
            "Use named keys such as ENTER, TAB, ESCAPE, or a single "
            "US-keyboard character."
        )
    return resolved


def is_shortcut(keys: list[str]) -> bool:
    if len(keys) < 2:
        return False
    mods = [key for key in keys if key.lower() in _MODIFIERS]
    rest = [key for key in keys if key.lower() not in _MODIFIERS]
    return len(mods) >= 1 and len(rest) == 1 and len(rest[0]) == 1 and rest[0].isalnum()
