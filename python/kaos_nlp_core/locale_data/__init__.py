# ruff: noqa: RUF001
"""Locale data for rule-based extraction.

Frozen gazetteers consumed by the alpha extractors in
:mod:`kaos_nlp_core.extract.alpha` (WS-TR.PR-6f; moved from
``kaos_llm_core.extract.alpha`` in 2026-05). These are pure data
dicts — no Rust binding, no regex compilation, no runtime mutation.
Every public attribute is a nested ``dict[language_code, dict[token,
value]]`` with values typed per gazetteer (``int`` for months / weekdays
/ ordinals, ``str`` for ISO currency codes, ``int | str`` for
written-number multipliers, ``float`` for percent scale factors).

Ported verbatim from
``kelvin-modules/references/kelvin-nlp/kelvin/nlp/extract/alpha/locale_data.py``
(~440 lines of pure data) to unblock the alpha-extractor sprint. The
data is hand-curated; no external vendor attribution is required beyond
the existing kelvin-modules → kaos-modules provenance.

These gazetteers are intentionally NOT in Rust. They are small enough
that Python dict lookup is free, and keeping them in Python lets
downstream consumers (custom recipes, ad-hoc analyses) extend them
without a Rust rebuild.

Usage::

    from kaos_nlp_core.locale_data import MONTH_MAP, MONEY_MAP

    month_number = MONTH_MAP["en"].get("january")  # 1
    iso_code = MONEY_MAP["en"].get("€")            # "EUR"

Extension pattern — callers can overlay their own gazetteer by
deep-merging; do NOT mutate the module-level dicts::

    import copy
    extended = copy.deepcopy(MONTH_MAP)
    extended["en"]["midsummer"] = 6
    # Pass `extended` into your extractor; leave MONTH_MAP alone.

.. note::
    Some ambiguous abbreviations are documented with inline comments —
    e.g., Spanish ``"mar"`` is both "marzo" (March) and "martes"
    (Tuesday). Extractors are expected to disambiguate via context.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Written numbers — for AlphaNumberExtractor (written-form parsing)
# ----------------------------------------------------------------------

WRITTEN_NUMBER_MAP: dict[str, dict[str, int | str]] = {
    "en": {
        "point": ".",
        "spot": ".",
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
        "hundred": 100,
        "hundreds": 100,
        "thousand": 10**3,
        "thousands": 10**3,
        "ten-thousand": 10**4,
        "hundred-thousand": 10**5,
        "million": 10**6,
        "millions": 10**6,
        "billion": 10**9,
        "billions": 10**9,
        "trillion": 10**12,
        "quadrillion": 10**15,
        "quintillion": 10**18,
    },
    "de": {
        "punkt": ".",
        "stelle": ".",
        "null": 0,
        "eins": 1,
    },
}

# ----------------------------------------------------------------------
# Ordinals — for date parsing ("first", "1st", "twenty-third", "23rd")
# ----------------------------------------------------------------------

ORDINAL_MAP: dict[str, dict[str, int]] = {
    "en": {
        "zeroth": 0,
        "first": 1,
        "1st": 1,
        "second": 2,
        "2nd": 2,
        "third": 3,
        "3rd": 3,
        "fourth": 4,
        "4th": 4,
        "fifth": 5,
        "5th": 5,
        "sixth": 6,
        "6th": 6,
        "seventh": 7,
        "7th": 7,
        "eighth": 8,
        "8th": 8,
        "ninth": 9,
        "9th": 9,
        "tenth": 10,
        "10th": 10,
        "eleventh": 11,
        "11th": 11,
        "twelfth": 12,
        "12th": 12,
        "thirteenth": 13,
        "13th": 13,
        "fourteenth": 14,
        "14th": 14,
        "fifteenth": 15,
        "15th": 15,
        "sixteenth": 16,
        "16th": 16,
        "seventeenth": 17,
        "17th": 17,
        "eighteenth": 18,
        "18th": 18,
        "nineteenth": 19,
        "19th": 19,
        "twentieth": 20,
        "20th": 20,
        "twenty-first": 21,
        "21st": 21,
        "twenty-second": 22,
        "22nd": 22,
        "twenty-third": 23,
        "23rd": 23,
        "twenty-fourth": 24,
        "24th": 24,
        "twenty-fifth": 25,
        "25th": 25,
        "twenty-sixth": 26,
        "26th": 26,
        "twenty-seventh": 27,
        "27th": 27,
        "twenty-eighth": 28,
        "28th": 28,
        "twenty-ninth": 29,
        "29th": 29,
        "thirtieth": 30,
        "30th": 30,
        "thirty-first": 31,
        "31st": 31,
    }
}

# ----------------------------------------------------------------------
# Durations — for AlphaDurationExtractor (keyword → seconds multiplier)
# ----------------------------------------------------------------------
#
# Months are approximated as 30 days (2,592,000 seconds); years as 365
# days (31,536,000 seconds). Consumers that need calendar-accurate
# durations should use ``dateutil.relativedelta``.

DURATION_MAP: dict[str, dict[str, int]] = {
    "en": {
        "second": 1,
        "seconds": 1,
        "minute": 60,
        "minutes": 60,
        "hour": 3600,
        "hours": 3600,
        "day": 86400,
        "days": 86400,
        "week": 604800,
        "weeks": 604800,
        "month": 2592000,
        "months": 2592000,
        "year": 31536000,
        "years": 31536000,
        "anniversary": 31536000,
    },
    "es": {
        "segundo": 1,
        "segundos": 1,
        "minuto": 60,
        "minutos": 60,
        "hora": 3600,
        "horas": 3600,
        "día": 86400,
        "días": 86400,
        "semana": 604800,
        "semanas": 604800,
        "mes": 2592000,
        "meses": 2592000,
        "año": 31536000,
        "años": 31536000,
        "aniversario": 31536000,
    },
    "fr": {
        "seconde": 1,
        "secondes": 1,
        "minute": 60,
        "minutes": 60,
        "heure": 3600,
        "heures": 3600,
        "jour": 86400,
        "jours": 86400,
        "semaine": 604800,
        "semaines": 604800,
        "mois": 2592000,
        "ans": 31536000,  # or "années"
        "anniversaire": 31536000,
    },
    "de": {
        "sekunde": 1,
        "sekunden": 1,
        "minute": 60,
        "minuten": 60,
        "stunde": 3600,
        "stunden": 3600,
        "tag": 86400,
        "tage": 86400,
        "woche": 604800,
        "wochen": 604800,
        "monat": 2592000,
        "monate": 2592000,
        "jahr": 31536000,
        "jahre": 31536000,
        "jubiläum": 31536000,  # or "Jahrestag"
    },
}

# ----------------------------------------------------------------------
# Money — for AlphaMoneyExtractor (currency word / symbol → ISO 4217)
# ----------------------------------------------------------------------
#
# Covers USD, GBP, EUR, JPY, CNY, INR, KRW, CHF. Symbols and words
# (case-insensitive variants) all map to the same ISO code.

MONEY_MAP: dict[str, dict[str, str]] = {
    "en": {
        "dollar": "USD",
        "Dollar": "USD",
        "dollars": "USD",
        "Dollars": "USD",
        "DOLLAR": "USD",
        "DOLLARS": "USD",
        "USD": "USD",
        "US$": "USD",
        "$": "USD",
        "pound": "GBP",
        "pounds": "GBP",
        "Pound": "GBP",
        "Pounds": "GBP",
        "GBP": "GBP",
        "£": "GBP",
        "euro": "EUR",
        "Euro": "EUR",
        "euros": "EUR",
        "Euros": "EUR",
        "EUR": "EUR",
        "€": "EUR",
        "yen": "JPY",
        "Yen": "JPY",
        "JPY": "JPY",
        "¥": "JPY",
        "yuan": "CNY",
        "Yuan": "CNY",
        "CNY": "CNY",
        "rupee": "INR",
        "Rupee": "INR",
        "INR": "INR",
        "₹": "INR",
        "won": "KRW",
        "Won": "KRW",
        "KRW": "KRW",
        "₩": "KRW",
        "franc": "CHF",
        "Franc": "CHF",
        "CHF": "CHF",
        "₣": "CHF",
    }
}

# ----------------------------------------------------------------------
# Months — for AlphaDateExtractor (month name → 1..12)
# ----------------------------------------------------------------------

MONTH_MAP: dict[str, dict[str, int]] = {
    "en": {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    },
    "es": {
        "enero": 1,
        "ene": 1,
        "febrero": 2,
        "feb": 2,
        "marzo": 3,
        "mar": 3,  # conflicts with "mar" for Tuesday (WEEKDAY_MAP["es"])
        "abril": 4,
        "abr": 4,
        "mayo": 5,
        "may": 5,
        "junio": 6,
        "jun": 6,
        "julio": 7,
        "jul": 7,
        "agosto": 8,
        "ago": 8,
        "septiembre": 9,
        "sept": 9,
        "octubre": 10,
        "oct": 10,
        "noviembre": 11,
        "nov": 11,
        "diciembre": 12,
        "dic": 12,
    },
    "fr": {
        "janvier": 1,
        "janv": 1,
        "février": 2,
        "févr": 2,
        "mars": 3,
        "avril": 4,
        "avr": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "juil": 7,
        "août": 8,
        "septembre": 9,
        "sept": 9,
        "octobre": 10,
        "oct": 10,
        "novembre": 11,
        "nov": 11,
        "décembre": 12,
        "déc": 12,
    },
    "de": {
        "januar": 1,
        "jän": 1,
        "februar": 2,
        "feb": 2,
        "märz": 3,
        "april": 4,
        "apr": 4,
        "mai": 5,
        "juni": 6,
        "jun": 6,
        "juli": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sept": 9,
        "oktober": 10,
        "okt": 10,
        "november": 11,
        "nov": 11,
        "dezember": 12,
        "dez": 12,
    },
}

# ----------------------------------------------------------------------
# Weekdays — for AlphaDateExtractor (weekday name → 1..7, Mon=1)
# ----------------------------------------------------------------------

WEEKDAY_MAP: dict[str, dict[str, int]] = {
    "en": {
        "monday": 1,
        "mon": 1,
        "tuesday": 2,
        "tue": 2,
        "wednesday": 3,
        "wed": 3,
        "thursday": 4,
        "thu": 4,
        "friday": 5,
        "fri": 5,
        "saturday": 6,
        "sat": 6,
        "sunday": 7,
        "sun": 7,
    },
    "es": {
        "lunes": 1,
        "lun": 1,
        "martes": 2,
        "mar": 2,  # conflicts with MONTH_MAP["es"]["mar"] = 3 (marzo)
        "miércoles": 3,
        "mié": 3,
        "jueves": 4,
        "jue": 4,
        "viernes": 5,
        "vie": 5,
        "sábado": 6,
        "sáb": 6,
        "domingo": 7,
        "dom": 7,
    },
    "fr": {
        "lundi": 1,
        "lun": 1,
        "mardi": 2,
        "mar": 2,  # conflicts with MONTH_MAP["fr"]["mar"] (mars absent — OK)
        "mercredi": 3,
        "mer": 3,
        "jeudi": 4,
        "jeu": 4,
        "vendredi": 5,
        "ven": 5,
        "samedi": 6,
        "sam": 6,
        "dimanche": 7,
        "dim": 7,
    },
    "de": {
        "montag": 1,
        "mo": 1,
        "dienstag": 2,
        "di": 2,
        "mittwoch": 3,
        "mi": 3,
        "donnerstag": 4,
        "do": 4,
        "freitag": 5,
        "fr": 5,
        "samstag": 6,
        "sa": 6,
        "sonntag": 7,
        "so": 7,
    },
}

# ----------------------------------------------------------------------
# Percent — for AlphaPercentExtractor (keyword → scale factor)
# ----------------------------------------------------------------------
#
# 50% → multiply by 0.01 to get 0.5 (fractional).
# 50 bps → multiply by 0.01/100.0 to get 0.005.
# Fullwidth/roman-prefix variants included; basis-point handling is
# context-aware in the extractor (e.g., "100 basis points of LIBOR").

PERCENT_MAP: dict[str, dict[str, float]] = {
    "en": {
        "percent": 0.01,
        "percentage": 0.01,
        "percents": 0.01,
        "percentages": 0.01,
        "pct.": 0.01,
        "hundredth": 0.01,
        "hundredths": 0.01,
        "%": 0.01,
        "﹪": 0.01,
        "％": 0.01,
        "‰": 0.001,
        "‱": 0.0001,
        "ppm": 0.000001,
        "ppb": 0.000000001,
        "bps": 0.01 / 100.0,
        "basis": 0.01 / 100.0,
    }
}


# ----------------------------------------------------------------------
# Units — for AlphaQuantityExtractor (NUMBER_WITH_UNITS slot)
# ----------------------------------------------------------------------
#
# Each entry maps the surface form of a unit to a (canonical_form,
# dimension) tuple. Canonical form follows SI conventions where
# applicable; dimension is one of the strings in :data:`UNIT_DIMENSIONS`
# below (mass, length, area, volume, time, temperature, pressure,
# energy, power, force, frequency, speed, angle, data, density,
# concentration, currency-rate, count).
#
# Coverage: SI (mass, length, area, volume, time, temperature, pressure,
# energy, power, force, frequency) plus US/imperial (mass, length,
# area, volume, speed). Plurals are listed explicitly so we don't have
# to do morphological matching.
#
# Composite units (e.g., "kg/m^3", "N·m") are out of scope — they need a
# real units library (Pint). This gazetteer covers single-token unit
# forms that appear in contracts, science writing, and shipping/spec
# documents.

UNIT_DIMENSIONS = (
    "mass",
    "length",
    "area",
    "volume",
    "time",
    "temperature",
    "pressure",
    "energy",
    "power",
    "force",
    "frequency",
    "speed",
    "angle",
    "data",
    "count",
)

UNIT_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "en": {
        # ── Mass ────────────────────────────────────────────────────
        # Single-letter "g" was removed — collides with two-letter
        # postal codes ("7327 GM") and other prose. Use "gram" /
        # "grams" instead. Same rationale for "t" (tonne), "n" (newton),
        # "j" (joule), "k" (kelvin), "l" (liter), "s" (second), "w"
        # (watt), "b" (byte), "in" (inch) — all removed below.
        "mg": ("mg", "mass"),
        "milligram": ("mg", "mass"),
        "milligrams": ("mg", "mass"),
        "gram": ("g", "mass"),
        "grams": ("g", "mass"),
        "gm": ("g", "mass"),
        "kg": ("kg", "mass"),
        "kilogram": ("kg", "mass"),
        "kilograms": ("kg", "mass"),
        "kgs": ("kg", "mass"),
        "tonne": ("t", "mass"),
        "tonnes": ("t", "mass"),
        "metric-ton": ("t", "mass"),
        "metric-tons": ("t", "mass"),
        "oz": ("oz", "mass"),
        "ounce": ("oz", "mass"),
        "ounces": ("oz", "mass"),
        "lb": ("lb", "mass"),
        "lbs": ("lb", "mass"),
        "pound": ("lb", "mass"),
        "pounds": ("lb", "mass"),
        "ton": ("ton", "mass"),
        "tons": ("ton", "mass"),
        # ── Length ──────────────────────────────────────────────────
        "mm": ("mm", "length"),
        "millimeter": ("mm", "length"),
        "millimeters": ("mm", "length"),
        "millimetre": ("mm", "length"),
        "millimetres": ("mm", "length"),
        "cm": ("cm", "length"),
        "centimeter": ("cm", "length"),
        "centimeters": ("cm", "length"),
        "centimetre": ("cm", "length"),
        "centimetres": ("cm", "length"),
        "m": ("m", "length"),
        "meter": ("m", "length"),
        "meters": ("m", "length"),
        "metre": ("m", "length"),
        "metres": ("m", "length"),
        "km": ("km", "length"),
        "kilometer": ("km", "length"),
        "kilometers": ("km", "length"),
        "kilometre": ("km", "length"),
        "kilometres": ("km", "length"),
        "inch": ("in", "length"),
        "inches": ("in", "length"),
        "ft": ("ft", "length"),
        "foot": ("ft", "length"),
        "feet": ("ft", "length"),
        "yd": ("yd", "length"),
        "yard": ("yd", "length"),
        "yards": ("yd", "length"),
        "mi": ("mi", "length"),
        "mile": ("mi", "length"),
        "miles": ("mi", "length"),
        "nm": ("nm", "length"),
        "nautical-mile": ("nmi", "length"),
        "nautical-miles": ("nmi", "length"),
        # ── Area ────────────────────────────────────────────────────
        "sqm": ("m^2", "area"),
        "sq-m": ("m^2", "area"),
        "sqft": ("ft^2", "area"),
        "sq-ft": ("ft^2", "area"),
        "sqin": ("in^2", "area"),
        "sq-in": ("in^2", "area"),
        "sqmi": ("mi^2", "area"),
        "sq-mi": ("mi^2", "area"),
        "ha": ("ha", "area"),
        "hectare": ("ha", "area"),
        "hectares": ("ha", "area"),
        "acre": ("acre", "area"),
        "acres": ("acre", "area"),
        # ── Volume ──────────────────────────────────────────────────
        "ml": ("mL", "volume"),
        "milliliter": ("mL", "volume"),
        "milliliters": ("mL", "volume"),
        "millilitre": ("mL", "volume"),
        "millilitres": ("mL", "volume"),
        "liter": ("L", "volume"),
        "liters": ("L", "volume"),
        "litre": ("L", "volume"),
        "litres": ("L", "volume"),
        "fl-oz": ("fl_oz", "volume"),
        "fluid-ounce": ("fl_oz", "volume"),
        "fluid-ounces": ("fl_oz", "volume"),
        "pint": ("pt", "volume"),
        "pints": ("pt", "volume"),
        "quart": ("qt", "volume"),
        "quarts": ("qt", "volume"),
        "gal": ("gal", "volume"),
        "gallon": ("gal", "volume"),
        "gallons": ("gal", "volume"),
        "barrel": ("bbl", "volume"),
        "barrels": ("bbl", "volume"),
        "bbl": ("bbl", "volume"),
        # ── Time (extractor-specific) ───────────────────────────────
        # Time-as-quantity is rare in practice (durations are caught by
        # AlphaDurationExtractor). Including only the SI base unit so
        # quantity covers "5 s" / "100 ms" expressions in scientific
        # text without competing with duration's coarser language.
        "ns": ("ns", "time"),
        "nanosecond": ("ns", "time"),
        "nanoseconds": ("ns", "time"),
        "us": ("us", "time"),
        "microsecond": ("us", "time"),
        "microseconds": ("us", "time"),
        "ms": ("ms", "time"),
        "millisecond": ("ms", "time"),
        "milliseconds": ("ms", "time"),
        # "sec" / "secs" removed — collide with the SEC acronym
        # ("Section 3.2 SEC Approval" → 3.2 seconds is a real false
        # positive on EDGAR documents). Use "second" / "seconds" or
        # leave time-as-quantity to AlphaDurationExtractor.
        # ── Temperature ─────────────────────────────────────────────
        "kelvin": ("K", "temperature"),
        # Celsius and Fahrenheit are usually written with the degree
        # symbol fused onto the number ("20°C") — the extractor handles
        # the fused form via a separate suffix branch.
        # ── Pressure ────────────────────────────────────────────────
        "pa": ("Pa", "pressure"),
        "pascal": ("Pa", "pressure"),
        "pascals": ("Pa", "pressure"),
        "kpa": ("kPa", "pressure"),
        "kilopascal": ("kPa", "pressure"),
        "kilopascals": ("kPa", "pressure"),
        "mpa": ("MPa", "pressure"),
        "megapascal": ("MPa", "pressure"),
        "megapascals": ("MPa", "pressure"),
        "bar": ("bar", "pressure"),
        "bars": ("bar", "pressure"),
        "atm": ("atm", "pressure"),
        "atmosphere": ("atm", "pressure"),
        "atmospheres": ("atm", "pressure"),
        "psi": ("psi", "pressure"),
        # ── Energy ──────────────────────────────────────────────────
        "joule": ("J", "energy"),
        "joules": ("J", "energy"),
        "kj": ("kJ", "energy"),
        "kilojoule": ("kJ", "energy"),
        "kilojoules": ("kJ", "energy"),
        "cal": ("cal", "energy"),
        "calorie": ("cal", "energy"),
        "calories": ("cal", "energy"),
        "kcal": ("kcal", "energy"),
        "kilocalorie": ("kcal", "energy"),
        "kilocalories": ("kcal", "energy"),
        "wh": ("Wh", "energy"),
        "kwh": ("kWh", "energy"),
        "mwh": ("MWh", "energy"),
        "btu": ("BTU", "energy"),
        # ── Power ───────────────────────────────────────────────────
        "watt": ("W", "power"),
        "watts": ("W", "power"),
        "kw": ("kW", "power"),
        "kilowatt": ("kW", "power"),
        "kilowatts": ("kW", "power"),
        "mw": ("MW", "power"),
        "megawatt": ("MW", "power"),
        "megawatts": ("MW", "power"),
        "hp": ("hp", "power"),
        "horsepower": ("hp", "power"),
        # ── Force ───────────────────────────────────────────────────
        "newton": ("N", "force"),
        "newtons": ("N", "force"),
        "kn": ("kN", "force"),
        # ── Frequency ───────────────────────────────────────────────
        "hz": ("Hz", "frequency"),
        "hertz": ("Hz", "frequency"),
        "khz": ("kHz", "frequency"),
        "mhz": ("MHz", "frequency"),
        "ghz": ("GHz", "frequency"),
        "thz": ("THz", "frequency"),
        # ── Speed ───────────────────────────────────────────────────
        "mph": ("mph", "speed"),
        "kph": ("km/h", "speed"),
        "kmh": ("km/h", "speed"),
        "knot": ("kn", "speed"),
        "knots": ("kn", "speed"),
        # ── Angle ───────────────────────────────────────────────────
        "rad": ("rad", "angle"),
        "radian": ("rad", "angle"),
        "radians": ("rad", "angle"),
        # Degrees with degree-symbol live in the fused-suffix branch.
        # ── Data ────────────────────────────────────────────────────
        "byte": ("B", "data"),
        "bytes": ("B", "data"),
        "kb": ("KB", "data"),
        "kilobyte": ("KB", "data"),
        "kilobytes": ("KB", "data"),
        "kib": ("KiB", "data"),
        "mb": ("MB", "data"),
        "megabyte": ("MB", "data"),
        "megabytes": ("MB", "data"),
        "mib": ("MiB", "data"),
        "gb": ("GB", "data"),
        "gigabyte": ("GB", "data"),
        "gigabytes": ("GB", "data"),
        "gib": ("GiB", "data"),
        "tb": ("TB", "data"),
        "terabyte": ("TB", "data"),
        "terabytes": ("TB", "data"),
        "tib": ("TiB", "data"),
        "pb": ("PB", "data"),
        "petabyte": ("PB", "data"),
        "petabytes": ("PB", "data"),
        # ── Count (dimensionless quantity-with-noun forms) ──────────
        # Counted things that don't fit a physical dimension. Tuned for
        # contract domains: financial (shares, lots, contracts), real
        # estate (units, lots, rooms, bedrooms, bathrooms, parking
        # spaces), supply chain (containers, pallets, cases, cartons,
        # drums, skids, boxes), construction (panels, posts).
        "unit": ("unit", "count"),
        "units": ("unit", "count"),
        "share": ("share", "count"),
        "shares": ("share", "count"),
        "piece": ("piece", "count"),
        "pieces": ("piece", "count"),
        "item": ("item", "count"),
        "items": ("item", "count"),
        # Real-estate counts.
        "lot": ("lot", "count"),
        "lots": ("lot", "count"),
        "room": ("room", "count"),
        "rooms": ("room", "count"),
        "bedroom": ("bedroom", "count"),
        "bedrooms": ("bedroom", "count"),
        "bathroom": ("bathroom", "count"),
        "bathrooms": ("bathroom", "count"),
        "story": ("story", "count"),
        "stories": ("story", "count"),
        "floor": ("floor", "count"),
        "floors": ("floor", "count"),
        # Supply-chain counts.
        "container": ("container", "count"),
        "containers": ("container", "count"),
        "teu": ("TEU", "count"),
        "feu": ("FEU", "count"),
        "pallet": ("pallet", "count"),
        "pallets": ("pallet", "count"),
        "case": ("case", "count"),
        "cases": ("case", "count"),
        "carton": ("carton", "count"),
        "cartons": ("carton", "count"),
        "drum": ("drum", "count"),
        "drums": ("drum", "count"),
        "skid": ("skid", "count"),
        "skids": ("skid", "count"),
        "box": ("box", "count"),
        "boxes": ("box", "count"),
        "crate": ("crate", "count"),
        "crates": ("crate", "count"),
        # Financial-instrument counts.
        "contract": ("contract", "count"),
        "contracts": ("contract", "count"),
        "warrant": ("warrant", "count"),
        "warrants": ("warrant", "count"),
        "option": ("option", "count"),
        "options": ("option", "count"),
        "bond": ("bond", "count"),
        "bonds": ("bond", "count"),
        "note": ("note", "count"),
        "notes": ("note", "count"),
        # Construction additions (non-multi-token).
        "ksi": ("ksi", "pressure"),
        # Oil & gas energy / volume.
        "mmbtu": ("MMBtu", "energy"),
        "mmcf": ("MMcf", "volume"),
        "bcf": ("BCF", "volume"),
        "tcf": ("TCF", "volume"),
        # ── Multi-token units (space-joined keys; extractor handles) ──
        # Real estate
        "square foot": ("ft^2", "area"),
        "square feet": ("ft^2", "area"),
        "square yard": ("yd^2", "area"),
        "square yards": ("yd^2", "area"),
        "square meter": ("m^2", "area"),
        "square meters": ("m^2", "area"),
        "square metre": ("m^2", "area"),
        "square metres": ("m^2", "area"),
        "square mile": ("mi^2", "area"),
        "square miles": ("mi^2", "area"),
        "square kilometer": ("km^2", "area"),
        "square kilometers": ("km^2", "area"),
        "parking space": ("parking-space", "count"),
        "parking spaces": ("parking-space", "count"),
        # Construction
        "cubic foot": ("ft^3", "volume"),
        "cubic feet": ("ft^3", "volume"),
        "cubic yard": ("yd^3", "volume"),
        "cubic yards": ("yd^3", "volume"),
        "cubic meter": ("m^3", "volume"),
        "cubic meters": ("m^3", "volume"),
        "cubic metre": ("m^3", "volume"),
        "cubic metres": ("m^3", "volume"),
        "linear foot": ("lin-ft", "length"),
        "linear feet": ("lin-ft", "length"),
        "linear meter": ("lin-m", "length"),
        "linear meters": ("lin-m", "length"),
        "board foot": ("bf", "volume"),
        "board feet": ("bf", "volume"),
        # Volume + descriptors
        "fluid ounce": ("fl_oz", "volume"),
        "fluid ounces": ("fl_oz", "volume"),
        "metric ton": ("t", "mass"),
        "metric tons": ("t", "mass"),
        "metric tonne": ("t", "mass"),
        "metric tonnes": ("t", "mass"),
        "short ton": ("short-ton", "mass"),
        "short tons": ("short-ton", "mass"),
        "long ton": ("long-ton", "mass"),
        "long tons": ("long-ton", "mass"),
        "nautical mile": ("nmi", "length"),
        "nautical miles": ("nmi", "length"),
        "shipping container": ("container", "count"),
        "shipping containers": ("container", "count"),
    }
}


__all__ = [
    "DURATION_MAP",
    "MONEY_MAP",
    "MONTH_MAP",
    "ORDINAL_MAP",
    "PERCENT_MAP",
    "UNIT_DIMENSIONS",
    "UNIT_MAP",
    "WEEKDAY_MAP",
    "WRITTEN_NUMBER_MAP",
]
