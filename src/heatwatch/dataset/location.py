"""GPS position -> city (offline: package 'reverse_geocode', GeoNames)."""
import re

try:
    import reverse_geocode
except ImportError:
    reverse_geocode = None
    print("\n  ! Package 'reverse_geocode' is not installed: GPS positions cannot be turned into a city /\n"
          "    village and are shown as 'lat, lon'. Install it with:  python -m pip install reverse_geocode\n")


# ------------------------------------------------------------------ GPS position -> city (offline)
def city(r):
    """Municipality in NL, otherwise the place name."""
    county = (r.get("county") or "").replace("Gemeente ", "").strip()
    return county if r.get("country_code") == "NL" and county else r.get("city")


def place_names(lats, lons):
    """City per position; 'lat, lon' if the package is missing; None without position."""
    ok = [la == la and lo == lo for la, lo in zip(lats, lons)]               # x == x: not NaN
    if reverse_geocode is None:
        return [f"{la:.4f}, {lo:.4f}" if k else None for la, lo, k in zip(lats, lons, ok)]
    pts = [(la, lo) for la, lo, k in zip(lats, lons, ok) if k]
    found = iter(city(r) for r in reverse_geocode.search(pts)) if pts else iter(())
    return [next(found) if k else None for k in ok]


# ------------------------------------------------------------------ place name in a text (Garmin Connect title) -> place and coordinates
ALIASES = {"den bosch": "'s-hertogenbosch", "the hague": "den haag", "s-gravenhage": "den haag",
           "'s-gravenhage": "den haag"}
_INDEX = None


def _index():
    """lower-case place / municipality name -> (lat, lon); NL first, then the largest place."""
    global _INDEX
    if _INDEX is None:
        _INDEX = {}
        locs = reverse_geocode.GeocodeData()._locations
        for loc in sorted(locs, key=lambda x: (x["country_code"] != "NL", -int(x.get("population") or 0))):
            for name in (loc["city"], (loc.get("county") or "").replace("Gemeente ", "")):
                key = name.strip().lower()
                if key and key not in _INDEX:
                    _INDEX[key] = (loc["latitude"], loc["longitude"])
    return _INDEX


def place_lookup(text):
    """Find a known place in a text such as a Garmin Connect title:
    e.g. 'Duurloop Delft' -> Delft,
    Tries the whole text, then shorter starts of it, then every capitalised word. -> (name, lat, lon) or None."""
    if reverse_geocode is None or not isinstance(text, str) or not text.strip():
        return None
    idx = _index()
    words = re.findall(r"[\w'][\w'.-]*", text)
    tries = [text.strip()] + [" ".join(words[:k]) for k in range(len(words) - 1, 0, -1)] + \
            [w for w in words[1:] if w[:1].isupper() and len(w) >= 3]
    for name in tries:
        key = ALIASES.get(name.lower(), name.lower())
        if key in idx:
            return (name, *idx[key])
    return None


def place_coords(names):
    """Place name -> (lat, lon) of that place, offline; (nan, nan) if no known place is found."""
    out = []
    for n in names:
        r = place_lookup(n)
        out.append((r[1], r[2]) if r else (float("nan"), float("nan")))
    return out
