"""Historical news (2020-2026) via Wikipedia's Current events portal.

The portal keeps a daily archive of world news; each day is a page like
``Portal:Current_events/2023_March_15``. No API key needed. GDELT would be
the richer source, but its API rate-limits this deployment's egress IP, so
the GDELT path is implemented but disabled by default (see below).
"""
import datetime
import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

_API = "https://en.wikipedia.org/w/api.php"
_UA = {"User-Agent": "XIIV-knowledge/0.1 (research prototype)"}

_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}

# GDELT 2.1 doc API — richer news coverage, but unreachable from some
# egress IPs (rate-limited). Flip to True where it works.
GDELT_ENABLED = False


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items, self._buf, self._in_li = [], [], False

    def handle_starttag(self, tag, attrs):
        if tag == "li":
            self._in_li, self._buf = True, []

    def handle_endtag(self, tag):
        if tag == "li" and self._in_li:
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if len(text) > 40:
                self.items.append(text)
            self._in_li = False

    def handle_data(self, data):
        if self._in_li:
            self._buf.append(data)


def _portal_page(year, month, day):
    name = datetime.date(year, month, day).strftime("%Y_%B_%d")
    params = {"action": "parse", "page": f"Portal:Current_events/{name}",
              "prop": "text", "format": "json"}
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(_API + "?" + qs, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            res = json.load(r)
    except Exception:
        return []
    html = res.get("parse", {}).get("text", {}).get("*", "")
    if not html:
        return []
    p = _TextExtractor()
    try:
        p.feed(html)
    except Exception:
        return []
    return p.items


def detect_dates(question):
    """Find (year, month, day) tuples referenced in ``question``.

    Defaults to a small window around 'today in history' when only a
    month/day is given; empty list when no date is referenced.
    """
    q = question.lower()
    years = [int(y) for y in re.findall(r"\b(20(?:2\d|30))\b", q)]
    years = [y for y in years if 2020 <= y <= 2026]
    months = [(m, _MONTHS[m]) for m in _MONTHS if m in q]
    days = [int(d) for d in re.findall(r"\b([12]?\d|3[01])(?:st|nd|rd|th)?\b",
                                       q)]
    dates = []
    today = datetime.date.today()
    if years and months:
        for y in years:
            for _, mo in months:
                d = days[0] if days and 1 <= days[0] <= 28 else 15
                try:
                    dates.append((y, mo, d))
                except ValueError:
                    pass
    elif months and days:
        # month + day, no year: check that month/day across 2020-2026
        for y in range(2020, 2027):
            for _, mo in months:
                if 1 <= days[0] <= 28:
                    try:
                        dates.append((y, mo, days[0]))
                    except ValueError:
                        pass
    return dates[:6]


def search_news(question, k=3):
    """Return up to ``k`` news items: [{date, text}]."""
    dates = detect_dates(question)
    if not dates:
        return []
    keywords = [w for w in re.findall(r"[a-zA-Z]{4,}", question.lower())
                if w not in _STOP]
    items = []
    for (y, mo, d) in dates:
        for text in _portal_page(y, mo, d):
            low = text.lower()
            if any(kw in low for kw in keywords):
                items.append({"date": f"{y}-{mo:02d}-{d:02d}", "text": text})
                if len(items) >= k:
                    return items
    return items


_STOP = frozenset((
    "what", "when", "where", "which", "that", "with", "from", "happened",
    "news", "about", "were", "was", "have", "has", "been", "will",
))
