"""Wikipedia retrieval for XIIV knowledge answers.

Live MediaWiki API: no key, no bulk download. Search ranks candidates;
article extracts supply the fact sentences.
"""
import json
import re
import urllib.parse
import urllib.request

_API = "https://en.wikipedia.org/w/api.php"
_UA = {"User-Agent": "XIIV-knowledge/0.1 (research prototype)"}

_QUESTION_PREFIX = re.compile(
    r"^(what|what's|whats|who|who's|whos|when|where|which|why|how|is|are|"
    r"was|were|do|does|did|can|could|tell me|explain)\b[\s']*",
    re.IGNORECASE)


def _stem(w):
    return re.sub(r"(s|es)$", "", w)


def _title_score(query, title, extract=""):
    def _words(s):
        return [_stem(w) for w in re.findall(r"[a-z]+", s.lower())
                if w not in _QSTOP]
    qw = _words(query)
    if not qw:
        return 0.0
    tw, ew = _words(title), _words(extract[:400])
    phrase = " ".join(qw)
    # exact-phrase hit (stemmed, stopwords dropped) in title or lead
    # paragraph dominates: "capital france" matches
    # "List of capitals of France" but not "Capital punishment in France".
    if phrase in " ".join(tw) or phrase in " ".join(ew):
        return 3.0 + len(set(qw) & set(tw)) / len(set(qw))
    return len(set(qw) & set(tw)) / len(set(qw))


_QSTOP = frozenset({"what", "which", "who", "when", "where", "why", "how",
                    "the", "a", "an", "of", "is", "are", "was", "in", "on"})


def clean_query(query):
    """Turn a natural question into a search-friendly phrase."""
    q = query.strip().rstrip("?")
    q = _QUESTION_PREFIX.sub("", q).strip()
    q = re.sub(r"^(is|are|was|were|the|a|an)\s+", "", q,
               flags=re.IGNORECASE).strip()
    return q or query.strip()


def _get(params):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(_API + "?" + qs, headers=_UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.load(r)


def search_wikipedia(query, k=3):
    """Return up to ``k`` articles: [{title, extract}]."""
    query = clean_query(query)
    try:
        res = _get({
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": k, "srnamespace": 0, "format": "json",
        })
    except Exception:
        return []
    hits = res.get("query", {}).get("search", [])
    # fetch extra candidates, then re-rank by title overlap: Wikipedia's
    # own ranking often prefers popular-but-wrong articles.
    titles = [h["title"] for h in hits[:8]]
    titles.sort(key=lambda t: _title_score(query, t), reverse=True)
    titles = titles[:k]
    if not titles:
        return []
    try:
        res = _get({
            "action": "query", "prop": "extracts", "exintro": 1,
            "explaintext": 1, "titles": "|".join(titles), "format": "json",
        })
    except Exception:
        return []
    pages = res.get("query", {}).get("pages", {}).values()
    out = []
    for p in pages:
        extract = p.get("extract", "")
        if extract:
            out.append({"title": p.get("title", ""),
                        "extract": _clean(extract)})
    # the extracts endpoint does not preserve request order: re-rank here,
    # now with extract text available for phrase matching
    out.sort(key=lambda h: _title_score(query, h["title"], h["extract"]),
             reverse=True)
    return out


def _clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return text
