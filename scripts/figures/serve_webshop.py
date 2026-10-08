#!/usr/bin/env python3
"""Serve the official WebShop Flask app (1,000-product catalogue, synthetic goals) for screenshots.

Usage: serve_webshop.py REPO_DIR PORT
REPO_DIR holds the official web_agent_site/ code, templates, static files and data/.
Only the Lucene search index is replaced, by BM25 over product title/query/category, because building
the official index needs Java. Goal construction follows SimServer (seed 233 shuffle),
so session "fixed_<i>" is evaluation session i.
"""
import json, sys
repo, port = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, repo)

import types  # noqa: E402
_lucene = types.ModuleType("pyserini.search.lucene")
_lucene.LuceneSearcher = object          # official index not used (needs a newer Java)
for _name, _mod in (("pyserini", types.ModuleType("pyserini")),
                    ("pyserini.search", types.ModuleType("pyserini.search")),
                    ("pyserini.search.lucene", _lucene)):
    sys.modules[_name] = _mod
from rank_bm25 import BM25Okapi  # noqa: E402
import web_agent_site.utils as U  # noqa: E402
U.DEBUG_PROD_SIZE = 1000

import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("ws_app", f"{repo}/app.py")
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)
A.app.template_folder = f"{repo}/templates"
A.app.static_folder = f"{repo}/static"
A.DEBUG_PROD_SIZE = 1000


class _Doc:
    def __init__(self, asin): self._asin = asin
    def raw(self): return json.dumps({"id": self._asin})


class _Hit:
    def __init__(self, asin): self.docid = asin


class TitleBM25:
    """Stand-in for the Lucene index: BM25 over product title, query, and category text."""
    def __init__(self, products):
        self.asins = [p["asin"] for p in products]
        def text(p):  # title plus the catalogue's own query and category fields
            return " ".join(str(p.get(k) or "") for k in ("Title", "name", "query", "category", "product_category"))
        self.bm25 = BM25Okapi([text(p).lower().replace("›", " ").split() for p in products])
    def search(self, query, k=50):
        scores = self.bm25.get_scores(query.lower().split())
        order = sorted(range(len(self.asins)), key=lambda i: -scores[i])[:k]
        return [_Hit(self.asins[i]) for i in order]
    def doc(self, docid): return _Doc(docid)


_orig_load, _orig_goals = A.load_products, A.get_goals
A.load_products = lambda filepath, num_products=None, **kw: _orig_load(filepath=filepath, num_products=1000, human_goals=0)
A.get_goals = lambda all_products, product_prices, *a, **kw: _orig_goals(all_products, product_prices, 0)


def _init_search_engine(num_products=None):
    return TitleBM25(A.all_products)


A.init_search_engine = _init_search_engine
LOGGED_SESSION0 = ('Find me double sided, machine washable decorative pillows with printing technology '
                   'with size: 28" x 28", and price lower than 50.00 dollars')


@A.app.route("/_goal0")
def _goal0():
    """Show the session-0 goal; set its text to the one logged during evaluation (prices are
    sampled without a seed at load time, so only the price bound differs)."""
    g = A.goals[0]
    g["instruction_text"] = LOGGED_SESSION0
    g["price_upper"] = 50.0
    A.user_sessions.clear()
    return json.dumps({k: g[k] for k in ("asin", "instruction_text", "attributes", "goal_options", "price_upper")
                       if k in g}, default=str)


A.app.run(host="127.0.0.1", port=port, threaded=False)
