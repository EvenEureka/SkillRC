#!/usr/bin/env python3
"""Walk evaluation session 0 through a running WebShop app and print the page URLs.

Start the app first with serve_webshop.py, then open /_goal0 once so session 0 shows
the instruction recorded during evaluation. The walk searches, opens the requested
product, selects the requested size, and buys it; the last URL is the score page.
Screenshots were taken with headless Firefox, e.g.
  firefox --headless --window-size=1280,1950 --screenshot results.png "<results URL>"

Usage: webshop_walk.py [BASE_URL]
"""
import html
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:3517"
QUERY = "double sided machine washable decorative pillows 28 x 28"
ASIN, SIZE = "B0743JKHBV", '28"x28"'


def get(url):
    with urllib.request.urlopen(url, timeout=300) as r:
        return r.read().decode()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


get(BASE + "/fixed_0")
try:
    urllib.request.build_opener(NoRedirect).open(BASE + "/fixed_0", data=urllib.parse.urlencode(
        {"search_query": QUERY}).encode())
    raise SystemExit("expected a redirect to the results page")
except urllib.error.HTTPError as e:
    results = urllib.parse.urljoin(BASE, e.headers["Location"])
page = get(results)
items = [html.unescape(i) for i in re.findall(r'href="([^"]*item_page[^"]*)"', page)]
asins = list(dict.fromkeys(re.search(r"item_page/[^/]+/([A-Z0-9]+)/", i).group(1) for i in items))
item = urllib.parse.urljoin(BASE, next(i for i in items if ASIN in i))
radios = re.findall(r'<input type="radio"[^>]*value="([^"]*)"[^>]*data-url="([^"]*)"', get(item))
selected = BASE + next(html.unescape(d) for v, d in radios if html.unescape(v).replace(" ", "") == SIZE)
done = BASE + html.unescape(re.findall(r'action="([^"]*done[^"]*)"', get(selected))[0])
score = re.search(r"Your score.*?<[^>]*>\s*([0-9.]+)", get(done), re.S)
print(json.dumps({"search": BASE + "/fixed_0", "results": results, "rank": asins.index(ASIN) + 1,
                  "item": item, "item_selected": selected, "done": done,
                  "reward": score.group(1) if score else None}, indent=1))
