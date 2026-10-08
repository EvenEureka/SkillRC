# Case-study figures

The environment images in Appendix L (`paper/figures/case_*.tex`) are built here. None of
these steps runs a model.

## ALFWorld scenes (AI2-THOR)

1. `extract_alfworld_scenes.py JSON_ZIP TWPDDL_ZIP SCENE_DIR` reads the six trials from
   the ALFWorld release archives (`json_2.1.1_json.zip`, `json_2.1.2_tw-pddl.zip`).
2. `render_cases.py SCENE_DIR RENDER_DIR` restores each trial in AI2-THOR 2.1.0 (the
   version ALFWorld pins), searches the reachable positions for a first-person view that
   shows the places of the expert solution, and saves that view, its 2D instance boxes, the
   top-down map, and the map camera parameters. It needs an X display; on our cluster it
   runs under `Xvfb :77 -screen 0 1600x1200x24 +extension GLX`. With Flask >= 2 the
   AI2-THOR 2.1.0 server drops its socket, so pin `flask==1.1.4 werkzeug==1.0.1
   jinja2<3 itsdangerous<2 markupsafe<2.1`.
3. `prepare_case_assets.py RENDER_DIR` writes `paper/figures/cases/<case>.json` and the
   JPEG images. Map labels are projected exactly from the orthographic map camera.

## WebShop pages

`serve_webshop.py REPO_DIR PORT` runs the official WebShop Flask app (code, templates, and
the 1,000-product catalogue) with synthetic goals, so session `fixed_<i>` is evaluation
session i. The Lucene search index is replaced by a BM25 stand-in because building it needs
a newer Java runtime. `webshop_walk.py` walks session 0 to a purchase (reward 1.0) and
prints the page URLs, which were captured with headless Firefox and cropped into
`paper/figures/cases/ws_*.jpg`.

## Figures

`python paper/figures/make_case_figures.py` writes the TikZ figures from the assets.
