"""Write site/models.html from site/data/stats.json and site/data/scan-pictures.json.
Then run release/site/build_charts.py to fill the chart markers (tiles, each model's charts and its numbers).
The header and footer come from site/run.html, so every page shares them; the styles are brand.css and results.css."""
import html
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_charts import text as t, webp_size  # noqa: E402

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(R, "site")
stats = json.load(open(os.path.join(S, "data/stats.json"), encoding="utf-8"))
pics = {p["file"]: p for p in json.load(open(os.path.join(S, "data/scan-pictures.json"), encoding="utf-8"))}
run = open(os.path.join(S, "run.html"), encoding="utf-8").read()
e = html.escape


head = run[:run.index("<title>")]
nav = run[run.index('  <a class="skip"'):run.index("  <main")].replace(' aria-current="page"', "")
nav = nav.replace('<a href="models.html">', '<a href="models.html" aria-current="page">')
foot = run[run.index("  <footer"):]
head_tail = run[run.index('  <link rel="icon"'):run.index("</head>")]
# This page loads the shared brand.css and its own results.css, never another page's stylesheet.
head_tail = re.sub(r'  <link rel="stylesheet" href="css/(?!brand\.css)[\w-]+\.css">\n', "", head_tail)
head_tail = head_tail.replace('<link rel="stylesheet" href="css/brand.css">\n',
                              '<link rel="stylesheet" href="css/brand.css">\n  <link rel="stylesheet" href="css/results.css">\n')
assert "css/results.css" in head_tail, "run.html no longer loads css/brand.css on its own line"

T = "rAidiology · Research results"
D = "Seven open research models across X-ray, CT, MRI and mammogram, trained on public data, with their results on public test data."
top = (head + f"<title>{T}</title>\n  <meta name=\"description\" content=\"{D}\">\n"
       f"  <meta property=\"og:title\" content=\"{T}\">\n  <meta property=\"og:description\" content=\"{D}\">\n"
       "  <meta property=\"og:type\" content=\"website\">\n"
       "  <meta property=\"og:image\" content=\"https://raw.githubusercontent.com/LunarWerxs/rAidiology/main/site/img/og.png\">\n"
       "  <meta name=\"twitter:card\" content=\"summary_large_image\">\n" + head_tail + "</head>\n<body>\n" + nav)
out = [top, '  <main id="main">\n']
out.append(f'''    <section class="page-head results-head" aria-labelledby="page-h">
      <div class="wrap">
        <h1 id="page-h">Research results</h1>
        <p class="lede">{t("Seven open research models across X-ray, CT, MRI and mammogram, trained on public data, for education and research.")}</p>
        <p class="callout">Research results on public test data. They say nothing about your own scan.</p>
        <h2 class="sr">The seven models</h2>
        <!-- chart:tiles -->
        <!-- /chart:tiles -->
      </div>
    </section>
''')
for i, m in enumerate(stats["models"]):
    k = m["key"]
    p = pics[m["picture"]]
    w, h = webp_size(m["picture"])
    bds = m.get("breakdowns", [])
    # The picture beside the name, summary and main results; then the breakdown charts with the facts beside
    # them (or the facts alone, in two columns); then "Show the numbers".
    out.append(f'''    <section class="model-sec{' soft' if i % 2 else ''}" id="{k}" aria-labelledby="{k}-h">
      <div class="wrap">
        <div class="model-top">
          <div class="model-name">
            <p class="eyebrow">{e(m["scan"])}</p>
            <h2 id="{k}-h">{e(m["name"])}</h2>
          </div>
          <figure class="lightbox model-pic">
            <img src="{m["picture"]}" width="{w}" height="{h}" alt="{e(p["title"])}, a public dataset picture used as an example of this scan type." loading="lazy">
            <span class="pic-chip" aria-hidden="true">{e(p["title"])}</span>
            <figcaption>{e(p["dataset"])}, {e(p["licence"])}. <a href="{e(p["url"])}">Source</a></figcaption>
          </figure>
          <div class="model-main">
            <p class="model-sum">{t(m["summary"])}</p>
            <!-- chart:{k}-headline -->
            <!-- /chart:{k}-headline -->
          </div>
        </div>
        <div class="model-more{' has-bd' if bds else ''}">
''')
    if bds:
        out.append('          <div class="model-bd">\n')
        for b in bds:
            out.append(f'''            <!-- chart:{b["id"]} -->
            <!-- /chart:{b["id"]} -->
''')
        out.append('          </div>\n')
    out.append('''          <div class="model-facts">
            <div>
              <h3>Where it falls short</h3>
              <ul class="short-list">
''')
    for st in m["struggles"]:
        out.append(f'                <li>{t(st["text"])}</li>\n')
    out.append('''              </ul>
            </div>
            <div>
              <h3>Trained and tested on</h3>
              <ul class="data-list">
''')
    for d in m["datasets"]:
        lic = e(d["licence"]).replace(" ", "&nbsp;") if len(d["licence"]) <= 24 else e(d["licence"])
        out.append(f'                <li><a href="{e(d["url"])}">{t(d["name"])}</a> <span class="lic">{lic}</span></li>\n')
    out.append(f'''              </ul>
              <p class="model-card"><a class="btn secondary small" href="https://github.com/LunarWerxs/rAidiology/blob/main/release/cards/{k}.md">Model card on GitHub</a></p>
            </div>
          </div>
        </div>
        <!-- chart:{k}-numbers -->
        <!-- /chart:{k}-numbers -->
      </div>
    </section>
''')
out.append("  </main>\n" + foot)
open(os.path.join(S, "models.html"), "w", encoding="utf-8", newline="\n").write("".join(out))
