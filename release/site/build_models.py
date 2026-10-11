"""Write site/models.html from site/data/stats.json and site/data/scan-pictures.json.
Then run release/site/build_charts.py to fill the chart markers."""
import json, os, html
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
def exact(h):
    iv = f" (95% interval {h['low']} to {h['high']})" if h.get("low") is not None else ""
    return f"{h['label']}: {h['metric']} {h['value']}{iv}"


T = "rAidiology · Research results"
D = "Seven open research models across X-ray, CT, MRI and mammogram, trained on public data, with their results on public test data."
top = (head + f"<title>{T}</title>\n  <meta name=\"description\" content=\"{D}\">\n"
       f"  <meta property=\"og:title\" content=\"{T}\">\n  <meta property=\"og:description\" content=\"{D}\">\n"
       "  <meta property=\"og:type\" content=\"website\">\n"
       "  <meta property=\"og:image\" content=\"https://raw.githubusercontent.com/LunarWerxs/rAidiology/main/site/img/og.png\">\n"
       "  <meta name=\"twitter:card\" content=\"summary_large_image\">\n" + head_tail + "</head>\n<body>\n" + nav)
out = [top, '  <main id="main">\n']
out.append('''    <section class="hero small-hero dark">
      <div class="wrap narrow">
        <h1>Research results</h1>
        <p class="lede">Seven open research models across X-ray, CT, MRI and mammogram, trained on public data, for education and research.</p>
        <p class="callout-dark">Research results on public test data. They say nothing about your own scan.</p>
      </div>
    </section>
    <section class="sec" aria-labelledby="read-h">
      <div class="wrap narrow">
        <h2 id="read-h">How to read the charts</h2>
        <ul class="short-list">
          <li><strong>AUROC:</strong> 0.5 is a coin toss and 1.0 is perfect.</li>
          <li><strong>Dice:</strong> how much the outline overlaps the dataset's own outlines; 1.0 is a perfect match.</li>
          <li><strong>Kappa:</strong> 0 is chance agreement and 1.0 is full agreement.</li>
          <li><strong>Dots:</strong> a filled dot is a held-out test, a hollow dot is development. The line through a dot is its 95% interval.</li>
        </ul>
        <ul class="model-links" aria-label="Jump to a model">
''')
for m in stats["models"]:
    out.append(f'          <li><a href="#{m["key"]}">{e(m["name"])}</a></li>\n')
out.append('        </ul>\n      </div>\n    </section>\n')
for i, m in enumerate(stats["models"]):
    k = m["key"]; p = pics[m["picture"]]
    out.append(f'''    <section class="sec model-sec{' alt' if i % 2 else ''}" id="{k}" aria-labelledby="{k}-h">
      <div class="wrap">
        <div class="model-head">
          <figure class="model-pic"><img src="{m["picture"]}" width="640" height="640" alt="{e(p["title"])}, a public dataset picture used as an example of this scan type." loading="lazy">
            <figcaption>{e(p["title"])}. {e(p["dataset"])}, {e(p["licence"])}. <a href="{e(p["url"])}">Source</a></figcaption></figure>
          <div>
            <p class="eyebrow">{e(m["scan"])}</p>
            <h2 id="{k}-h">{e(m["name"])}</h2>
            <p>{e(m["summary"])}</p>
            <p class="how-to">Research results on public test data. They say nothing about your own scan.</p>
          </div>
        </div>
        <h3>Results</h3>
        <!-- chart:{k}-headline -->
        <!-- /chart:{k}-headline -->
        <p class="how-to">Exact values: {e("; ".join(exact(h) for h in m["headline"]))}.</p>
''')
    for b in m["breakdowns"]:
        out.append(f'''        <h3>{e(b["title"])}</h3>
        <!-- chart:{b["id"]} -->
        <!-- /chart:{b["id"]} -->
        <p class="how-to">Test set: {e(b["set"])}.</p>
''')
    out.append('        <div class="model-cols">\n          <div>\n            <h3>Where it falls short</h3>\n            <ul class="short-list">\n')
    for s in m["struggles"]:
        out.append(f'              <li>{e(s["text"])}</li>\n')
    out.append('            </ul>\n          </div>\n          <div>\n            <h3>Trained and tested on</h3>\n            <ul class="short-list">\n')
    for d in m["datasets"]:
        out.append(f'              <li><a href="{e(d["url"])}">{e(d["name"])}</a>. Licence: {e(d["licence"])}</li>\n')
    out.append(f'''            </ul>
            <h3>Weights</h3>
            <p>{e(m["weights"])}</p>
            <p><a href="https://github.com/LunarWerxs/rAidiology/blob/main/release/cards/{k}.md">Read the model card on GitHub</a></p>
          </div>
        </div>
      </div>
    </section>
''')
out.append("  </main>\n" + foot)
open(os.path.join(S, "models.html"), "w", encoding="utf-8", newline="\n").write("".join(out))
