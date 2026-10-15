"""Browser app for the manual review of the preprocessed foot crops.

The automatic quality control (src/preprocess.py) marks every crop as usable or
not. This app shows the crops in a grid with the automatic decision pre-filled
(green = usable, red = rejected). The reviewer only clicks the mistakes and
saves the page; corrected and confirmed decisions go to
data/review/review_<reviewer>.csv, one file per reviewer, so several people can
review in parallel and the files are merged with --summary.

Queues:
    borderline  crops whose QC metrics are close to a threshold (most errors are here)
    random      a reproducible random sample, to measure the accuracy of the QC
    rejected    crops the QC rejected
    passed      crops the QC accepted

Usage (from the repository root, only the standard library is needed):
    python tools/review_app.py --reviewer anna --queue borderline
    python tools/review_app.py --reviewer anna --queue random --shard 1/3
    python tools/review_app.py --summary

Keys in the browser: click = usable/rejected, Shift+click = unsure,
Enter = save page and load the next one, O on a hovered crop = open the full original image.
"""

import argparse
import csv
import glob
import io
import json
import os
import random
import sys
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import preprocess as pp  # noqa: E402  (QC thresholds and the image decoder)

LABELS = ("good", "bad", "unsure")
BORDERLINE_DISTANCE = 0.03  # relative distance to a QC threshold; ~4,400 crops
REVIEW_FIELDS = ["foot_file", "image_file", "label", "qc_ok", "queue", "reviewer", "time"]


def borderline_score(row):
    """Smallest relative distance of any QC metric to its threshold (0 = on the threshold)."""
    distances = [
        abs(row.border_brightness - pp.MAX_BORDER_BRIGHTNESS) / pp.MAX_BORDER_BRIGHTNESS,
        abs(row.roughness - pp.MAX_ROUGHNESS) / pp.MAX_ROUGHNESS,
        abs(row.foot_height - pp.MIN_FOOT_HEIGHT) / pp.MIN_FOOT_HEIGHT,
        abs(row.foot_fill - pp.MIN_FILL) / pp.MIN_FILL,
        min(abs(row.foot_aspect - limit) / limit for limit in pp.FOOT_ASPECT_RANGE),
    ]
    if row.axis_reliable:  # the tilt rule only applies when the foot axis is trusted
        distances.append(abs(abs(row.tilt_deg) - pp.MAX_TILT) / pp.MAX_TILT)
    return min(distances)


def read_reviews(review_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(review_dir, "review_*.csv"))):
        with open(path, encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def build_queue(feet, queue, shard, seed=0):
    """Return the ordered list of crop rows to review for this queue and shard."""
    if queue == "borderline":
        scored = sorted(feet, key=borderline_score)
        items = [r for r in scored if borderline_score(r) < BORDERLINE_DISTANCE]
    elif queue == "random":
        items = list(feet)
        random.Random(seed).shuffle(items)
    elif queue == "rejected":
        items = [r for r in feet if not r.qc_ok]
    else:
        items = [r for r in feet if r.qc_ok]
    index, count = shard
    return items[index - 1::count]


class Crop:
    """One row of feet.csv with typed QC fields."""

    def __init__(self, row):
        self.foot_file = row["foot_file"]
        self.image_file = row["image_file"]
        self.qc_ok = int(float(row["qc_ok"]))
        for name in ("border_brightness", "roughness", "foot_height", "foot_aspect", "foot_fill", "tilt_deg"):
            setattr(self, name, float(row[name]))
        self.axis_reliable = int(float(row["axis_reliable"]))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Foot crop review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;background:#1e1e1e;color:#ddd}
header{position:sticky;top:0;background:#2a2a2a;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:1}
header b{color:#fff} button{font-size:15px;padding:6px 14px;cursor:pointer}
#grid{display:flex;flex-wrap:wrap;gap:8px;padding:16px}
.tile{position:relative;border:4px solid;border-radius:6px;cursor:pointer;line-height:0}
.tile img{width:96px;height:224px;display:block}
.good{border-color:#3ba55d}.bad{border-color:#e5484d;opacity:.55}.unsure{border-color:#f5a524}
.tag{position:absolute;bottom:0;left:0;right:0;font-size:10px;line-height:14px;background:#000a;padding:0 3px}
.help{color:#999;font-size:13px}
</style></head><body>
<header><b>Foot crop review</b><span id="info"></span>
<button onclick="save()">Save page and next (Enter)</button>
<span class="help">click: usable/rejected · Shift+click: unsure · O over a crop: original image</span></header>
<div id="grid"></div>
<script>
let items=[], hovered=null;
const cls={good:'good',bad:'bad',unsure:'unsure'};
async function load(){
  const r=await fetch('/api/page'); const d=await r.json(); items=d.items;
  document.getElementById('info').textContent=`queue: ${d.queue} · reviewer: ${d.reviewer} · done: ${d.done}/${d.total}`;
  const g=document.getElementById('grid'); g.innerHTML='';
  if(!items.length){g.textContent='Queue finished. Thank you!';return;}
  items.forEach((it,i)=>{
    const t=document.createElement('div'); t.className='tile '+it.label;
    t.title=`${it.foot_file}\\n${it.metrics}`;
    t.innerHTML=`<img src="/crop/${encodeURIComponent(it.foot_file)}"><div class="tag">${it.foot_file}</div>`;
    t.onclick=e=>{it.label=e.shiftKey?'unsure':(it.label==='good'?'bad':'good'); t.className='tile '+it.label;};
    t.onmouseenter=()=>hovered=it; t.onmouseleave=()=>hovered=null;
    g.appendChild(t);
  });
  window.scrollTo(0,0);
}
async function save(){
  if(!items.length) return;
  await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(items.map(it=>({foot_file:it.foot_file,label:it.label})))});
  load();
}
document.addEventListener('keydown',e=>{
  if(e.key==='Enter') save();
  if((e.key==='o'||e.key==='O') && hovered) window.open('/original/'+encodeURIComponent(hovered.image_file),'_blank');
});
load();
</script></body></html>"""


class ReviewServer:
    def __init__(self, args):
        self.args = args
        self.feet_dir = os.path.join(args.data_dir, "processed", "feet")
        with open(os.path.join(args.data_dir, "processed", "feet.csv"), encoding="utf-8") as f:
            feet = [Crop(row) for row in csv.DictReader(f)]
        self.by_file = {c.foot_file: c for c in feet}
        self.review_dir = os.path.join(args.data_dir, "review")
        os.makedirs(self.review_dir, exist_ok=True)
        self.out_path = os.path.join(self.review_dir, f"review_{args.reviewer}.csv")
        self.queue = build_queue(feet, args.queue, args.shard)
        # Every crop reviewed by anyone is skipped, so parallel reviewers do not duplicate work.
        self.done = {r["foot_file"] for r in read_reviews(self.review_dir)}

    def next_page(self):
        pending = [c for c in self.queue if c.foot_file not in self.done][:self.args.page_size]
        items = [{"foot_file": c.foot_file, "image_file": c.image_file,
                  "label": "good" if c.qc_ok else "bad",
                  "metrics": (f"border {c.border_brightness:.2f} · rough {c.roughness:.2f} · height {c.foot_height:.2f}"
                              f" · aspect {c.foot_aspect:.2f} · fill {c.foot_fill:.2f}"
                              f" · tilt {c.tilt_deg:+.0f}°{'' if c.axis_reliable else ' (axis not trusted)'}")}
                 for c in pending]
        done = sum(1 for c in self.queue if c.foot_file in self.done)
        return {"items": items, "queue": self.args.queue, "reviewer": self.args.reviewer,
                "done": done, "total": len(self.queue)}

    def save(self, decisions):
        new_file = not os.path.exists(self.out_path)
        with open(self.out_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            if new_file:
                writer.writeheader()
            now = datetime.now().isoformat(timespec="seconds")
            for d in decisions:
                crop = self.by_file.get(d.get("foot_file"))
                if crop is None or d.get("label") not in LABELS:
                    continue
                writer.writerow({"foot_file": crop.foot_file, "image_file": crop.image_file, "label": d["label"],
                                 "qc_ok": crop.qc_ok, "queue": self.args.queue, "reviewer": self.args.reviewer,
                                 "time": now})
                self.done.add(crop.foot_file)

    def original_png(self, image_file):
        """Decoded, inverted full image (both feet) as PNG bytes, for context."""
        if not self.args.raw or os.path.basename(image_file) != image_file:
            return None
        image = pp.decode(os.path.join(self.args.raw, image_file))
        if image is None:
            return None
        buffer = io.BytesIO()
        pp.Image.fromarray((image * 255).astype("uint8")).resize(
            (image.shape[1] * 2, image.shape[0] * 2)).save(buffer, "PNG")
        return buffer.getvalue()

    def handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, body, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urllib.parse.unquote(self.path)
                if path == "/":
                    self.send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif path == "/api/page":
                    self.send(json.dumps(server.next_page()).encode(), "application/json")
                elif path.startswith("/crop/"):
                    name = os.path.basename(path[len("/crop/"):])
                    file_path = os.path.join(server.feet_dir, name)
                    if name in server.by_file and os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            self.send(f.read(), "image/png")
                    else:
                        self.send(b"not found", "text/plain", 404)
                elif path.startswith("/original/"):
                    png = server.original_png(path[len("/original/"):])
                    if png:
                        self.send(png, "image/png")
                    else:
                        self.send(b"original not available (start with --raw)", "text/plain", 404)
                else:
                    self.send(b"not found", "text/plain", 404)

            def do_POST(self):
                if self.path != "/api/save":
                    self.send(b"not found", "text/plain", 404)
                    return
                length = int(self.headers.get("Content-Length", 0))
                server.save(json.loads(self.rfile.read(length) or b"[]"))
                self.send(b"{}", "application/json")

        return Handler


def summary(data_dir):
    """Merge all reviewers' files and report how often the reviewers disagree with the QC."""
    reviews = read_reviews(os.path.join(data_dir, "review"))
    if not reviews:
        print("no reviews yet")
        return
    latest = {}
    for r in reviews:  # a later decision on the same crop overrides an earlier one
        latest[(r["foot_file"], r["reviewer"])] = r
    merged = {}
    for (foot_file, _), r in latest.items():
        merged.setdefault(foot_file, []).append(r)

    rows, stats = [], {}
    for foot_file, decisions in merged.items():
        votes = [d["label"] for d in decisions if d["label"] != "unsure"]
        label = max(set(votes), key=votes.count) if votes else "unsure"
        qc_ok = int(decisions[0]["qc_ok"])
        rows.append({"foot_file": foot_file, "review_label": label, "qc_ok": qc_ok, "reviewers": len(decisions)})
        for d in decisions:
            key = (d["queue"], "qc usable" if qc_ok else "qc rejected")
            stats.setdefault(key, {"good": 0, "bad": 0, "unsure": 0})[d["label"]] += 1

    out_path = os.path.join(data_dir, "review", "review_merged.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} reviewed crops by {len({r['reviewer'] for r in reviews})} reviewer(s) -> {out_path}")
    print(f"{'queue':12s} {'QC decision':12s} {'good':>6s} {'bad':>6s} {'unsure':>6s}  QC agrees")
    for (queue, qc), c in sorted(stats.items()):
        agree = c["good"] if qc == "qc usable" else c["bad"]
        decided = c["good"] + c["bad"]
        share = f"{agree / decided:.0%}" if decided else "-"
        print(f"{queue:12s} {qc:12s} {c['good']:6d} {c['bad']:6d} {c['unsure']:6d}  {share}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    parser.add_argument("--reviewer", help="your name; reviews go to data/review/review_<name>.csv")
    parser.add_argument("--queue", choices=["borderline", "random", "rejected", "passed"], default="borderline")
    parser.add_argument("--shard", default="1/1", help="i/n: review only every n-th crop starting at i (split work)")
    parser.add_argument("--page-size", type=int, default=48)
    parser.add_argument("--raw", default=next(iter(glob.glob(os.path.join(ROOT, "PEDIS*", "PEDIS"))), None),
                        help="raw image folder, for the O key (full original image)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--summary", action="store_true", help="merge all reviews and print QC agreement")
    args = parser.parse_args()

    if args.summary:
        summary(args.data_dir)
        return
    if not args.reviewer or not args.reviewer.replace("-", "").replace("_", "").isalnum():
        parser.error("--reviewer is required (letters, digits, - and _ only)")
    index, count = (int(x) for x in args.shard.split("/"))
    if not 1 <= index <= count:
        parser.error("--shard must be i/n with 1 <= i <= n")
    args.shard = (index, count)

    app = ReviewServer(args)
    print(f"queue '{args.queue}' shard {index}/{count}: {len(app.queue)} crops, "
          f"{sum(1 for c in app.queue if c.foot_file in app.done)} already reviewed")
    print(f"open http://localhost:{args.port}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", args.port), app.handler()).serve_forever()


if __name__ == "__main__":
    main()
