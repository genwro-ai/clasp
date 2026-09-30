"""Download only the data this repository uses, pinned to fixed revisions.

* The benchmark of Dong et al. (2024): a sparse checkout of its repository at a fixed commit,
  with only datasets/images, datasets/caption and datasets/evaluation_prompts (plus its README
  and LICENSE), into data/CIFC; none of its code. Needs git >= 2.25.
* CustomConcept101 of Kumari et al. (2023): the 515 photographs listed in data/cc101_images.txt,
  read out of the dataset's archive on the Hugging Face hub with HTTP range requests, so the rest
  of the 3.4 GB archive is never downloaded. zipfile checks the CRC of every member it extracts.
  They land in data/benchmark_dataset/, the layout scripts/prepare_cc101.py expects.

  python -m scripts.download_data              # both
  python -m scripts.download_data --only cifc  # or --only cc101
  python -m scripts.prepare_cc101 --src data/benchmark_dataset
"""
import argparse
import io
import os
import shutil
import subprocess
import urllib.request
import zipfile

CIFC_REPO = "https://github.com/JiahuaDong/CIFC.git"
CIFC_COMMIT = "9819aa843a4162e68becb3a2f46a97024b020d98"
CIFC_DIRS = ["datasets/images", "datasets/caption", "datasets/evaluation_prompts"]
CC101_URL = ("https://huggingface.co/datasets/nupurkmr9/custom-diffusion/resolve/"
             "540157a37f501911645cdaaec0eaee932f3d6a29/benchmark_dataset.zip")
CC101_ROOT = "benchmark_dataset/"


def fetch_cifc(dst):
    if os.path.isdir(os.path.join(dst, "datasets", "images")):
        print(f"[cifc] {dst} already has datasets/images, skipping")
        return
    git = ["git", "-C", dst]
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", "--sparse", CIFC_REPO, dst],
                   check=True)
    subprocess.run(git + ["sparse-checkout", "set", "--no-cone", "/LICENSE", "/README.md",
                          *[f"/{d}/" for d in CIFC_DIRS]], check=True)
    subprocess.run(git + ["checkout", "--quiet", CIFC_COMMIT], check=True)
    for d in CIFC_DIRS:
        assert os.path.isdir(os.path.join(dst, d)), d
    print(f"[cifc] {dst} at {CIFC_COMMIT[:7]}: {', '.join(CIFC_DIRS)}")


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (enough for zipfile)."""

    def __init__(self, url):
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.geturl()          # the signed CDN location the hub redirects to
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size or len(b) == 0:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
        with urllib.request.urlopen(req) as r:
            data = r.read()
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch_cc101(dst, manifest, limit=0):
    rels = [ln.strip() for ln in open(manifest, encoding="utf-8") if ln.strip()]
    if limit:
        rels = rels[:limit]
    todo = [r for r in rels if not os.path.exists(os.path.join(dst, r))]
    print(f"[cc101] {len(rels) - len(todo)} of {len(rels)} photographs already present")
    if not todo:
        return
    z = zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(CC101_URL), buffer_size=1 << 20))
    names = set(z.namelist())
    missing = [r for r in todo if CC101_ROOT + r not in names]
    if missing:
        raise SystemExit(f"{len(missing)} listed files are not in the archive, e.g. {missing[:3]}")
    for i, rel in enumerate(todo, 1):
        out = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with z.open(CC101_ROOT + rel) as src, open(out + ".part", "wb") as f:
            shutil.copyfileobj(src, f)       # raises on a CRC mismatch
        os.replace(out + ".part", out)
        if i % 50 == 0 or i == len(todo):
            print(f"[cc101] {i}/{len(todo)}", flush=True)
    print(f"[cc101] done -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["cifc", "cc101"], default=None)
    ap.add_argument("--cifc_dst", default="data/CIFC")
    ap.add_argument("--cc101_dst", default="data/benchmark_dataset")
    ap.add_argument("--manifest", default="data/cc101_images.txt")
    ap.add_argument("--limit", type=int, default=0, help="first N photographs only (a quick test)")
    a = ap.parse_args()
    if a.only in (None, "cifc"):
        fetch_cifc(a.cifc_dst)
    if a.only in (None, "cc101"):
        fetch_cc101(a.cc101_dst, a.manifest, a.limit)


if __name__ == "__main__":
    main()
