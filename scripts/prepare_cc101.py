"""Link the CustomConcept101 photographs used by the two sequences into
data/customconcept101/<concept>/, following data/cc101_images.txt (515 files of 90 concepts: the
first 225, of 40 concepts, are the fifty-concept sequence, the rest the fifty concepts the
hundred-concept sequence adds). The images are not redistributed; point --src at the dataset's
`benchmark_dataset` folder (scripts/download_data.sh fetches it). A space in a file name becomes
'_' in the link, which is the name our captions and cutouts use.

  python -m scripts.prepare_cc101 --src data/benchmark_dataset
"""
import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="CustomConcept101 benchmark_dataset directory")
    ap.add_argument("--dst", default="data/customconcept101")
    ap.add_argument("--manifest", default="data/cc101_images.txt")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlinking")
    a = ap.parse_args()
    missing = 0
    for line in open(a.manifest, encoding="utf-8"):
        rel = line.strip()
        if not rel:
            continue
        src, dst = os.path.join(a.src, rel), os.path.join(a.dst, rel.replace(" ", "_"))
        if not os.path.exists(src):
            print(f"missing: {src}")
            missing += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            continue
        if a.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.abspath(src), dst)
    if missing:
        raise SystemExit(f"{missing} files not found under {a.src}")
    print(f"done -> {a.dst}")


if __name__ == "__main__":
    main()
