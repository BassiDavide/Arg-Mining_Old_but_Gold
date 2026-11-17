#!/usr/bin/env python3
"""Rehydrate the dehydrated YouTube comment-chain dataset.

  1. Paste your YouTube Data API v3 key into API_KEY below
     (or set the YOUTUBE_API_KEY environment variable).
  2. Run:  python rehydrate.py

Restores CommentText, AuthorName, ParentCommentText and responding_to.author
for train_dataset.json and test_dataset.json, writing the results to ./rehydrated/.
No third-party packages are required. See README.md for details.
"""

API_KEY = ""  # <-- paste your YouTube Data API v3 key here

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_URL = "https://www.googleapis.com/youtube/v3/comments"
BATCH_SIZE = 50  # max comment IDs per comments.list request
MAX_RETRIES = 5
SPLITS = ("train_dataset.json", "test_dataset.json")
HERE = Path(__file__).resolve().parent


def api_get(params):
    """GET the comments endpoint, retrying transient errors with exponential backoff."""
    url = API_URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")
            try:
                err = json.loads(body)["error"]
                reasons = {x.get("reason") for x in err.get("errors", [])}
                message = err.get("message", body)
            except (ValueError, KeyError, TypeError):
                reasons, message = set(), body
            transient = e.code >= 500 or e.code == 429 or "rateLimitExceeded" in reasons
            if not transient:  # e.g. invalid key, API not enabled, daily quota exceeded
                sys.exit(f"\nYouTube API error {e.code} ({', '.join(sorted(r for r in reasons if r)) or 'unknown'}): "
                         f"{message}\nProgress so far is saved in the cache; re-run to resume.")
            error = f"HTTP {e.code}: {message}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = str(e)
        wait = 2 ** attempt
        print(f"  request failed ({error}); retrying in {wait}s", file=sys.stderr)
        time.sleep(wait)
    sys.exit("\nToo many failed requests. Progress is saved in the cache; re-run to resume.")


def fetch_comments(comment_ids, api_key, cache, cache_path):
    """Fill `cache` with {comment_id: {"text", "author"} or None if unavailable}."""
    todo = sorted(cid for cid in comment_ids if cid not in cache)
    if not todo:
        print("All comments already in cache.")
        return
    n_batches = -(-len(todo) // BATCH_SIZE)
    print(f"Fetching {len(todo)} comments in {n_batches} requests "
          f"({len(comment_ids) - len(todo)} already cached)...")
    for i in range(n_batches):
        batch = todo[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        data = api_get({"part": "snippet", "id": ",".join(batch), "textFormat": "plainText", "key": api_key})
        found = {}
        for item in data.get("items", []):
            snippet = item.get("snippet", {})
            found[item["id"]] = {
                "text": snippet.get("textOriginal", snippet.get("textDisplay")),
                "author": snippet.get("authorDisplayName"),
            }
        for cid in batch:
            cache[cid] = found.get(cid)
        save_json(cache, cache_path)
        print(f"  [{i + 1}/{n_batches}] {len(found)}/{len(batch)} comments retrieved")


def rehydrate_message(msg, comments):
    """Return a copy of `msg` with the removed fields restored, in the original field order."""
    own = comments.get(msg["CommentID"])
    rt = msg.get("responding_to")
    parent = comments.get(rt["comment_id"]) if rt else None
    out = {}
    for key, value in msg.items():
        out[key] = value
        if key == "ParentCommentID":
            out["CommentText"] = own["text"] if own else None
            out["AuthorName"] = own["author"] if own else None
        elif key == "Level":
            # Top-level comments have an empty ParentCommentText in the original data.
            out["ParentCommentText"] = (parent["text"] if parent else None) if rt else ""
        elif key == "responding_to" and rt is not None:
            out[key] = {"author": parent["author"] if parent else None, **rt}
    return out


def rehydrate(data, comments):
    return {
        **data,
        "discussions": {
            chain_id: {**chain, "messages": [rehydrate_message(m, comments) for m in chain["messages"]]}
            for chain_id, chain in data["discussions"].items()
        },
    }


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, default=HERE, help="folder with the dehydrated JSON files")
    ap.add_argument("--out-dir", type=Path, default=HERE / "rehydrated", help="output folder")
    args = ap.parse_args()

    api_key = (API_KEY or os.environ.get("YOUTUBE_API_KEY", "")).strip()
    if not api_key:
        sys.exit("No API key found. Paste your YouTube Data API v3 key into API_KEY at the top of "
                 "rehydrate.py, or set the YOUTUBE_API_KEY environment variable. See README.md.")

    datasets = {name: load_json(args.in_dir / name) for name in SPLITS}
    comment_ids = {m["CommentID"]
                   for data in datasets.values()
                   for chain in data["discussions"].values()
                   for m in chain["messages"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.out_dir / "comments_cache.json"
    cache = load_json(cache_path) if cache_path.exists() else {}
    fetch_comments(comment_ids, api_key, cache, cache_path)

    print()
    for name, data in datasets.items():
        out = rehydrate(data, cache)
        save_json(out, args.out_dir / name)
        msgs = [m for chain in out["discussions"].values() for m in chain["messages"]]
        n_missing = sum(m["CommentText"] is None for m in msgs)
        print(f"{name}: {len(msgs)} messages, {len(msgs) - n_missing} with text, "
              f"{n_missing} unavailable -> {args.out_dir / name}")

    missing = sorted(cid for cid in comment_ids if cache.get(cid) is None)
    missing_path = args.out_dir / "missing_comments.txt"
    with open(missing_path, "w", encoding="utf-8") as f:
        f.writelines(cid + "\n" for cid in missing)
    print(f"\n{len(comment_ids) - len(missing)}/{len(comment_ids)} unique comments retrieved. "
          f"{len(missing)} unavailable (deleted, hidden or private) listed in {missing_path}")


if __name__ == "__main__":
    main()
