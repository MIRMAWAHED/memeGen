import os
import re
import json
import csv
import time
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests
import praw
from dotenv import load_dotenv

# -------------------------
# CONFIG
# -------------------------

TOTAL_TARGET = 210
COMMENTS_PER_MEME = 5
PER_CATEGORY_TARGET = 30  # 7 categories -> 210

OUTPUT_DIR = "memes_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
JSONL_PATH = os.path.join(OUTPUT_DIR, "dataset.jsonl")
CSV_PATH = os.path.join(OUTPUT_DIR, "dataset.csv")
LOG_PATH = os.path.join(OUTPUT_DIR, "run.log")

# If a subreddit is dry, we'll try more posts before giving up
POSTS_TO_SCAN_PER_SUBREDDIT = 250

# How many retries for downloading images
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 20

# Skip comments likely to be useless in a demo
SKIP_COMMENT_PATTERNS = [
    r"^(\[deleted\]|\[removed\])$",
    r"^i am a bot",
    r"^beep boop",
    r"^/r/",
    r"^this\.$",
]

# Categories and subreddits tuned for a marketing-agency demo:
# - Trend/news reaction
# - Consumer behavior relatability
# - Marketing/ads industry memes
# - Reusable templates/reaction formats
# - Brand-safe/wholesome
# - Viral/high-engagement feeders
# - Risky/controversial (for brand-safety flagging demos)
CATEGORIES: Dict[str, List[str]] = {
    "trend_news": ["PoliticalHumor", "WhitePeopleTwitter", "BlackPeopleTwitter", "nottheonion", "facepalm"],
    "consumer_behavior": ["antiwork", "mildlyinfuriating", "firstworldproblems", "me_irl", "adulting"],
    "marketing_ads": ["marketing", "advertising", "socialmedia", "Entrepreneur", "smallbusiness"],
    "templates_reaction": ["MemeTemplatesOfficial", "AdviceAnimals", "funny", "memes", "dankmemes"],
    "brand_safe": ["wholesomememes", "MadeMeSmile", "AnimalsBeingDerps", "aww", "wholesome"],
    "viral_feeders": ["memes", "dankmemes", "funny", "Unexpected", "pics"],
    "risky_controversial": ["PoliticalCompassMemes", "unpopularopinion", "cringe", "im14andthisisdeep", "insanepeoplefacebook"],
}

# For a demo, prefer "hot" and "top". You can change this.
LISTING_STRATEGY = ["hot", "top"]  # We'll mix for diversity

# -------------------------
# HELPERS
# -------------------------

def log(msg: str) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def safe_filename(s: str, max_len: int = 120) -> str:
    s = re.sub(r"[^\w\-\. ]+", "", s, flags=re.UNICODE).strip()
    s = re.sub(r"\s+", "_", s)
    return s[:max_len].strip("_") or "file"

def is_image_url(url: str) -> bool:
    url_lower = url.lower()
    # Basic direct image extensions
    if url_lower.endswith((".jpg", ".jpeg", ".png", ".gif")):
        return True
    # Some reddit previews are served as i.redd.it without explicit extension sometimes
    if "i.redd.it" in url_lower or "i.imgur.com" in url_lower:
        return True
    return False

def normalize_url_for_dedupe(url: str) -> str:
    # Remove query params/fragments for stable hashing
    return url.split("?")[0].split("#")[0].strip()

def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def comment_is_good(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    lower = t.lower()
    for pat in SKIP_COMMENT_PATTERNS:
        if re.search(pat, lower):
            return False
    # Avoid ultra-short comments that add no context
    if len(t) < 10:
        return False
    return True

def get_top_comments(submission, k: int) -> List[str]:
    """
    Pull top comments (best/most upvoted) from a submission.
    We remove "MoreComments" for a clean list.
    """
    try:
        submission.comment_sort = "top"
        submission.comments.replace_more(limit=0)
        comments = []
        for c in submission.comments.list():
            if len(comments) >= k:
                break
            if hasattr(c, "body"):
                body = c.body.strip()
                if comment_is_good(body):
                    comments.append(body)
        return comments
    except Exception as e:
        log(f"comments_failed id={submission.id} err={e}")
        return []

def download_image(url: str, out_path: str) -> bool:
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            r = requests.get(url, timeout=DOWNLOAD_TIMEOUT, headers={"User-Agent": "memeGen-demo-downloader"})
            if r.status_code == 200 and r.content:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return True
            log(f"download_bad_status attempt={attempt} status={r.status_code} url={url}")
        except Exception as e:
            log(f"download_error attempt={attempt} err={e} url={url}")
        time.sleep(1.2 * attempt)
    return False

def pick_listing(subreddit, strategy: str, limit: int):
    if strategy == "hot":
        return subreddit.hot(limit=limit)
    if strategy == "top":
        # top for the week gives better “viral-ish” content without going too old
        return subreddit.top(time_filter="week", limit=limit)
    if strategy == "new":
        return subreddit.new(limit=limit)
    return subreddit.hot(limit=limit)

@dataclass
class MemeItem:
    post_id: str
    title: str
    subreddit: str
    category: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str
    url: str
    image_path: str
    top_comments: List[str]

# -------------------------
# MAIN PIPELINE
# -------------------------

def main():
    load_dotenv()

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "memeGen by Mir (demo dataset)")

    if not client_id or not client_secret:
        raise RuntimeError("Missing REDDIT_CLIENT_ID or REDDIT_CLIENT_SECRET. Put them in .env or env vars.")

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
    )

    os.makedirs(IMAGES_DIR, exist_ok=True)

    # Prepare outputs fresh
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for p in [JSONL_PATH, CSV_PATH, LOG_PATH]:
        if os.path.exists(p):
            os.remove(p)

    # Dedupe sets
    seen_posts: Set[str] = set()
    seen_urls: Set[str] = set()

    collected: List[MemeItem] = []

    # Round-robin per category to ensure balanced dataset
    category_targets = {cat: PER_CATEGORY_TARGET for cat in CATEGORIES.keys()}
    category_counts = {cat: 0 for cat in CATEGORIES.keys()}

    # Shuffle subreddit lists for variety
    category_subs = {cat: subs[:] for cat, subs in CATEGORIES.items()}
    for cat in category_subs:
        random.shuffle(category_subs[cat])

    log("Starting dataset collection...")
    log(f"Targets: {len(CATEGORIES)} categories x {PER_CATEGORY_TARGET} = {TOTAL_TARGET} memes")
    log(f"Comments per meme: {COMMENTS_PER_MEME}")

    # Continue until all categories meet target or we can't progress
    stalled_rounds = 0
    MAX_STALLED_ROUNDS = 20

    while sum(category_counts.values()) < TOTAL_TARGET and stalled_rounds < MAX_STALLED_ROUNDS:
        progressed_this_round = False

        for category, subs in category_subs.items():
            if category_counts[category] >= category_targets[category]:
                continue  # already filled

            # Choose a subreddit from this category (rotate)
            sub_name = subs[category_counts[category] % len(subs)]
            strategy = random.choice(LISTING_STRATEGY)

            try:
                sr = reddit.subreddit(sub_name)
                posts = pick_listing(sr, strategy=strategy, limit=POSTS_TO_SCAN_PER_SUBREDDIT)
            except Exception as e:
                log(f"subreddit_failed category={category} sub={sub_name} err={e}")
                continue

            for post in posts:
                if category_counts[category] >= category_targets[category]:
                    break

                try:
                    if post.id in seen_posts:
                        continue

                    # Skip stickied or nsfw for safer demo by default
                    if getattr(post, "stickied", False):
                        continue
                    if getattr(post, "over_18", False):
                        continue

                    # We want image-like posts only
                    url = post.url
                    norm_url = normalize_url_for_dedupe(url)

                    if norm_url in seen_urls:
                        continue

                    # Skip galleries/videos
                    if getattr(post, "is_video", False):
                        continue
                    if hasattr(post, "is_gallery") and post.is_gallery:
                        continue

                    # Only accept direct-ish image links
                    if not is_image_url(url):
                        continue

                    # Pull comments first (so we don't download junk)
                    top_comments = get_top_comments(post, COMMENTS_PER_MEME)
                    if len(top_comments) < COMMENTS_PER_MEME:
                        # Not enough useful comment context -> skip for demo quality
                        continue

                    # Download image
                    base_name = safe_filename(f"{category}_{sub_name}_{post.id}")
                    # Guess extension; fallback to jpg
                    ext = ".jpg"
                    url_lower = url.lower()
                    if url_lower.endswith(".png"):
                        ext = ".png"
                    elif url_lower.endswith(".gif"):
                        ext = ".gif"
                    elif url_lower.endswith(".jpeg"):
                        ext = ".jpeg"
                    elif url_lower.endswith(".jpg"):
                        ext = ".jpg"

                    image_filename = f"{base_name}{ext}"
                    image_path = os.path.join(IMAGES_DIR, image_filename)

                    ok = download_image(url, image_path)
                    if not ok:
                        continue

                    item = MemeItem(
                        post_id=post.id,
                        title=post.title or "",
                        subreddit=sub_name,
                        category=category,
                        score=int(getattr(post, "score", 0) or 0),
                        num_comments=int(getattr(post, "num_comments", 0) or 0),
                        created_utc=float(getattr(post, "created_utc", 0.0) or 0.0),
                        permalink=f"https://www.reddit.com{post.permalink}",
                        url=url,
                        image_path=image_path,
                        top_comments=top_comments,
                    )

                    collected.append(item)
                    category_counts[category] += 1
                    seen_posts.add(post.id)
                    seen_urls.add(norm_url)
                    progressed_this_round = True

                    log(
                        f"collected {sum(category_counts.values())}/{TOTAL_TARGET} "
                        f"[{category} {category_counts[category]}/{category_targets[category]}] "
                        f"sub={sub_name} score={item.score}"
                    )

                    # Write incrementally (safe if script stops)
                    append_jsonl(item, JSONL_PATH)
                    append_csv(item, CSV_PATH)

                except Exception as e:
                    log(f"post_failed category={category} sub={sub_name} id={getattr(post,'id','?')} err={e}")
                    continue

        if not progressed_this_round:
            stalled_rounds += 1
            log(f"No progress this round. stalled_rounds={stalled_rounds}/{MAX_STALLED_ROUNDS}")
            # To improve odds, reshuffle subreddits a bit
            for cat in category_subs:
                random.shuffle(category_subs[cat])
            time.sleep(2)
        else:
            stalled_rounds = 0

    total = sum(category_counts.values())
    log("Finished.")
    log(f"Collected total: {total}/{TOTAL_TARGET}")
    log(f"Per category: {category_counts}")
    log(f"Saved: {JSONL_PATH}, {CSV_PATH}, images in {IMAGES_DIR}")

    if total < TOTAL_TARGET:
        log("WARNING: Did not reach full target. Try adding more subreddits or increasing POSTS_TO_SCAN_PER_SUBREDDIT.")

def append_jsonl(item: MemeItem, path: str) -> None:
    record = {
        "post_id": item.post_id,
        "title": item.title,
        "subreddit": item.subreddit,
        "category": item.category,
        "score": item.score,
        "num_comments": item.num_comments,
        "created_utc": item.created_utc,
        "permalink": item.permalink,
        "url": item.url,
        "image_path": item.image_path,
        "top_comments": item.top_comments,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def append_csv(item: MemeItem, path: str) -> None:
    file_exists = os.path.exists(path)
    headers = [
        "post_id",
        "title",
        "subreddit",
        "category",
        "score",
        "num_comments",
        "created_utc",
        "permalink",
        "url",
        "image_path",
        "top_comments",
    ]
    row = {
        "post_id": item.post_id,
        "title": item.title,
        "subreddit": item.subreddit,
        "category": item.category,
        "score": item.score,
        "num_comments": item.num_comments,
        "created_utc": item.created_utc,
        "permalink": item.permalink,
        "url": item.url,
        "image_path": item.image_path,
        "top_comments": " || ".join(item.top_comments),
    }
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

if __name__ == "__main__":
    main()