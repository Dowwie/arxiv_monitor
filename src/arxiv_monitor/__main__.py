#!/usr/bin/env python3

import os
import json
import datetime
import sys
import argparse
from loguru import logger

from .arxiv_monitor import (
    get_daily_papers,
    get_last_run_date,
    update_last_run_date,
    update_json_file,
    DATA_DIR,
    JSON_FILE,
    LOGS_DIR,
    LAST_RUN_FILE,
    MD_FILE,
    GH_REPO,
)


os.makedirs(LOGS_DIR, exist_ok=True)
logger.add(
    f"{LOGS_DIR}/arxiv_tracker.log",
    format="{time} {level} {message}",
    level="DEBUG",
    rotation="1 MB",
)


def json_to_md(json_filename, md_filename):
    """Generate Markdown from JSON."""
    today = datetime.datetime.today().strftime("%Y.%m.%d")
    with open(json_filename, "r") as f:
        data = json.load(f)

    with open(md_filename, "w") as f:
        f.write(f"## Last updated on {today}\n\n")

        # About section immediately following issues shield
        f.write("\n## About\n")
        f.write(
            "This repository tracks Arxiv papers on Task Oriented Dialogue Systems.\n"
        )
        f.write(
            "- **Seeding:** Initial population covers papers from the last 5 years (run with --seed).\n"
        )
        f.write(
            "- **Daily Updates:** Adds papers since the last run (stored in last_run.txt).\n"
        )
        f.write(
            "- **Backfill:** Edit last_run.txt to an earlier date to fetch missed papers.\n\n"
        )

        for topic, papers in data.items():
            if papers:
                f.write(f"## {topic}\n\n")
                f.write("| Date | Title | Categories | Abstract | PDF | Code |\n")
                f.write("|:-----|:------|:-----------|:---------|:----|:----|\n")
                for entry in sorted(
                    papers.values(), key=lambda x: x.split("|")[1], reverse=True
                ):
                    f.write(entry)
                f.write(
                    f"<p align=right>(<a href=#Updated-on-{today.replace('.', '')}>back to top</a>)</p>\n\n"
                )

        f.write(
            f"[contributors-shield]: https://img.shields.io/github/contributors/{GH_REPO}.svg?style=for-the-badge\n"
        )
        f.write(
            f"[contributors-url]: https://github.com/{GH_REPO}/graphs/contributors\n"
        )
        f.write(
            f"[forks-shield]: https://img.shields.io/github/forks/{GH_REPO}.svg?style=for-the-badge\n"
        )
        f.write(f"[forks-url]: https://github.com/{GH_REPO}/network/members\n")
        f.write(
            f"[stars-shield]: https://img.shields.io/github/stars/{GH_REPO}.svg?style=for-the-badge\n"
        )
        f.write(f"[stars-url]: https://github.com/{GH_REPO}/stargazers\n")
        f.write(
            f"[issues-shield]: https://img.shields.io/github/issues/{GH_REPO}.svg?style=for-the-badge\n"
        )
        f.write(f"[issues-url]: https://github.com/{GH_REPO}/issues\n")

    logger.info("Markdown generated")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Monitor Arxiv for task-oriented dialogue papers."
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed the repository with 5 years of data (run once, then omit)",
    )
    args = parser.parse_args()

    logger.info("Starting Arxiv tracking")
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.seed:
        last_run_date = datetime.date.today() - datetime.timedelta(days=1825)  # 5 years
        logger.info(f"Seeding with papers since {last_run_date}")
    else:
        last_run_date = get_last_run_date()
        logger.info(f"Updating with papers since {last_run_date}")

    topic = "Task Oriented Dialogue Systems"
    query = 'all:"task oriented dialogue" OR all:"task oriented dialog" OR all:"TOD system" OR all:"task-oriented dialog"'

    new_data = get_daily_papers(topic, query, last_run_date, seed=args.seed)
    update_json_file(JSON_FILE, new_data)
    json_to_md(JSON_FILE, MD_FILE)
    update_last_run_date()

    logger.info("Process completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
