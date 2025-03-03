import datetime
from pathlib import Path
import requests
import json
import arxiv
import os
import time
from typing import Generator
from loguru import logger
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

API_DELAY = 4
DATA_DIR = "data"
GH_REPO = "dowwie/daily_tods"
JSON_FILE = "docs/arxiv-daily.json"
LOGS_DIR = "logs"
LAST_RUN_FILE = "docs/last_run.txt"
MD_FILE = "README.md"
CATEGORIES = [
    "cs.CL",
    "cs.LG",
    "cs.AI",
    "cs.HC",
    "cs.IR",
    "cs.SD",
    "cs.MA",
    "cs.DC",
    "cs.SI",
]


class RateLimitError(Exception):
    pass


def get_last_run_date() -> datetime.date:
    """Get the last run date or default to 7 days ago."""
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, "r") as f:
            return datetime.datetime.strptime(f.read().strip(), "%Y-%m-%d").date()
    return datetime.date.today() - datetime.timedelta(days=7)


def update_last_run_date() -> None:
    """Update the last run date."""
    with open(LAST_RUN_FILE, "w") as f:
        f.write(datetime.date.today().strftime("%Y-%m-%d"))


@retry(
    stop=stop_after_attempt(5),  # More retries
    wait=wait_exponential(multiplier=1, min=4, max=60),  # Longer backoff
    retry=retry_if_exception_type(RateLimitError),  # Retry only for 429s
)
def fetch_code_url(paper_id: str) -> str:
    try:
        response = requests.get(
            f"https://api.semanticscholar.org/v1/paper/arXiv:{paper_id}", timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return data.get("official", {}).get("url", "null")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            retry_after = e.response.headers.get("Retry-After")
            if retry_after:
                wait_time = int(retry_after)
                print(f"Rate limited for {paper_id}. Waiting {wait_time}s.")
                time.sleep(wait_time)
            raise RateLimitError("Hit 429 rate limit")
        else:
            print(f"HTTP error for {paper_id}: {e}")
            return "null"
    except requests.exceptions.RequestException as e:
        print(f"Network error for {paper_id}: {e}")
        return "null"


@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def download_pdf(pdf_url: str, title: str) -> None:
    """Download the PDF from the specified URL and save it with the given title.

    Args:
        pdf_url (str): The URL of the PDF to download.
        title (str): The title to use for naming the saved file.
    """
    try:
        # Define headers with a user-agent to mimic a legitimate client
        headers = {"User-Agent": "arxiv.py/2.1.3"}
        logger.debug(f"Attempting to download PDF from: {pdf_url}")

        # Send GET request with streaming enabled and timeout
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx, 5xx)

        # Verify that the response contains a PDF
        content_type = response.headers.get("Content-Type")
        if content_type != "application/pdf":
            logger.error(f"Unexpected content-type for {pdf_url}: {content_type}")
            return  # Skip saving if not a PDF

        # Sanitize the title for a safe filename
        sanitized_title = "".join(
            c if c.isalnum() or c in " ._-" else "_" for c in title
        )
        file_path = os.path.join(DATA_DIR, f"{sanitized_title}.pdf")

        # Write the PDF content in chunks to handle large files efficiently
        with open(file_path, "wb") as pdf_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    pdf_file.write(chunk)

        logger.info(f"PDF saved successfully at: {file_path}")

    except requests.RequestException as e:
        logger.error(f"Error downloading the PDF: {e}")
        raise


class RobustClient(arxiv.Client):
    """Custom Arxiv client that handles pagination robustly for seeding."""

    def results(
        self, search: arxiv.Search, offset: int = 0
    ) -> Generator[arxiv.Result, None, None]:
        """
        Fetch all results without crashing on empty pages, respecting max_results=None.
        """
        limit = search.max_results - offset if search.max_results else None
        if limit and limit < 0:
            return iter(())

        page_url = self._format_url(search, offset, self.page_size)
        try:
            feed = self._parse_feed(page_url, first_page=True)
        except (arxiv.HTTPError, requests.exceptions.RequestException) as e:
            logger.error(f"Failed to fetch first page: {str(e)}")
            return iter(())

        if not feed.entries:
            logger.info("Got empty first page; stopping generation")
            return

        total_results = int(feed.feed.get("opensearch_totalresults", 0))
        logger.info(
            f"Got first page: {len(feed.entries)} of {total_results} total results"
        )

        yielded = 0
        while feed.entries:
            for entry in feed.entries:
                try:
                    result = arxiv.Result._from_feed_entry(entry)
                    if (
                        result.updated.date() >= search.start_date
                    ):  # Custom filter for our use case
                        yield result
                        yielded += 1
                except arxiv.Result.MissingFieldError as e:
                    logger.warning(f"Skipping partial result: {e}")

            offset += len(feed.entries)
            if limit and yielded >= limit:
                break
            if offset >= total_results:
                logger.info(f"Reached total results: {total_results}")
                break

            page_url = self._format_url(search, offset, self.page_size)
            try:
                feed = self._parse_feed(page_url, first_page=False)
            except arxiv.UnexpectedEmptyPageError:
                logger.info(
                    f"Encountered empty page at offset {offset}; treating as end of results"
                )
                break
            except (arxiv.HTTPError, requests.exceptions.RequestException) as e:
                logger.error(
                    f"Failed to fetch page at offset {offset}: {str(e)}. Stopping with collected results."
                )
                break


def get_daily_papers(
    topic: str, query: str, last_run_date: datetime.date, seed: bool = False
) -> dict:
    """Fetch new papers since last_run_date with optional seeding."""
    start_time = time.time()
    papers = {}
    new_papers_count = 0

    # Build query with categories and date filter
    cat_query = " OR ".join(f"cat:{cat}" for cat in CATEGORIES)
    date_query = f"submittedDate:[{last_run_date.strftime('%Y%m%d')} TO {datetime.date.today().strftime('%Y%m%d')}]"
    full_query = f"({query}) AND ({cat_query}) AND {date_query}"

    # Configure robust client
    client = RobustClient(
        page_size=100,  # Fetch 100 papers per page
        delay_seconds=3.0,  # Respect API rate limit
        num_retries=5,  # More retries for robustness
    )
    search = arxiv.Search(
        query=full_query,
        max_results=None if seed else 50,  # All results for seeding, 50 for daily
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    search.start_date = last_run_date  # Custom attribute for filtering

    for result in client.results(search):
        paper_id = result.get_short_id().split("v")[0]
        update_time = result.updated.date()
        new_papers_count += 1
        title = result.title
        url = result.entry_id
        abstract = result.summary.replace("\n", " ")
        categories = ", ".join(result.categories)
        paper_repo_url = fetch_code_url(paper_id)

        pdf_url = url.replace("/abs/", "/pdf/") + ".pdf"
        try:
            download_pdf(pdf_url, title)
            time.sleep(API_DELAY)
        except Exception as e:
            logger.error(f"Failed to download PDF for {title}: {e}")

        logger.info(f"New paper: {title} ({update_time})")
        papers[paper_id] = (
            f"|**{update_time}**|**{title}**|{categories}|{abstract}|[{paper_id}]({url})|**[{paper_repo_url}]({paper_repo_url})**|\n"
            if paper_repo_url != "null"
            else f"|**{update_time}**|**{title}**|{categories}|{abstract}|[{paper_id}]({url})|null|\n"
        )

        # Incremental update for seeding
        if seed and new_papers_count % 100 == 0:
            logger.info(
                f"Processed {new_papers_count} papers; updating JSON incrementally"
            )
            update_json_file(JSON_FILE, {topic: papers})

    logger.info(f"Fetched {new_papers_count} papers in {time.time() - start_time:.2f}s")
    return {topic: papers}


def update_json_file(filename: str, new_data: dict) -> None:
    """Update JSON file, keeping all papers."""
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}

    for topic, papers in new_data.items():
        existing_data.setdefault(topic, {}).update(papers)

    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=2)
