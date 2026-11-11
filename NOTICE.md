# Sources and attribution

This prototype indexes third-party legal material. Attribution and terms:

## Consumer Protection Act, 2019 (Act 35 of 2019)
Central legislation of India. Statutory text is not subject to copyright under
s.52(1)(q) of the Copyright Act, 1957.

The authoritative publisher is **India Code** (indiacode.nic.in), maintained by
the Legislative Department, Ministry of Law and Justice. `scripts/01_statutes.py`
attempts India Code first on every run.

**At build time India Code's document endpoints returned HTTP 504** (origin
timeout; the site root responds but `/handle/...` does not). The vendored copy
in `data/raw/cpa2019_indiankanoon.html` was therefore retrieved from
**Indian Kanoon** (https://indiankanoon.org/doc/48103131/), which publishes the
Act in Akoma Ntoso markup. One document, fetched once.

## Consumer commission decisions
Retrieved from **Indian Kanoon** (https://indiankanoon.org), filtered to
`doctypes:ncdrc`. Each stored judgment keeps its source URL and is linked back
to indiankanoon.org in every answer that relies on it.

Crawl conduct:
- `robots.txt` is fetched and parsed on every run; the per-document denylist it
  publishes (de-indexed / right-to-be-forgotten documents) is honoured, and the
  script aborts rather than proceeding if robots.txt cannot be read.
- 1 request/second, single-threaded.
- Descriptive User-Agent identifying the client as a coursework prototype.
- Hard cap on document count; responses cached on disk so nothing is re-fetched.
- Documents are used for retrieval and quotation with attribution, and are not
  redistributed as a dataset.

Indian Kanoon sells an API for programmatic access. This prototype does not use
it: the brief states that free tiers are sufficient and that paid tools earn no
extra credit. At production volume the API would be the correct and fair route,
and `src/retrieve.py` would be unaffected — only `scripts/02_fetch_cases.py`
would change.

Indian Kanoon disclaims warranties as to the accuracy of its data. So does this
prototype. Nothing here is legal advice.

## Supreme Court judgments (evaluated, not used)
`s3://indian-supreme-court-judgments` (AWS Open Data, CC-BY-4.0) was measured
and rejected for this domain: 1 consumer matter in 856 judgments for 2023.
