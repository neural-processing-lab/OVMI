# OVMI project website

This directory is a dependency-free static site deployed at:

<https://neural-processing-lab.github.io/OVMI/>

## Preview locally

From the repository root:

```bash
python -m http.server 8000 --directory site
```

Then open <http://localhost:8000/>. A local server is required because browsers
do not allow `fetch()` of `data/leaderboard.json` from a `file://` page.

## Data

The browser reads `site/data/leaderboard.json`. Regenerate it from the paper
analysis pipeline with:

```bash
python scripts/build_site_data.py
```

See `site/data/README.md` for the schema and provenance.

## External links

Repository, package, issue, paper, and blog URLs are centralised in
`site/config.js`. The paper and blog entries link to their public publications.
Keep the corresponding fallback links in `site/index.html` in sync.

## GitHub Pages

`.github/workflows/pages.yml` uploads this directory and deploys it with the
official Pages actions. In repository settings, choose **GitHub Actions** as the
Pages build and deployment source. The workflow can then run on a push to
`main` that changes `site/`, or manually through **Actions → Deploy project
website → Run workflow**.
