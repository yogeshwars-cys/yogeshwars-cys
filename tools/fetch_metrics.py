"""Snapshot public GitHub stats into tools/metrics.json (needs an authenticated `gh`).

    python tools/fetch_metrics.py && python tools/build.py
"""

import datetime as dt
import json
import subprocess
from collections import Counter
from pathlib import Path

USER = "yogeshwars-cys"
OUT = Path(__file__).resolve().parent / "metrics.json"
# kept out of the language stats: the profile repo itself, and nitrostack (third-party code, not mine)
EXCLUDE = {USER, "nitrostack"}

CALENDAR = """{ viewer { contributionsCollection {
  totalCommitContributions totalPullRequestContributions
  contributionCalendar { totalContributions weeks { firstDay contributionDays { contributionCount } } }
} } }"""


def gh(*args: str):
    return json.loads(subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout)


def main() -> None:
    cc = gh("api", "graphql", "-f", f"query={CALENDAR}")["data"]["viewer"]["contributionsCollection"]
    weeks = [
        {"start": w["firstDay"], "count": sum(d["contributionCount"] for d in w["contributionDays"])}
        for w in cc["contributionCalendar"]["weeks"]
    ]

    repos = gh("repo", "list", USER, "--limit", "200", "--json", "name,isPrivate,isFork")
    public = [r["name"] for r in repos if not r["isPrivate"] and not r["isFork"] and r["name"] != USER]

    langs: Counter = Counter()
    for name in public:
        if name not in EXCLUDE:
            langs.update(gh("api", f"repos/{USER}/{name}/languages"))

    OUT.write_text(json.dumps({
        "as_of": dt.date.today().isoformat(),
        "contributions": cc["contributionCalendar"]["totalContributions"],
        "commits": cc["totalCommitContributions"],
        "pull_requests": cc["totalPullRequestContributions"],
        "public_repos": len(public),
        "excluded": sorted(EXCLUDE - {USER}),
        "weeks": weeks,
        "languages": dict(langs.most_common()),
    }, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
