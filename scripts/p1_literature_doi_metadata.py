#!/usr/bin/env python3

from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

import csv
import json


PROJECT = Path(__file__).resolve().parents[1]

OUT = (
    PROJECT
    / "docs/literature/p1_doi_metadata.csv"
)


DOIS = [
    (
        "Window length AD 2019",
        "10.3390/brainsci9040081"
    ),
    (
        "EEG duration MCI",
        "10.1177/1550059420983624"
    ),
    (
        "EEG recommendations",
        "10.1002/alz.12311"
    ),
    (
        "EEG dataset descriptor",
        "10.3390/data8060095"
    ),
    (
        "External EEG cohort",
        "10.1007/s11517-021-02427-6"
    ),
    (
        "BrainLat data descriptor",
        "10.1038/s41597-023-02806-8"
    ),
    (
        "External study 2026",
        "10.1038/s41598-026-57069-1"
    ),
    (
        "EEG dementia study 2026",
        "10.1186/s12984-026-01897-9"
    ),
    (
        "Duration CNN study",
        "10.1186/s40537-025-01107-0"
    )
]


def lookup(label, doi):

    result = {
        "screen_label": label,
        "doi": doi,
        "status": "UNVERIFIED",
        "title": "",
        "year": "",
        "journal": "",
        "url": "",
        "error": ""
    }

    try:

        url = (
            "https://api.crossref.org/works/"
            + quote(doi, safe="")
        )

        request = Request(
            url,
            headers={
                "User-Agent":
                    "AlzheimersThesisResearch/1.0"
            }
        )

        with urlopen(
            request,
            timeout=22
        ) as response:

            item = (
                json.load(response)["message"]
            )

        dates = (
            item.get("published")
            or item.get("published-online")
            or item.get("issued")
            or {}
        )

        year = (
            dates.get("date-parts")
            or [[None]]
        )[0][0]

        result.update({
            "status": "DOI_METADATA_RETRIEVED",

            "title":
                (item.get("title") or [""])[0],

            "year":
                year,

            "journal":
                (
                    item.get("container-title")
                    or [""]
                )[0],

            "url":
                item.get("URL", "")
        })

    except Exception as exc:

        result["error"] = str(exc)[:300]

    return result


def main():

    rows = [
        lookup(label, doi)
        for label, doi in DOIS
    ]

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUT.open(
        "w",
        newline=""
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "screen_label",
                "doi",
                "status",
                "title",
                "year",
                "journal",
                "url",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "===== PUBLICATION METADATA CHECK ====="
    )

    print(
        "DOIs screened:",
        len(rows)
    )

    print(
        "Metadata retrieved:",
        sum(
            row["status"] == "DOI_METADATA_RETRIEVED"
            for row in rows
        )
    )

    for row in rows:

        print(
            row["status"],
            row["doi"],
            str(row["title"])[:100]
        )

    print()
    print(
        "NOTE: DOI metadata alone cannot prove novelty."
    )


if __name__ == "__main__":
    main()
