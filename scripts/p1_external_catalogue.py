#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from datetime import datetime, timezone

import json


PROJECT = Path(__file__).resolve().parents[1]

OUT = (
    PROJECT
    / "docs/audit/p1_external_dataset_metadata.json"
)


SOURCES = [
    {
        "name": "Cejnek_Figshare",
        "url":
            "https://api.figshare.com/v2/articles/5450293",
        "reported":
            "Potential continuous independent EEG dataset; "
            "participants, labels and durations require verification."
    },
    {
        "name": "ADFSU_OSF_node",
        "url":
            "https://api.osf.io/v2/nodes/jbysn/",
        "reported":
            "External EEG benchmark; published descriptions "
            "mention short segments."
    },
    {
        "name": "ADFSU_OSF_files",
        "url":
            "https://api.osf.io/v2/nodes/jbysn/files/",
        "reported":
            "Metadata-only inspection of OSF file providers."
    },
    {
        "name": "BrainLat_Synapse",
        "url":
            "https://repo-prod.prod.sagebase.org/repo/v1/entity/syn51549340",
        "reported":
            "Independent EEG candidate; registration or "
            "data-use authorization may be required."
    }
]


def fetch(source):

    record = {
        "name": source["name"],
        "url": source["url"],
        "reported_context_not_verified":
            source["reported"]
    }

    try:

        request = Request(
            source["url"],
            headers={
                "Accept": "application/json",
                "User-Agent": "research-feasibility/1.0"
            }
        )

        with urlopen(
            request,
            timeout=25
        ) as response:

            status = getattr(
                response,
                "status",
                200
            )

            payload = response.read(
                2 * 1024 * 1024
            )

        item = json.loads(
            payload
        )

        record.update({
            "api_status": status,
            "metadata_retrievable": True
        })

        if source["name"] == "Cejnek_Figshare":

            files = item.get(
                "files",
                []
            )

            record.update({
                "title": item.get("title"),
                "doi": item.get("doi"),
                "listed_files_count": len(files),
                "listed_files": [
                    {
                        "name": file.get("name"),
                        "size": file.get("size"),
                        "id": file.get("id")
                    }
                    for file in files[:30]
                ],
                "license": item.get("license")
            })

        elif source["name"] == "ADFSU_OSF_node":

            attributes = (
                item
                .get("data", {})
                .get("attributes", {})
            )

            record.update({
                "title": attributes.get("title"),
                "public": attributes.get("public"),
                "description_excerpt":
                    (
                        attributes.get("description")
                        or ""
                    )[:1000]
            })

        elif source["name"] == "ADFSU_OSF_files":

            providers = item.get(
                "data",
                []
            )

            record["listed_providers"] = [
                {
                    "name":
                        provider.get(
                            "attributes",
                            {}
                        ).get("name"),
                    "id":
                        provider.get("id")
                }
                for provider in providers[:20]
            ]

        elif source["name"] == "BrainLat_Synapse":

            record.update({
                "name_from_api":
                    item.get("name"),

                "entity_type":
                    item.get("concreteType"),

                "access_requirements":
                    "Requires separate access and license review"
            })

    except HTTPError as exc:

        record.update({
            "api_status": exc.code,
            "metadata_retrievable": False,
            "error": str(exc)[:250]
        })

    except Exception as exc:

        record.update({
            "api_status": None,
            "metadata_retrievable": False,
            "error": str(exc)[:250]
        })

    record["research_eligibility"] = (
        "UNVERIFIED: raw signal, participant identity, "
        "duration, diagnostic labels, license and "
        "cohort independence must be checked."
    )

    return record


def main():

    entries = [
        fetch(source)
        for source in SOURCES
    ]

    report = {
        "run_utc":
            datetime.now(timezone.utc).isoformat(),

        "scope":
            "Public metadata-only research feasibility",

        "candidates":
            entries
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    OUT.write_text(
        json.dumps(
            report,
            indent=2,
            default=str
        ) + "\n"
    )

    print(
        "===== EXTERNAL DATASET METADATA ====="
    )

    for record in entries:

        print()
        print(
            record["name"],
            "HTTP:", record["api_status"],
            "retrievable:",
            record["metadata_retrievable"]
        )

        if record.get("title"):
            print(
                "Title:",
                record["title"][:130]
            )

        if "listed_files_count" in record:

            print(
                "Listed files:",
                record["listed_files_count"]
            )

            print(
                "First five:",
                [
                    file.get("name")
                    for file in record["listed_files"][:5]
                ]
            )

        if not record["metadata_retrievable"]:
            print(
                "Blocker:",
                record.get("error")
            )

    print()
    print(
        "EXTERNAL COHORT STATUS: NONE YET APPROVED"
    )


if __name__ == "__main__":
    main()
