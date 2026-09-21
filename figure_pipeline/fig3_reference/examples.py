"""Anchored English renderings of actual report excerpts for c3 and f5."""

from __future__ import annotations

from typing import Any
from pydantic import Field

from .compute import call
from .models import OutputModel
from .settings import JUDGES, OUTPUT, read, roster, write


class EnglishExamples(OutputModel):
    mention: str = Field(max_length=110)
    substantive: str = Field(max_length=110)
    supported: str = Field(max_length=110)
    target: str = Field(max_length=100)
    prior: str = Field(max_length=100)
    relation: str = Field(max_length=140)


def select() -> dict[str,Any]:
    for index,paper in enumerate(roster(),1):
        ident=paper["paper_id"]
        data=read(OUTPUT/"data/inputs"/f"{ident}.json")
        extraction=read(OUTPUT/"data/units/fusion"/f"{ident}.json")
        checks={u["unit_id"]:u for u in read(OUTPUT/"data/support/fusion"/f"{ident}.json")["units"]}
        sources={s["source_id"]:s for s in data["sources"]}
        for claim in data["claims"]:
            mentions=next(m["quotes"] for m in extraction["mentions"] if m["claim_id"]==claim["claim_id"])
            units=[u for u in extraction["units"] if claim["claim_id"] in u["claim_ids"] and u["substantive"] and not u["paraphrase"]]
            for unit in units:
                audit=checks[unit["unit_id"]]
                historical=[sources[k] for k in audit["source_ids"] if k in sources and sources[k]["source_type"] not in ("manuscript","reference_only")]
                if mentions and historical and audit["support"]=="supported" and audit["scope_correct"] and audit["nonparaphrase_insight"]:
                    return {"paper_id":ident,"method":"fusion","alias":f"P{index:03d} · {claim['claim_id'].split('::')[-1]} · Full",
                            "claim_id":claim["claim_id"],"unit_id":unit["unit_id"],"report_offset":[unit["char_start"],unit["char_end"]],
                            "mention":mentions[0],"substantive":units[0]["quote"],"supported":unit["quote"],
                            "target":claim["normalized_claim_text"],"prior":historical[0]["title"],"relation":unit["quote"],"sources":audit["source_ids"]}
    raise ValueError("No fully source-supported historical example; select another eligible method without inventing content")


def main() -> None:
    selected=select()
    prompt="Translate/summarize the provided exact Chinese report excerpts into concise English figure labels. Preserve scope, hedges and evidence limitations. Prefer 8–12 words per field, staying below the schema character cap by rewriting shorter. Every field must be a complete grammatical phrase: never cut a word or leave a trailing comma, conjunction or preposition to satisfy a length cap. Prior at most 12 words. Do not add claims or replace citations. These are explicitly labeled English renderings, not verbatim English source quotes. Return the six text fields only."
    result=call(JUDGES["extract"],prompt,selected,EnglishExamples)
    write(OUTPUT/"data/examples.json",{"original":selected,
        "commentary":{k:result[k] for k in ("mention","substantive","supported")}|{"alias":selected["alias"]},
        "insight":{k:result[k] for k in ("target","prior","relation")}|{"sources":selected["alias"]+" · "+", ".join(selected["sources"][:3])}})


if __name__=="__main__":
    main()
