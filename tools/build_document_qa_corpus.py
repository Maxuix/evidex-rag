#!/usr/bin/env python3
"""Build and validate the bounded public document-QA evaluation corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable
from urllib.request import Request, urlopen
import zipfile

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "document-qa-v1"
CORPUS_SCHEMA = "document_qa_corpus_v1"
CASE_SCHEMA = "document_qa_case_v1"
COMPLEX_CASE_SCHEMA = "document_qa_complex_case_v1"
COMPLEX_CASES_FILENAME = "complex-cases.jsonl"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    filename: str
    url: str
    sha256: str
    dataset: str
    revision: str | None


FINANCEBENCH_REVISION = "cc39aeb4afdf33909ee1412188bf89035950c2eb"
TATQA_REVISION = "870accc41953dcde885aabeb963d94aabdc0fbc3"
CONTRACTNLI_REVISION = "eced6528dd3c1d14d73f9a87df8f7bdbc03126f9"
CFQA_REVISION = "61c9ec3c4335d0411a1735cd228af8b3ead114fc"


def _github_raw(repository: str, revision: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"


SOURCE_SPECS = (
    SourceSpec(
        key="financebench_cases",
        filename="financebench_open_source.jsonl",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "data/financebench_open_source.jsonl",
        ),
        sha256="a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_documents",
        filename="financebench_document_information.jsonl",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "data/financebench_document_information.jsonl",
        ),
        sha256="1c69127783879de8cdadb159d2181f39bc3123b8e0ebf74031c3969d69189575",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_amd_pdf",
        filename="AMD_2022_10K.pdf",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "pdfs/AMD_2022_10K.pdf",
        ),
        sha256="a3bd74088fae0ad4aa03d04998a1f4c64fd0b3c693841601b0ac49b46cf5c1f4",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_amex_pdf",
        filename="AMERICANEXPRESS_2022_10K.pdf",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "pdfs/AMERICANEXPRESS_2022_10K.pdf",
        ),
        sha256="d3bbc7ab23d6160e07eab50a359e4d68c40efff9fde3baa168e7c40e06f568e6",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="financebench_boeing_pdf",
        filename="BOEING_2022_10K.pdf",
        url=_github_raw(
            "patronus-ai/financebench",
            FINANCEBENCH_REVISION,
            "pdfs/BOEING_2022_10K.pdf",
        ),
        sha256="09285ff7ee737d3302977104aa05cc53c9dfb0161f3461a89b583dc38b9f2ab6",
        dataset="FinanceBench",
        revision=FINANCEBENCH_REVISION,
    ),
    SourceSpec(
        key="tatqa_dev",
        filename="tatqa_dataset_dev.json",
        url=_github_raw(
            "NExTplusplus/TAT-QA",
            TATQA_REVISION,
            "dataset_raw/tatqa_dataset_dev.json",
        ),
        sha256="8da095a819af6db3c14877c6df2d4d29960e41d1a63dd1fa853507bd2a616af5",
        dataset="TAT-QA",
        revision=TATQA_REVISION,
    ),
    SourceSpec(
        key="contractnli_zip",
        filename="contract-nli.zip",
        url=_github_raw(
            "stanfordnlp/contract-nli",
            CONTRACTNLI_REVISION,
            "resources/contract-nli.zip",
        ),
        sha256="e03fc77bbf8b53e2976a250e81d8a294bc3d5e5fb014521e477dee9340d6287b",
        dataset="ContractNLI",
        revision=CONTRACTNLI_REVISION,
    ),
    SourceSpec(
        key="cfqa_cases",
        filename="cfqa_split_by_company_test.json",
        url=_github_raw(
            "ygan/CFQA",
            CFQA_REVISION,
            "dataset/split_by_company/split_by_company_test.json",
        ),
        sha256="e4c08332a1f6aada430ac94fe62106fd557c3d80cd68c495118412bb7704bc7d",
        dataset="CFQA",
        revision=CFQA_REVISION,
    ),
    SourceSpec(
        key="cfqa_fenghuo_pdf",
        filename="fenghuo-electronics-2022-annual-report.pdf",
        url="https://static.cninfo.com.cn/finalpage/2023-04-12/1216382408.PDF",
        sha256="be9bef33ee8d5df5cd74c14bd01c089a5a583cd2b2a436ce1edf2754623cfdec",
        dataset="CFQA / CNINFO",
        revision=None,
    ),
)

SOURCE_BY_KEY = {source.key: source for source in SOURCE_SPECS}

FINANCE_DOCUMENTS = (
    (
        "AMD_2022_10K",
        "financebench-amd-2022-10k",
        "financebench_amd_pdf",
    ),
    (
        "AMERICANEXPRESS_2022_10K",
        "financebench-american-express-2022-10k",
        "financebench_amex_pdf",
    ),
    (
        "BOEING_2022_10K",
        "financebench-boeing-2022-10k",
        "financebench_boeing_pdf",
    ),
)

TATQA_DOCUMENTS = (
    (
        "3ffd9053-a45d-491c-957a-1b2fa0af0570",
        "tatqa-sales-by-contract-type",
        "Sales by Contract Type",
    ),
    (
        "53474060-2736-46cb-bd97-1eb42f0ff3c1",
        "tatqa-net-sales-by-end-market",
        "Net Sales by Segment and Industry End Market",
    ),
    (
        "285a1ced-709e-4f45-a227-b6cd04e725f9",
        "tatqa-other-operating-expenses",
        "Other Operating Expenses",
    ),
    (
        "ba26cd64-e448-4ffb-bfaa-c6ad4760fba7",
        "tatqa-loan-to-value",
        "Loan-to-Value Ratio",
    ),
)

CONTRACT_DOCUMENTS = (
    (
        488,
        "contractnli-sec-text-488",
        ("nda-11", "nda-2", "nda-16"),
    ),
    (
        15,
        "contractnli-pdf-15",
        ("nda-15", "nda-17", "nda-11"),
    ),
    (
        547,
        "contractnli-sec-html-547",
        ("nda-10", "nda-2", "nda-18"),
    ),
    (
        82,
        "contractnli-pdf-82",
        ("nda-19", "nda-20", "nda-8"),
    ),
)

CFQA_CASE_IDS = frozenset({73, 77, 81, 85, 89, 93, 97, 101})


def _complex_source(
    case_id: str,
    *,
    locator: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "source_case_id": case_id,
        "evidence_locator": locator or {"kind": "base_case_evidence"},
    }


def _complex_aspect(
    aspect_id: str,
    answer_variants: tuple[str, ...],
    sources: tuple[dict[str, object], ...],
    *,
    expected_decimal: str | None = None,
    numeric_tolerance: str | None = None,
    allow_not_mentioned: bool = False,
    requires_complete_scan: bool = False,
    answer_match: str = "all",
) -> dict[str, object]:
    value: dict[str, object] = {
        "aspect_id": aspect_id,
        "answer_variants": list(answer_variants),
        "source": list(sources),
        "allow_not_mentioned": allow_not_mentioned,
        "requires_complete_scan": requires_complete_scan,
        "answer_match": answer_match,
    }
    if expected_decimal is not None:
        value["expected_decimal"] = expected_decimal
        value["numeric_tolerance"] = numeric_tolerance or "0.01"
    else:
        value["expected_decimal"] = None
        value["numeric_tolerance"] = None
    return value


def _complex_case(
    case_id: str,
    question: str,
    *,
    language: str,
    reasoning_type: str,
    required_document_ids: tuple[str, ...],
    aspects: tuple[dict[str, object], ...],
    evaluation_group: str = "evidence_only",
    requires_complete_scan: bool = False,
    allow_not_mentioned: bool = False,
    notes: str,
) -> dict[str, object]:
    return {
        "schema_version": COMPLEX_CASE_SCHEMA,
        "case_id": case_id,
        "question": question,
        "language": language,
        "reasoning_type": reasoning_type,
        "required_document_ids": list(required_document_ids),
        "required_citation_document_ids": list(required_document_ids),
        "forbid_unrelated_citations": True,
        "aspects": list(aspects),
        "evaluation_group": evaluation_group,
        "requires_complete_scan": requires_complete_scan,
        "allow_not_mentioned": allow_not_mentioned,
        "notes": notes,
    }


# These are intentionally hand-composed from the pinned base cases above.  The
# builder copies them into an ignored JSONL artifact and the validator checks
# every source case and locator against the generated corpus.  Keeping the
# definitions in executable code makes the benchmark reproducible without
# storing third-party documents in Git.
COMPLEX_CASE_DEFINITIONS = (
    _complex_case(
        "complex-01",
        (
            "Using financebench-amd-2022-10k.pdf and "
            "financebench-boeing-2022-10k.pdf, compare the FY2022 "
            "customer-concentration evidence. Report AMD's single-customer "
            "percentage and Boeing's U.S.-government-contract percentage, state "
            "the numerical difference, explain why the two percentages are not "
            "directly comparable measures of customer dependence, and mention "
            "Boeing's cyclicality."
        ),
        language="en",
        reasoning_type="cross_document_comparison",
        required_document_ids=(
            "financebench-amd-2022-10k",
            "financebench-boeing-2022-10k",
        ),
        aspects=(
            _complex_aspect(
                "amd_single_customer_share",
                ("one customer",),
                (_complex_source("financebench_id_00757"),),
                expected_decimal="16",
            ),
            _complex_aspect(
                "boeing_government_contract_share",
                ("U.S. government",),
                (_complex_source("financebench_id_01290"),),
                expected_decimal="40",
            ),
            _complex_aspect(
                "comparison_difference_and_scope",
                ("not directly comparable",),
                (
                    _complex_source("financebench_id_00757"),
                    _complex_source("financebench_id_01290"),
                ),
                expected_decimal="24",
            ),
            _complex_aspect(
                "boeing_cyclicality",
                ("cyclicality", "cyclical"),
                (_complex_source("financebench_id_00464"),),
                answer_match="any",
            ),
        ),
        notes="The two percentages use different business definitions; only the arithmetic difference is comparable.",
    ),
    _complex_case(
        "complex-02",
        (
            "Using financebench-american-express-2022-10k.pdf and "
            "financebench-boeing-2022-10k.pdf, report "
            "the issuer-disclosed effective tax rates for 2021 and 2022 for "
            "each company, calculate each year-over-year change in percentage "
            "points, and state which company's change in magnitude was larger."
        ),
        language="en",
        reasoning_type="cross_document_financial_calculation",
        required_document_ids=(
            "financebench-american-express-2022-10k",
            "financebench-boeing-2022-10k",
        ),
        aspects=(
            _complex_aspect(
                "amex_effective_tax_rate_change",
                ("21.6%", "24.6%", "decreas"),
                (_complex_source("financebench_id_01351"),),
                expected_decimal="-3.0",
            ),
            _complex_aspect(
                "boeing_effective_tax_rate_change",
                ("-0.6%", "14.8%", "decreas"),
                (
                    _complex_source(
                        "financebench_id_00585",
                        locator={"kind": "pdf_page", "page": 77},
                    ),
                ),
                expected_decimal="-15.4",
            ),
            _complex_aspect(
                "larger_change_magnitude",
                ("Boeing",),
                (
                    _complex_source("financebench_id_01351"),
                    _complex_source(
                        "financebench_id_00585",
                        locator={"kind": "pdf_page", "page": 77},
                    ),
                ),
            ),
        ),
        notes="Use the report's effective-tax-rate reconciliation rows; do not score the conflicting FinanceBench derived answer as gold.",
    ),
    _complex_case(
        "complex-03",
        (
            "Using financebench-american-express-2022-10k.pdf and "
            "financebench-boeing-2022-10k.pdf, assess "
            "whether gross margin is a comparable performance measure. State "
            "what the AmEx report says about gross margin and calculate Boeing's "
            "2021 and 2022 gross margins from gross profit divided by revenue, "
            "clearly separating report facts from the limited inference."
        ),
        language="en",
        reasoning_type="cross_document_metric_comparability",
        required_document_ids=(
            "financebench-american-express-2022-10k",
            "financebench-boeing-2022-10k",
        ),
        evaluation_group="domain_inference",
        aspects=(
            _complex_aspect(
                "amex_gross_margin_not_used",
                ("not measured through gross margin",),
                (_complex_source("financebench_id_00720"),),
            ),
            _complex_aspect(
                "boeing_gross_profit",
                ("3,502", "3,017", "gross profit"),
                (_complex_source("financebench_id_00678"),),
                expected_decimal="3502",
            ),
            _complex_aspect(
                "boeing_gross_margin_computation",
                ("4.8%", "5.3%", "improving"),
                (_complex_source("financebench_id_00678"),),
                expected_decimal="5.3",
            ),
        ),
        notes="The gross-margin comparability conclusion is a bounded domain inference, not a directly stated AmEx fact.",
    ),
    _complex_case(
        "complex-04",
        (
            "Using financebench-american-express-2022-10k.pdf and "
            "cfqa-fenghuo-electronics-2022-annual-report.pdf, list AmEx's reported "
            "geographies and state exactly what the Fenghuo source discloses "
            "about overseas investment or foreign assets. Do not turn an "
            "unavailable disclosure into a stronger conclusion."
        ),
        language="en",
        reasoning_type="cross_document_scope_and_qualification",
        required_document_ids=(
            "financebench-american-express-2022-10k",
            "cfqa-fenghuo-electronics-2022-annual-report",
        ),
        aspects=(
            _complex_aspect(
                "amex_geographies",
                ("United States", "EMEA", "APAC", "LACC"),
                (_complex_source("financebench_id_01028"),),
            ),
            _complex_aspect(
                "fenghuo_overseas_disclosure",
                ("境外资产占比较高", "不适用"),
                (_complex_source("cfqa-85"),),
            ),
        ),
        notes="The source wording is intentionally qualified; benchmark scoring rejects an unsupported 'no overseas investment' assertion.",
    ),
    _complex_case(
        "complex-05",
        (
            "Using cfqa-fenghuo-electronics-2022-annual-report.pdf, compare the "
            "raw-material cost (原材料) with the management expense (管理费用) and "
            "R&D expense (研发费用) rows in the consolidated income statement "
            "(合并利润表). Report the two amounts and the exact difference, retaining "
            "cents."
        ),
        language="en",
        reasoning_type="financial_decimal_calculation",
        required_document_ids=("cfqa-fenghuo-electronics-2022-annual-report",),
        aspects=(
            _complex_aspect(
                "raw_material_cost",
                ("785,646,432.47",),
                (_complex_source("cfqa-101"),),
                expected_decimal="785646432.47",
            ),
            _complex_aspect(
                "management_plus_rd",
                ("491,210,498.01", "229,129,291.07", "262,081,206.94"),
                (_complex_source("cfqa-81"),),
                expected_decimal="491210498.01",
            ),
            _complex_aspect(
                "raw_material_difference",
                ("294,435,934.46",),
                (_complex_source("cfqa-101"), _complex_source("cfqa-81")),
                expected_decimal="294435934.46",
            ),
        ),
        notes="The two derived values are gold facts for the evaluator and must be produced with Decimal arithmetic.",
    ),
    _complex_case(
        "complex-06",
        (
            "For AMD's FY2022 report, excluding Embedded, identify the segment "
            "with the largest proportional sales increase, quantify it, name "
            "the main revenue drivers, and explain the Xilinx-related pressure "
            "on operating income."
        ),
        language="en",
        reasoning_type="multi_aspect_financial_summary",
        required_document_ids=("financebench-amd-2022-10k",),
        aspects=(
            _complex_aspect(
                "fastest_non_embedded_segment",
                ("Data Center", "64%"),
                (_complex_source("financebench_id_00563"), _complex_source("financebench_id_01198")),
                expected_decimal="64",
            ),
            _complex_aspect(
                "amd_revenue_drivers",
                ("EPYC", "semi-custom", "Xilinx", "embedded"),
                (_complex_source("financebench_id_01198"),),
            ),
            _complex_aspect(
                "xilinx_operating_income_pressure",
                ("amortization", "intangible assets", "Xilinx acquisition"),
                (_complex_source("financebench_id_00917"),),
            ),
        ),
        notes="All three aspects are directly grounded in the AMD filing.",
    ),
    _complex_case(
        "complex-07",
        (
            "Using the Sales by Contract Type table, calculate Other as a "
            "percentage of total sales in 2019 and explain how the change in "
            "Fixed Price versus Other still leaves total sales at the three-year "
            "high."
        ),
        language="en",
        reasoning_type="table_calculation_and_explanation",
        required_document_ids=("tatqa-sales-by-contract-type",),
        aspects=(
            _complex_aspect(
                "other_share_of_total_sales",
                ("Other",),
                (
                    _complex_source("tatqa-4960801d-277d-4f79-8eca-c4d0200fa9d6"),
                    _complex_source("tatqa-05b670d3-5b19-438c-873f-9bf6de29c69e"),
                ),
                expected_decimal="2.95",
            ),
            _complex_aspect(
                "fixed_price_offsets_other",
                ("Fixed Price", "1,146.2", "1,452.4", "total sales", "highest"),
                (
                    _complex_source("tatqa-4960801d-277d-4f79-8eca-c4d0200fa9d6"),
                    _complex_source("tatqa-eb787966-fa02-401f-bfaf-ccabf3828b23"),
                ),
            ),
        ),
        notes="The percentage is an exact Decimal calculation from the table values.",
    ),
    _complex_case(
        "complex-08",
        (
            "Using the Other Operating Expenses table, calculate the residual "
            "amount after subtracting the 2019 impairment charges and net losses "
            "on disposals from the 2019 total, and identify how many 2019 expense "
            "segments exceed $50 million."
        ),
        language="en",
        reasoning_type="table_calculation_and_threshold",
        required_document_ids=("tatqa-other-operating-expenses",),
        aspects=(
            _complex_aspect(
                "other_operating_expense_residual",
                ("94.2",),
                (
                    _complex_source("tatqa-3d384cee-82de-48f1-98ff-a972404bce4c"),
                    _complex_source("tatqa-a5992e2e-726e-469c-88f3-e7b2ea8db24c"),
                    _complex_source("tatqa-ce6dd8c2-37c9-4c55-80d5-89ab35ae254f"),
                ),
                expected_decimal="94.2",
            ),
            _complex_aspect(
                "segments_above_fifty",
                ("$50 million",),
                (_complex_source("tatqa-3d384cee-82de-48f1-98ff-a972404bce4c"),),
                expected_decimal="1",
            ),
        ),
        notes="The residual is 166.3 - 45.1 - 27.0; it is not a new source fact.",
    ),
    _complex_case(
        "complex-09",
        (
            "Across the three named tables, rank the percentage declines from "
            "largest to smallest: Total Other operating expenses, contract-type "
            "Other, and Appliances. Include each percentage."
        ),
        language="en",
        reasoning_type="cross_document_numeric_ranking",
        required_document_ids=(
            "tatqa-other-operating-expenses",
            "tatqa-sales-by-contract-type",
            "tatqa-net-sales-by-end-market",
        ),
        aspects=(
            _complex_aspect(
                "decline_ranking",
                ("Total Other operating expenses", "67.60%", "Other", "22.22%", "Appliances", "12.14%"),
                (
                    _complex_source("tatqa-58adf6c4-41ae-4f3f-84cb-cf3469a80ce4"),
                    _complex_source("tatqa-05b670d3-5b19-438c-873f-9bf6de29c69e"),
                    _complex_source("tatqa-fe11f001-3bfe-4089-8108-412676f0a780"),
                ),
                expected_decimal="67.60",
            ),
        ),
        notes="Report absolute decline magnitudes in the requested order; retain the source table labels.",
    ),
    _complex_case(
        "complex-10",
        (
            "For the named TORM Loan-to-Value Ratio document, identify the year "
            "with the highest table LTV, calculate the 2019 change in Total (loan) "
            "from 2018 in amount and percent, and flag the conflict between the "
            "written LTV definition and the table's displayed ratio."
        ),
        language="en",
        reasoning_type="table_definition_conflict_and_calculation",
        required_document_ids=("tatqa-loan-to-value",),
        aspects=(
            _complex_aspect(
                "highest_ltv_year",
                ("2017", "55.8%"),
                (_complex_source("tatqa-4048700c-347c-4cef-b3dd-c75e903a29d2"),),
                expected_decimal="55.8",
            ),
            _complex_aspect(
                "total_loan_change",
                ("885.3", "828.8", "56.5", "6.38%"),
                (
                    _complex_source("tatqa-4a253bf4-b9fc-4ec3-b99b-d2744646a296"),
                    _complex_source("tatqa-74183d5d-7633-43b5-bd7b-f0894fbe2299"),
                ),
                expected_decimal="-56.5",
            ),
            _complex_aspect(
                "ltv_definition_conflict",
                ("Vessel values divided by net borrowings", "table", "opposite"),
                (
                    _complex_source("tatqa-8d8801d9-ea00-4bf9-ac9c-825d05ac88fd"),
                    _complex_source("tatqa-01dedbe0-7d7e-4db1-ac51-eb84b4eb7d98"),
                ),
            ),
        ),
        notes="The table's 46.0% equals Total (loan) / Total (value), contrary to the prose definition.",
    ),
    _complex_case(
        "complex-11",
        (
            "Compare the reverse-engineering statement separately in "
            "contractnli-pdf-15.txt and contractnli-sec-text-488.txt. State the "
            "NLI label for each named contract and do not treat the absence in "
            "one contract as evidence from the other."
        ),
        language="en",
        reasoning_type="scoped_contract_nli_comparison",
        required_document_ids=("contractnli-pdf-15", "contractnli-sec-text-488"),
        requires_complete_scan=True,
        allow_not_mentioned=True,
        aspects=(
            _complex_aspect(
                "contract_15_reverse_engineering",
                ("contractnli-pdf-15.txt", "not mentioned"),
                (_complex_source("contractnli-15-nda-11"),),
                allow_not_mentioned=True,
                requires_complete_scan=True,
                answer_match="all",
            ),
            _complex_aspect(
                "contract_488_reverse_engineering",
                ("contractnli-sec-text-488.txt", "entailment"),
                (_complex_source("contractnli-488-nda-11"),),
            ),
        ),
        notes="The not-mentioned label is valid only with a complete serving-document scan.",
    ),
    _complex_case(
        "complex-12",
        (
            "For contractnli-pdf-82.txt only, classify these three statements in "
            "order: obligations may survive termination; the recipient may retain "
            "confidential information after return or destruction; and the "
            "recipient must notify the disclosing party when disclosure is legally "
            "required. Use not_mentioned only after the complete document scan."
        ),
        language="en",
        reasoning_type="scoped_contract_nli_three_way",
        required_document_ids=("contractnli-pdf-82",),
        requires_complete_scan=True,
        allow_not_mentioned=True,
        aspects=(
            _complex_aspect(
                "contract_82_survival",
                ("entailment", "entailed"),
                (_complex_source("contractnli-82-nda-19"),),
            ),
            _complex_aspect(
                "contract_82_retention",
                ("contradiction", "contradicted"),
                (_complex_source("contractnli-82-nda-20"),),
            ),
            _complex_aspect(
                "contract_82_legal_notice",
                ("not_mentioned", "not mentioned"),
                (_complex_source("contractnli-82-nda-8"),),
                allow_not_mentioned=True,
                requires_complete_scan=True,
                answer_match="any",
            ),
        ),
        notes="The third label is an absence claim and cannot be inferred from a retrieval miss.",
    ),
    _complex_case(
        "complex-13",
        (
            "For contractnli-sec-html-547.txt and contractnli-sec-text-488.txt, "
            "classify whether Confidential Information shall only include "
            "technical information, and state whether the two contracts agree."
        ),
        language="en",
        reasoning_type="scoped_contract_nli_consistency",
        required_document_ids=("contractnli-sec-html-547", "contractnli-sec-text-488"),
        aspects=(
            _complex_aspect(
                "contract_547_technical_only",
                ("contractnli-sec-html-547.txt", "contradiction", "contradicted"),
                (_complex_source("contractnli-547-nda-2"),),
            ),
            _complex_aspect(
                "contract_488_technical_only",
                ("contractnli-sec-text-488.txt", "contradiction", "contradicted"),
                (_complex_source("contractnli-488-nda-2"),),
            ),
            _complex_aspect(
                "contracts_agree",
                ("both", "agree", "consistent", "same conclusion"),
                (
                    _complex_source("contractnli-547-nda-2"),
                    _complex_source("contractnli-488-nda-2"),
                ),
                answer_match="any",
            ),
        ),
        notes="Both named contracts contradict the technical-only hypothesis.",
    ),
    _complex_case(
        "complex-14",
        (
            "Using only these four named ContractNLI files — contractnli-pdf-15.txt, "
            "contractnli-sec-text-488.txt, contractnli-sec-html-547.txt, and "
            "contractnli-pdf-82.txt — provide one grounded checklist item from "
            "each: 15 does not grant rights to Confidential Information; 488 "
            "prohibits reverse engineering; 547 protects the fact that the "
            "agreement or negotiations occurred; and 82 contains obligations that "
            "continue after termination. Do not substitute annual-report or other "
            "contract evidence."
        ),
        language="en",
        reasoning_type="scoped_four_document_contract_checklist",
        required_document_ids=(
            "contractnli-pdf-15",
            "contractnli-sec-text-488",
            "contractnli-sec-html-547",
            "contractnli-pdf-82",
        ),
        aspects=(
            _complex_aspect(
                "contract_15_no_rights",
                ("contractnli-pdf-15.txt", "does not grant"),
                (_complex_source("contractnli-15-nda-15"),),
            ),
            _complex_aspect(
                "contract_488_no_reverse_engineering",
                ("contractnli-sec-text-488.txt", "reverse engineer"),
                (_complex_source("contractnli-488-nda-11"),),
            ),
            _complex_aspect(
                "contract_547_negotiation_confidentiality",
                ("contractnli-sec-html-547.txt", "negotiated"),
                (_complex_source("contractnli-547-nda-10"),),
            ),
            _complex_aspect(
                "contract_82_surviving_obligations",
                ("contractnli-pdf-82.txt", "continue in effect"),
                (_complex_source("contractnli-82-nda-19"),),
            ),
        ),
        notes="Citation coverage is exact: all four named files are required and unrelated documents are forbidden.",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="download sources and build corpus")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument(
        "--cache",
        type=Path,
        help="source cache (default: <output>/.source-cache)",
    )
    build.add_argument(
        "--offline",
        action="store_true",
        help="fail instead of downloading a missing or invalid cached source",
    )
    build.add_argument(
        "--force",
        action="store_true",
        help="replace only the generated documents, cases, and manifest",
    )

    validate = subparsers.add_parser("validate", help="validate built corpus")
    validate.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)

    arguments = parser.parse_args()
    if arguments.command == "build":
        output = arguments.output.resolve()
        cache = (arguments.cache or output / ".source-cache").resolve()
        report = build_corpus(
            output,
            cache=cache,
            offline=arguments.offline,
            force=arguments.force,
        )
    else:
        report = validate_corpus(arguments.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def build_corpus(
    output: Path,
    *,
    cache: Path,
    offline: bool,
    force: bool,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    sources = {
        spec.key: _ensure_source(cache, spec, offline=offline)
        for spec in SOURCE_SPECS
    }
    generated_targets = (
        output / "documents",
        output / "cases.jsonl",
        output / "manifest.json",
        output / COMPLEX_CASES_FILENAME,
    )
    existing = [path for path in generated_targets if path.exists()]
    if existing and not force:
        joined = ", ".join(str(path) for path in existing)
        raise RuntimeError(f"generated corpus already exists: {joined}; use --force")

    with tempfile.TemporaryDirectory(
        prefix="document-qa-v1-",
        dir=output.parent,
    ) as directory:
        staging = Path(directory)
        documents: list[dict[str, object]] = []
        cases: list[dict[str, object]] = []
        _build_financebench(staging, sources, documents, cases)
        _build_cfqa(staging, sources, documents, cases)
        _build_tatqa(staging, sources, documents, cases)
        _build_contractnli(staging, sources, documents, cases)
        cases.sort(key=lambda case: str(case["case_id"]))
        _write_jsonl(staging / "cases.jsonl", cases)
        complex_cases = _materialize_complex_cases(cases, documents)
        _write_jsonl(
            staging / COMPLEX_CASES_FILENAME,
            complex_cases,
        )
        manifest = _manifest(staging, sources, documents, cases)
        _write_json(staging / "manifest.json", manifest)
        validate_corpus(staging)

        for target in existing:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(staging / "documents"), output / "documents")
        shutil.move(str(staging / "cases.jsonl"), output / "cases.jsonl")
        shutil.move(str(staging / "manifest.json"), output / "manifest.json")
        shutil.move(
            str(staging / COMPLEX_CASES_FILENAME),
            output / COMPLEX_CASES_FILENAME,
        )

    return validate_corpus(output)


def _materialize_complex_cases(
    cases: list[dict[str, object]],
    documents: list[dict[str, object]],
) -> list[dict[str, object]]:
    base_by_id = {
        str(case["case_id"]): case
        for case in cases
    }
    document_ids = {str(document["document_id"]) for document in documents}
    complex_cases: list[dict[str, object]] = []
    for definition in COMPLEX_CASE_DEFINITIONS:
        value = json.loads(json.dumps(definition, ensure_ascii=False))
        value["source_case_ids"] = sorted(
            {
                str(source["source_case_id"])
                for aspect in value["aspects"]
                for source in aspect["source"]
            }
        )
        _validate_complex_case(
            value,
            base_by_id=base_by_id,
            document_ids=document_ids,
            manifest_pages={
                str(document["document_id"]): document.get("pages")
                for document in documents
            },
            root=None,
        )
        complex_cases.append(value)
    referenced = {
        document_id
        for value in complex_cases
        for document_id in value["required_document_ids"]
    }
    if referenced != document_ids:
        raise RuntimeError(
            "complex cases must reference every corpus document: "
            f"missing={sorted(document_ids - referenced)}, "
            f"unknown={sorted(referenced - document_ids)}"
        )
    return complex_cases


def _ensure_source(cache: Path, spec: SourceSpec, *, offline: bool) -> Path:
    path = cache / spec.filename
    if path.exists() and _sha256(path) == spec.sha256:
        return path
    if offline:
        raise RuntimeError(f"offline source is missing or invalid: {path}")
    temporary = path.with_suffix(path.suffix + ".part")
    if temporary.exists():
        temporary.unlink()
    request = Request(
        spec.url,
        headers={"User-Agent": "rag-kb-document-qa-corpus/1.0"},
    )
    with urlopen(request, timeout=120) as response, temporary.open("wb") as target:
        shutil.copyfileobj(response, target)
    actual = _sha256(temporary)
    if actual != spec.sha256:
        temporary.unlink()
        raise RuntimeError(
            f"source checksum mismatch for {spec.key}: "
            f"expected {spec.sha256}, got {actual}"
        )
    temporary.replace(path)
    return path


def _build_financebench(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    rows = _read_jsonl(sources["financebench_cases"])
    metadata = {
        row["doc_name"]: row
        for row in _read_jsonl(sources["financebench_documents"])
    }
    for source_name, document_id, source_key in FINANCE_DOCUMENTS:
        relative = Path("documents") / "pdf" / f"{source_name}.pdf"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(sources[source_key], target)
        source_spec = SOURCE_BY_KEY[source_key]
        document_metadata = metadata[source_name]
        documents.append(
            _document_record(
                document_id,
                relative,
                target,
                source_dataset="FinanceBench",
                language="en",
                source_url=source_spec.url,
                source_revision=source_spec.revision,
                original_url=document_metadata.get("doc_link"),
                redistribution="local-only; review original publisher terms",
            )
        )
        selected = [row for row in rows if row["doc_name"] == source_name]
        if len(selected) != 7:
            raise RuntimeError(
                f"expected seven FinanceBench cases for {source_name}, "
                f"found {len(selected)}"
            )
        for row in selected:
            evidence_items = []
            for evidence in row["evidence"]:
                evidence_items.append(
                    {
                        "page": int(evidence["evidence_page_num"]) + 1,
                        "source_page_index": int(evidence["evidence_page_num"]),
                        "quote": _clean_text(evidence["evidence_text"]),
                    }
                )
            cases.append(
                {
                    "schema_version": CASE_SCHEMA,
                    "case_id": row["financebench_id"],
                    "document_id": document_id,
                    "document_path": relative.as_posix(),
                    "source_dataset": "FinanceBench",
                    "source_case_id": row["financebench_id"],
                    "language": "en",
                    "question": row["question"],
                    "gold": {
                        "answer": row["answer"],
                        "acceptable_answers": [row["answer"]],
                        "answerable": True,
                        "scale": None,
                        "numeric_tolerance": None,
                    },
                    "question_type": _normal_key(row.get("question_reasoning")),
                    "evidence": {
                        "kind": "pdf_pages",
                        "page_numbering": "pdf_1_based",
                        "items": evidence_items,
                    },
                    "justification": row.get("justification"),
                }
            )


def _cfqa_document_relative_path() -> Path:
    """Keep the physical filename aligned with the pinned source manifest."""

    return Path("documents") / "pdf" / SOURCE_BY_KEY["cfqa_fenghuo_pdf"].filename


def _build_cfqa(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    document_id = "cfqa-fenghuo-electronics-2022-annual-report"
    relative = _cfqa_document_relative_path()
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sources["cfqa_fenghuo_pdf"], target)
    source_spec = SOURCE_BY_KEY["cfqa_fenghuo_pdf"]
    documents.append(
        _document_record(
            document_id,
            relative,
            target,
            source_dataset="CFQA",
            language="zh",
            source_url=source_spec.url,
            source_revision=CFQA_REVISION,
            original_url=source_spec.url,
            redistribution="local-only; public filing remains under publisher terms",
        )
    )
    rows = json.loads(sources["cfqa_cases"].read_text(encoding="utf-8"))
    selected = [row for row in rows if int(row["id"]) in CFQA_CASE_IDS]
    if {int(row["id"]) for row in selected} != CFQA_CASE_IDS:
        raise RuntimeError("CFQA source did not contain the pinned case IDs")
    for row in selected:
        if row["公司"] != "烽火电子" or "2022" not in row["问题"]:
            raise RuntimeError(f"unexpected CFQA case content: {row['id']}")
        cases.append(
            {
                "schema_version": CASE_SCHEMA,
                "case_id": f"cfqa-{row['id']}",
                "document_id": document_id,
                "document_path": relative.as_posix(),
                "source_dataset": "CFQA",
                "source_case_id": row["id"],
                "language": "zh",
                "question": row["问题"],
                "gold": {
                    "answer": row["答案"],
                    "acceptable_answers": _cfqa_acceptable_answers(row["答案"]),
                    "answerable": True,
                    "scale": None,
                    "numeric_tolerance": None,
                },
                "question_type": "financial_report_qa",
                "evidence": {
                    "kind": "pdf_page_alternatives",
                    "page_numbering": "pdf_1_based",
                    "alternatives": [
                        {"pages": [int(page) for page in page_group]}
                        for page_group in row["答案出自"]
                    ],
                },
                "justification": None,
            }
        )


def _build_tatqa(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    rows = json.loads(sources["tatqa_dev"].read_text(encoding="utf-8"))
    by_uid = {row["table"]["uid"]: row for row in rows}
    for uid, document_id, title in TATQA_DOCUMENTS:
        row = by_uid.get(uid)
        if row is None:
            raise RuntimeError(f"TAT-QA source did not contain table {uid}")
        relative = Path("documents") / "markdown" / f"{document_id}.md"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_tatqa_markdown(title, uid, row), encoding="utf-8")
        documents.append(
            _document_record(
                document_id,
                relative,
                target,
                source_dataset="TAT-QA",
                language="en",
                source_url=SOURCE_BY_KEY["tatqa_dev"].url,
                source_revision=TATQA_REVISION,
                original_url=None,
                redistribution="CC BY 4.0 derivative",
            )
        )
        if len(row["questions"]) != 6:
            raise RuntimeError(f"expected six TAT-QA cases for {uid}")
        for question in row["questions"]:
            sections = [
                f"paragraph-{order}" for order in question["rel_paragraphs"]
            ]
            if question["answer_from"] in {"table", "table-text"}:
                sections.append("table")
            cases.append(
                {
                    "schema_version": CASE_SCHEMA,
                    "case_id": f"tatqa-{question['uid']}",
                    "document_id": document_id,
                    "document_path": relative.as_posix(),
                    "source_dataset": "TAT-QA",
                    "source_case_id": question["uid"],
                    "language": "en",
                    "question": question["question"],
                    "gold": {
                        "answer": question["answer"],
                        "acceptable_answers": _acceptable_answers(
                            question["answer"], question["scale"]
                        ),
                        "answerable": True,
                        "scale": question["scale"] or None,
                        "numeric_tolerance": (
                            {"absolute": 0.01}
                            if question["answer_type"] == "arithmetic"
                            else None
                        ),
                    },
                    "question_type": question["answer_type"],
                    "evidence": {
                        "kind": "markdown_sections",
                        "sections": list(dict.fromkeys(sections)),
                        "answer_from": question["answer_from"],
                    },
                    "justification": question["derivation"] or None,
                    "requires_comparison": bool(question["req_comparison"]),
                }
            )


def _build_contractnli(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> None:
    with zipfile.ZipFile(sources["contractnli_zip"]) as archive:
        payload = json.loads(archive.read("contract-nli/dev.json"))
    by_id = {int(document["id"]): document for document in payload["documents"]}
    labels = payload["labels"]
    for source_id, document_id, label_ids in CONTRACT_DOCUMENTS:
        document = by_id.get(source_id)
        if document is None:
            raise RuntimeError(f"ContractNLI source did not contain document {source_id}")
        relative = Path("documents") / "text" / f"{document_id}.txt"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(document["text"], encoding="utf-8", newline="\n")
        documents.append(
            _document_record(
                document_id,
                relative,
                target,
                source_dataset="ContractNLI",
                language="en",
                source_url=SOURCE_BY_KEY["contractnli_zip"].url,
                source_revision=CONTRACTNLI_REVISION,
                original_url=document.get("url"),
                redistribution="CC BY 4.0 derivative",
                extra={
                    "source_document_id": source_id,
                    "source_document_type": document["document_type"],
                    "source_filename": document["file_name"],
                },
            )
        )
        annotations = document["annotation_sets"][0]["annotations"]
        for label_id in label_ids:
            annotation = annotations[label_id]
            choice = annotation["choice"]
            spans = []
            for span_index in annotation["spans"]:
                start, end = document["spans"][span_index]
                quote = document["text"][start:end]
                spans.append({"start": start, "end": end, "quote": quote})
            canonical = {
                "Entailment": "entailment",
                "Contradiction": "contradiction",
                "NotMentioned": "not_mentioned",
            }[choice]
            aliases = {
                "entailment": ["entailment", "entailed", "supported"],
                "contradiction": ["contradiction", "contradicted"],
                "not_mentioned": ["not_mentioned", "not mentioned", "unknown"],
            }[canonical]
            hypothesis = labels[label_id]["hypothesis"]
            cases.append(
                {
                    "schema_version": CASE_SCHEMA,
                    "case_id": f"contractnli-{source_id}-{label_id}",
                    "document_id": document_id,
                    "document_path": relative.as_posix(),
                    "source_dataset": "ContractNLI",
                    "source_case_id": f"{source_id}:{label_id}",
                    "language": "en",
                    "question": (
                        "According to the agreement, is the following statement "
                        "entailed, contradicted, or not mentioned? "
                        f"{hypothesis}"
                    ),
                    "gold": {
                        "answer": canonical,
                        "acceptable_answers": aliases,
                        "answerable": canonical != "not_mentioned",
                        "scale": None,
                        "numeric_tolerance": None,
                    },
                    "question_type": "document_nli",
                    "evidence": {
                        "kind": "absence" if not spans else "text_spans",
                        "spans": spans,
                    },
                    "justification": labels[label_id]["short_description"],
                }
            )


def _tatqa_markdown(
    title: str,
    uid: str,
    row: dict[str, Any],
) -> str:
    lines = [
        f"# {title}",
        "",
        (
            "_Derived from a TAT-QA financial-report context under CC BY 4.0; "
            f"source table UID `{uid}`._"
        ),
        "",
        "## Narrative",
        "",
    ]
    for paragraph in sorted(row["paragraphs"], key=lambda item: item["order"]):
        order = paragraph["order"]
        lines.extend(
            (
                f'<a id="paragraph-{order}"></a>',
                f"### Paragraph {order}",
                "",
                paragraph["text"].strip(),
                "",
            )
        )
    table = row["table"]["table"]
    width = max(len(table_row) for table_row in table)
    lines.extend(
        (
            '<a id="table"></a>',
            "## Financial table",
            "",
            _markdown_row([f"Column {index + 1}" for index in range(width)]),
            _markdown_row(["---"] * width),
        )
    )
    for table_row in table:
        values = list(table_row) + [""] * (width - len(table_row))
        lines.append(_markdown_row(values))
    lines.append("")
    return "\n".join(lines)


def _markdown_row(values: Iterable[object]) -> str:
    escaped = [
        str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
        for value in values
    ]
    return "| " + " | ".join(escaped) + " |"


def _document_record(
    document_id: str,
    relative: Path,
    target: Path,
    *,
    source_dataset: str,
    language: str,
    source_url: str,
    source_revision: str | None,
    original_url: str | None,
    redistribution: str,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    suffix = target.suffix.lower().lstrip(".")
    record: dict[str, object] = {
        "document_id": document_id,
        "path": relative.as_posix(),
        "format": suffix,
        "language": language,
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "source_dataset": source_dataset,
        "source_url": source_url,
        "source_revision": source_revision,
        "original_url": original_url,
        "redistribution": redistribution,
    }
    if suffix == "pdf":
        record["pages"] = len(PdfReader(target).pages)
    if extra:
        record.update(extra)
    return record


def _manifest(
    root: Path,
    sources: dict[str, Path],
    documents: list[dict[str, object]],
    cases: list[dict[str, object]],
) -> dict[str, object]:
    documents.sort(key=lambda document: str(document["document_id"]))
    cases_sha256 = _sha256(root / "cases.jsonl")
    complex_cases_path = root / COMPLEX_CASES_FILENAME
    complex_cases_sha256 = (
        _sha256(complex_cases_path)
        if complex_cases_path.is_file()
        else None
    )
    dataset_material = {
        "cases_sha256": cases_sha256,
        "complex_cases_sha256": complex_cases_sha256,
        "documents": [
            (document["document_id"], document["sha256"])
            for document in documents
        ],
    }
    dataset_sha256 = hashlib.sha256(
        json.dumps(
            dataset_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": CORPUS_SCHEMA,
        "dataset_id": "public-enterprise-document-qa-v1",
        "dataset_sha256": dataset_sha256,
        "cases_path": "cases.jsonl",
        "cases_sha256": cases_sha256,
        "complex_cases_path": COMPLEX_CASES_FILENAME,
        "complex_cases_sha256": complex_cases_sha256,
        "complex_case_count": len(COMPLEX_CASE_DEFINITIONS),
        "document_count": len(documents),
        "case_count": len(cases),
        "format_counts": dict(sorted(Counter(d["format"] for d in documents).items())),
        "language_counts": dict(
            sorted(Counter(case["language"] for case in cases).items())
        ),
        "source_case_counts": dict(
            sorted(Counter(case["source_dataset"] for case in cases).items())
        ),
        "documents": documents,
        "sources": [
            {
                "key": spec.key,
                "dataset": spec.dataset,
                "url": spec.url,
                "revision": spec.revision,
                "sha256": spec.sha256,
                "cached_sha256": _sha256(sources[spec.key]),
            }
            for spec in SOURCE_SPECS
        ],
    }


def validate_corpus(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    cases_path = root / "cases.jsonl"
    if not manifest_path.is_file() or not cases_path.is_file():
        raise RuntimeError(f"corpus is incomplete: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CORPUS_SCHEMA:
        raise RuntimeError("unsupported corpus schema")
    if _sha256(cases_path) != manifest.get("cases_sha256"):
        raise RuntimeError("cases checksum does not match manifest")
    cases = _read_jsonl(cases_path)
    if len(cases) != manifest.get("case_count"):
        raise RuntimeError("case count does not match manifest")

    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) != manifest.get(
        "document_count"
    ):
        raise RuntimeError("document count does not match manifest")
    by_id: dict[str, dict[str, object]] = {}
    for document in documents:
        document_id = _required_string(document, "document_id")
        if document_id in by_id:
            raise RuntimeError(f"duplicate document ID: {document_id}")
        path = _safe_document_path(root, _required_string(document, "path"))
        if not path.is_file():
            raise RuntimeError(f"document is missing: {path}")
        if _sha256(path) != document.get("sha256"):
            raise RuntimeError(f"document checksum mismatch: {path}")
        if path.stat().st_size != document.get("bytes"):
            raise RuntimeError(f"document size mismatch: {path}")
        expected_format = path.suffix.lower().lstrip(".")
        if document.get("format") != expected_format:
            raise RuntimeError(f"document format mismatch: {path}")
        if expected_format == "pdf":
            pages = len(PdfReader(path).pages)
            if pages != document.get("pages"):
                raise RuntimeError(f"PDF page count mismatch: {path}")
        by_id[document_id] = document

    case_ids: set[str] = set()
    referenced_documents: Counter[str] = Counter()
    for case in cases:
        if case.get("schema_version") != CASE_SCHEMA:
            raise RuntimeError("unsupported case schema")
        case_id = _required_string(case, "case_id")
        if case_id in case_ids:
            raise RuntimeError(f"duplicate case ID: {case_id}")
        case_ids.add(case_id)
        document_id = _required_string(case, "document_id")
        document = by_id.get(document_id)
        if document is None:
            raise RuntimeError(f"case references unknown document: {case_id}")
        if case.get("document_path") != document["path"]:
            raise RuntimeError(f"case document path mismatch: {case_id}")
        referenced_documents[document_id] += 1
        gold = case.get("gold")
        if not isinstance(gold, dict) or "answer" not in gold:
            raise RuntimeError(f"case has no gold answer: {case_id}")
        _validate_evidence(root, document, case)

    complex_cases_path_value = manifest.get(
        "complex_cases_path", COMPLEX_CASES_FILENAME
    )
    complex_cases_path = root / str(complex_cases_path_value)
    complex_cases: list[dict[str, object]] = []
    if complex_cases_path.is_file():
        expected_hash = manifest.get("complex_cases_sha256")
        if not isinstance(expected_hash, str) or _sha256(complex_cases_path) != expected_hash:
            raise RuntimeError("complex case checksum does not match manifest")
        complex_cases = _read_jsonl(complex_cases_path)
        expected_count = manifest.get("complex_case_count")
        if expected_count is not None and len(complex_cases) != expected_count:
            raise RuntimeError("complex case count does not match manifest")
        complex_ids: set[str] = set()
        manifest_pages = {
            str(document["document_id"]): document.get("pages")
            for document in documents
        }
        for complex_case in complex_cases:
            case_id = _required_string(complex_case, "case_id")
            if case_id in complex_ids:
                raise RuntimeError(f"duplicate complex case ID: {case_id}")
            complex_ids.add(case_id)
            _validate_complex_case(
                complex_case,
                base_by_id={
                    str(case["case_id"]): case
                    for case in cases
                },
                document_ids=set(by_id),
                manifest_pages=manifest_pages,
                root=root,
            )
        if len(complex_cases) != len(COMPLEX_CASE_DEFINITIONS):
            raise RuntimeError("complex case definition count drifted")
    elif manifest.get("complex_cases_sha256") is not None:
        raise RuntimeError("manifest declares missing complex cases")

    missing_cases = set(by_id) - set(referenced_documents)
    if missing_cases:
        raise RuntimeError(f"documents have no cases: {sorted(missing_cases)}")
    actual_formats = dict(sorted(Counter(d["format"] for d in documents).items()))
    if actual_formats != manifest.get("format_counts"):
        raise RuntimeError("format counts do not match manifest")
    dataset_material: dict[str, object] = {
        "cases_sha256": manifest["cases_sha256"],
        "documents": [
            (document["document_id"], document["sha256"])
            for document in documents
        ],
    }
    if "complex_cases_sha256" in manifest:
        dataset_material["complex_cases_sha256"] = manifest["complex_cases_sha256"]
    actual_dataset_hash = hashlib.sha256(
        json.dumps(
            dataset_material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual_dataset_hash != manifest.get("dataset_sha256"):
        raise RuntimeError("dataset checksum does not match manifest")
    return {
        "dataset_id": manifest["dataset_id"],
        "dataset_sha256": manifest["dataset_sha256"],
        "documents": len(documents),
        "cases": len(cases),
        "complex_cases": len(complex_cases),
        "formats": actual_formats,
        "languages": dict(sorted(Counter(c["language"] for c in cases).items())),
        "source_cases": dict(
            sorted(Counter(c["source_dataset"] for c in cases).items())
        ),
        "status": "valid",
    }


def _validate_complex_case(
    value: dict[str, object],
    *,
    base_by_id: dict[str, dict[str, object]],
    document_ids: set[str],
    manifest_pages: dict[str, object],
    root: Path | None,
) -> None:
    if value.get("schema_version") != COMPLEX_CASE_SCHEMA:
        raise RuntimeError("unsupported complex case schema")
    case_id = _required_string(value, "case_id")
    _required_string(value, "question")
    _required_string(value, "language")
    _required_string(value, "reasoning_type")
    required_documents = value.get("required_document_ids")
    if not isinstance(required_documents, list) or not required_documents:
        raise RuntimeError(f"complex case has no required documents: {case_id}")
    if any(not isinstance(item, str) or not item for item in required_documents):
        raise RuntimeError(f"complex case has invalid document IDs: {case_id}")
    if len(required_documents) != len(set(required_documents)):
        raise RuntimeError(f"complex case document IDs are duplicated: {case_id}")
    if not set(required_documents) <= document_ids:
        raise RuntimeError(f"complex case references unknown document: {case_id}")
    citation_documents = value.get("required_citation_document_ids")
    if citation_documents != required_documents:
        raise RuntimeError(f"complex case citation scope mismatch: {case_id}")
    if value.get("forbid_unrelated_citations") is not True:
        raise RuntimeError(f"complex case must forbid unrelated citations: {case_id}")
    aspects = value.get("aspects")
    if not isinstance(aspects, list) or not aspects:
        raise RuntimeError(f"complex case has no aspects: {case_id}")
    aspect_ids: set[str] = set()
    referenced_source_cases: set[str] = set()
    for aspect in aspects:
        if not isinstance(aspect, dict):
            raise RuntimeError(f"complex case has invalid aspect: {case_id}")
        aspect_id = _required_string(aspect, "aspect_id")
        if aspect_id in aspect_ids:
            raise RuntimeError(f"duplicate complex aspect: {case_id}")
        aspect_ids.add(aspect_id)
        variants = aspect.get("answer_variants")
        if (
            not isinstance(variants, list)
            or not variants
            or any(not isinstance(item, str) or not item.strip() for item in variants)
        ):
            raise RuntimeError(f"complex aspect has no answer variants: {case_id}")
        answer_match = aspect.get("answer_match", "all")
        if answer_match not in {"all", "any"}:
            raise RuntimeError(f"complex aspect answer_match is invalid: {case_id}")
        expected_decimal = aspect.get("expected_decimal")
        tolerance = aspect.get("numeric_tolerance")
        if expected_decimal is not None:
            if not isinstance(expected_decimal, str):
                raise RuntimeError(f"complex decimal must be a string: {case_id}")
            try:
                Decimal(expected_decimal)
            except (InvalidOperation, ValueError):
                raise RuntimeError(f"invalid complex decimal: {case_id}") from None
            if not isinstance(tolerance, str):
                raise RuntimeError(f"complex decimal has no tolerance: {case_id}")
            try:
                if Decimal(tolerance) < 0:
                    raise InvalidOperation
            except (InvalidOperation, ValueError):
                raise RuntimeError(f"invalid complex tolerance: {case_id}") from None
        elif tolerance is not None:
            raise RuntimeError(f"complex tolerance has no decimal: {case_id}")
        source = aspect.get("source")
        if not isinstance(source, list) or not source:
            raise RuntimeError(f"complex aspect has no source: {case_id}")
        aspect_requires_scan = aspect.get("requires_complete_scan") is True
        if aspect_requires_scan and value.get("requires_complete_scan") is not True:
            raise RuntimeError(f"complex scan aspect is not enabled at case level: {case_id}")
        for reference in source:
            if not isinstance(reference, dict):
                raise RuntimeError(f"invalid complex source reference: {case_id}")
            source_case_id = _required_string(reference, "source_case_id")
            referenced_source_cases.add(source_case_id)
            base_case = base_by_id.get(source_case_id)
            if base_case is None:
                raise RuntimeError(
                    f"complex case references unknown source case {source_case_id}: {case_id}"
                )
            source_document_id = str(base_case["document_id"])
            if source_document_id not in required_documents:
                raise RuntimeError(f"complex source is outside required scope: {case_id}")
            locator = reference.get("evidence_locator")
            if not isinstance(locator, dict):
                raise RuntimeError(f"complex source has no evidence locator: {case_id}")
            _validate_complex_locator(
                locator,
                base_case=base_case,
                document_id=source_document_id,
                manifest_pages=manifest_pages,
                case_id=case_id,
            )
            if base_case.get("evidence", {}).get("kind") == "absence":
                if not aspect.get("allow_not_mentioned") and not aspect_requires_scan:
                    raise RuntimeError(
                        f"absence source is not explicitly allowed: {case_id}"
                    )
                if base_case.get("gold", {}).get("answerable") is not False:
                    raise RuntimeError(f"absence source is answerable: {case_id}")
    declared_sources = value.get("source_case_ids")
    if not isinstance(declared_sources, list) or set(declared_sources) != referenced_source_cases:
        raise RuntimeError(f"complex source case index mismatch: {case_id}")
    if value.get("requires_complete_scan") is True and not any(
        aspect.get("requires_complete_scan") is True for aspect in aspects
    ):
        raise RuntimeError(f"complex case scan flag has no scan aspect: {case_id}")
    if value.get("allow_not_mentioned") is True and not any(
        aspect.get("allow_not_mentioned") is True for aspect in aspects
    ):
        raise RuntimeError(f"complex absence flag has no absence aspect: {case_id}")


def _validate_complex_locator(
    locator: dict[str, object],
    *,
    base_case: dict[str, object],
    document_id: str,
    manifest_pages: dict[str, object],
    case_id: str,
) -> None:
    kind = locator.get("kind")
    evidence = base_case.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(f"source case has no evidence: {case_id}")
    evidence_kind = evidence.get("kind")
    if kind == "base_case_evidence":
        return
    if kind == "pdf_page":
        page = locator.get("page")
        pages = manifest_pages.get(document_id)
        if not isinstance(page, int) or not isinstance(pages, int) or not 1 <= page <= pages:
            raise RuntimeError(f"complex PDF locator is invalid: {case_id}")
        if evidence_kind != "pdf_pages":
            raise RuntimeError(f"complex PDF locator has mismatched source kind: {case_id}")
        return
    if kind == "markdown_section":
        section = locator.get("section")
        if not isinstance(section, str) or section not in evidence.get("sections", ()):
            raise RuntimeError(f"complex Markdown locator is invalid: {case_id}")
        return
    if kind == "text_span":
        span_index = locator.get("span_index", 0)
        spans = evidence.get("spans")
        if evidence_kind != "text_spans" or not isinstance(span_index, int):
            raise RuntimeError(f"complex text locator is invalid: {case_id}")
        if not isinstance(spans, list) or not 0 <= span_index < len(spans):
            raise RuntimeError(f"complex text span locator is out of range: {case_id}")
        return
    if kind == "absence":
        if evidence_kind != "absence":
            raise RuntimeError(f"complex absence locator is invalid: {case_id}")
        return
    raise RuntimeError(f"unknown complex evidence locator: {case_id}")


def _validate_evidence(
    root: Path,
    document: dict[str, object],
    case: dict[str, object],
) -> None:
    evidence = case.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError(f"case has invalid evidence: {case['case_id']}")
    kind = evidence.get("kind")
    path = _safe_document_path(root, str(document["path"]))
    if kind == "pdf_pages":
        pages = int(document["pages"])
        for item in evidence.get("items", []):
            _validate_page(item.get("page"), pages, case)
    elif kind == "pdf_page_alternatives":
        pages = int(document["pages"])
        alternatives = evidence.get("alternatives", [])
        if not alternatives:
            raise RuntimeError(f"case has no PDF evidence: {case['case_id']}")
        for alternative in alternatives:
            for page in alternative.get("pages", []):
                _validate_page(page, pages, case)
    elif kind == "markdown_sections":
        markdown = path.read_text(encoding="utf-8")
        for section in evidence.get("sections", []):
            if f'id="{section}"' not in markdown:
                raise RuntimeError(
                    f"missing Markdown evidence section {section}: {case['case_id']}"
                )
    elif kind == "text_spans":
        text = path.read_text(encoding="utf-8")
        spans = evidence.get("spans", [])
        if not spans:
            raise RuntimeError(f"case has no text spans: {case['case_id']}")
        for span in spans:
            start = int(span["start"])
            end = int(span["end"])
            if not 0 <= start < end <= len(text):
                raise RuntimeError(f"invalid evidence span: {case['case_id']}")
            if text[start:end] != span["quote"]:
                raise RuntimeError(f"evidence quote mismatch: {case['case_id']}")
    elif kind == "absence":
        if evidence.get("spans"):
            raise RuntimeError(f"absence case contains spans: {case['case_id']}")
        if case["gold"].get("answerable") is not False:
            raise RuntimeError(f"absence case is marked answerable: {case['case_id']}")
    else:
        raise RuntimeError(f"unknown evidence kind {kind}: {case['case_id']}")


def _validate_page(value: object, pages: int, case: dict[str, object]) -> None:
    if not isinstance(value, int) or not 1 <= value <= pages:
        raise RuntimeError(f"invalid PDF evidence page: {case['case_id']}")


def _safe_document_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise RuntimeError(f"unsafe document path: {relative}")
    path = (root / raw).resolve()
    documents_root = (root / "documents").resolve()
    if documents_root not in path.parents:
        raise RuntimeError(f"document path escapes corpus: {relative}")
    return path


def _acceptable_answers(answer: object, scale: str) -> list[str]:
    if isinstance(answer, list):
        values = [str(value) for value in answer]
        if len(values) == 1:
            return values
        return ["; ".join(values), ", ".join(values)]
    rendered = str(answer)
    answers = [rendered]
    if scale:
        answers.append(f"{rendered} {scale}")
    return answers


def _cfqa_acceptable_answers(answer: str) -> list[str]:
    values = [answer]
    without_source_marker = re.sub(r"\[\d+\]\s*$", "", answer).rstrip()
    if without_source_marker != answer:
        values.append(without_source_marker)
    without_layout_spaces = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", answer)
    if without_layout_spaces not in values:
        values.append(without_layout_spaces)
    return values


def _normal_key(value: object) -> str:
    if value is None or str(value).strip().lower() == "none":
        return "unspecified"
    normalized = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value)
    )
    return "_".join(part for part in normalized.split("_") if part)


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _required_string(value: dict[str, object], key: str) -> str:
    candidate = value.get(key)
    if not isinstance(candidate, str) or not candidate:
        raise RuntimeError(f"missing required string: {key}")
    return candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as target:
        for value in values:
            target.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            target.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
