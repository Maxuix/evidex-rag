#!/usr/bin/env python3
"""Build and validate the source-derived routing-rag-v3 corpus.

The corpus contains short, author-written factual summaries derived from public
project metadata and official project pages.  It deliberately does not copy
long source passages.  Gold relations and cases are kept outside the upload
documents so that source attribution is auditable without leaking relation IDs
into retrieval text.

This builder is offline and deterministic.  It never fetches the network, runs
Docker, starts a service, or calls a model.  Web source URLs and access dates
are recorded in SOURCE_NOTICES.md; a host baseline is required before the
candidate Graph route labels can become the primary routing metric.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import textwrap
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "routing-rag-v3"
DATASET_ID = "routing-rag-v3-open-source"
SCHEMA = "open_source_rag_corpus_v1"
CASE_SCHEMA = "open_source_rag_case_v1"
ACCESS_DATE = "2026-08-21"


def _clean(value: str) -> str:
    return textwrap.dedent(value).strip() + "\n"


DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "document_id": "os-doc-01",
        "filename": "01_python_foundation.md",
        "format": "md",
        "material_type": "official_organization_summary",
        "source_ids": ["S01", "S02"],
        "content": _clean(
            """
            # Python Software Foundation 与 PyPI

            ## 组织定位

            Python Software Foundation（PSF）是围绕 Python 语言发展的非营利组织。其公开使命包括推动、保护和发展 Python 语言，并支持 Python 社区。

            ## 知识产权职责

            PSF 负责 Python 相关知识产权的管理，并为大多数 Python 发布版本持有相应权利。这里的“负责”是基金会的治理职责，不表示 PSF 是 Python 的唯一代码贡献者。

            ## 包索引基础设施

            PSF 运营 Python Package Index，通常简称 PyPI。PyPI 是 Python 软件包的公共索引，项目发布页和包元数据由该索引提供。
            """
        ),
    },
    {
        "document_id": "os-doc-02",
        "filename": "02_python_packaging_history.txt",
        "format": "txt",
        "material_type": "official_history_summary",
        "source_ids": ["S03"],
        "content": _clean(
            """
            Python Packaging Authority 历史摘录

            2011 年，Python Packaging Authority（PyPA）成立，用来接手 pip 和 virtualenv 的维护工作。这个时间事实来自 PyPA 的历史页面；它不意味着 PyPA 是所有 Python 项目的版权持有人。

            维护范围

            pip 是 Python 包安装工具，virtualenv 用于创建隔离环境。两者在该历史叙述中作为 PyPA 接手维护的项目分别出现。
            """
        ),
    },
    {
        "document_id": "os-doc-03",
        "filename": "03_numpy_github.md",
        "format": "md",
        "material_type": "github_repository_summary",
        "source_ids": ["S04"],
        "content": _clean(
            """
            # NumPy GitHub 仓库快照

            ## 仓库身份

            NumPy 的公开源代码仓库是 `numpy/numpy`。GitHub 项目简介把 NumPy 定位为 Python 科学计算的基础包；仓库页面同时提供源代码、问题跟踪和文档入口。

            ## 许可证

            GitHub 项目页将 NumPy 标为 BSD-3-Clause。这个结论只描述仓库项目页的主许可证标签；NumPy 的发行包还会列出若干第三方组件许可证，不能把两者混成一个单一许可证。
            """
        ),
    },
    {
        "document_id": "os-doc-04",
        "filename": "04_numpy_pypi.md",
        "format": "md",
        "material_type": "pypi_metadata_summary",
        "source_ids": ["S05"],
        "content": _clean(
            """
            # NumPy PyPI 元数据快照

            ## 项目归属

            PyPI 项目页的 owner 显示为 NumPy，维护者字段显示 NumPy 开发团队相关账户。该页面将项目名称、维护者和发布包信息分开列出。

            ## 发行版许可证表达式

            当前快照中的许可证表达式为 `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`。这是 PyPI 元数据的组合表达式，不应被简化成只有 BSD-3-Clause。

            ## 运行时要求

            该快照的项目元数据声明 Python 版本要求为 3.12 或更高。版本要求属于时间敏感字段，测试集把访问日期固定在来源清单中。
            """
        ),
    },
    {
        "document_id": "os-doc-05",
        "filename": "05_pandas_project.md",
        "format": "md",
        "material_type": "github_pypi_project_summary",
        "source_ids": ["S06", "S07"],
        "content": _clean(
            """
            # pandas 项目资料

            ## 代码仓库与许可证

            pandas 的源代码仓库是 `pandas-dev/pandas`。项目构建元数据把许可写为 BSD-3-Clause，作者字段使用 Pandas Development Team 的名称。

            ## 文档入口

            pandas 的 PyPI 项目页把官方文档入口指向 PyData。PyData 是文档和社区生态入口，不是 pandas 的 GitHub 代码仓库所有者。

            ## 项目定位

            pandas 提供面向数据分析的 Python 数据结构与数据处理工具。该描述是项目简介的事实摘要，不是 API 使用教程。
            """
        ),
    },
    {
        "document_id": "os-doc-06",
        "filename": "06_scikit_learn_repository.txt",
        "format": "txt",
        "material_type": "github_project_summary",
        "source_ids": ["S08", "S09", "S10"],
        "content": _clean(
            """
            scikit-learn 项目资料

            代码仓库

            scikit-learn 的公开源代码仓库是 `scikit-learn/scikit-learn`。GitHub 项目简介将其描述为 Python 机器学习模块，仓库使用 3-Clause BSD 许可证。

            名称与导入空间

            PyPI 的正式项目名是 `scikit-learn`，而 Python 代码中的导入空间使用 `sklearn`。两者指向同一个项目，但不能据此把 `sklearn` 当成官方 PyPI 分发名。

            历史背景

            项目资料记载 scikit-learn 于 2007 年由 David Cournapeau 发起，之后由志愿者社区持续维护。历史发起人和当前维护者是不同关系。
            """
        ),
    },
    {
        "document_id": "os-doc-07",
        "filename": "07_scikit_learn_pypi.md",
        "format": "md",
        "material_type": "pypi_dependency_summary",
        "source_ids": ["S11"],
        "content": _clean(
            """
            # scikit-learn PyPI 依赖快照

            ## 分发元数据

            PyPI 记录的正式项目名称是 `scikit-learn`，许可证表达式为 BSD-3-Clause。安装名和导入名在项目资料中分别出现。

            ## 直接依赖

            该快照列出的核心依赖包括 NumPy 和 SciPy。这里的“依赖”是包元数据中的安装要求，不表示 scikit-learn 由 NumPy 或 SciPy 负责维护。

            ## 查询提示

            仅输入 `sklearn` 时，问题可能是在问导入空间；只有出现“安装包名”或 PyPI 上的项目名时，答案才应使用 `scikit-learn`。
            """
        ),
    },
    {
        "document_id": "os-doc-08",
        "filename": "08_scipy_official.md",
        "format": "md",
        "material_type": "official_project_summary",
        "source_ids": ["S12", "S13"],
        "content": _clean(
            """
            # SciPy 官方项目摘要

            ## 源码与许可证

            SciPy 的公开 GitHub 仓库是 `scipy/scipy`，项目页标注 BSD-3-Clause。SciPy 官方网站说明其代码在 GitHub 上由开放社区开发和维护。

            ## 科学计算关系

            SciPy 项目资料把 SciPy 描述为建立在 NumPy 之上的科学计算库，并称它是 scikit-learn 等 Python 包的重要基础。这里同时存在“建立在 NumPy 之上”和“为 scikit-learn 提供基础”两条不同关系。

            ## 边界

            “SciPy 是 scikit-learn 的基础之一”不等于“scikit-learn 是 SciPy 的子项目”，也不等于两者使用同一 GitHub 仓库。
            """
        ),
    },
    {
        "document_id": "os-doc-09",
        "filename": "09_jupyter_history.md",
        "format": "md",
        "material_type": "official_project_summary",
        "source_ids": ["S14", "S15"],
        "content": _clean(
            """
            # Project Jupyter 资料

            ## 起源

            Project Jupyter 于 2014 年从 IPython 项目演化而来，目标是支持跨编程语言的交互式数据科学和科学计算。IPython 是起源项目，不是 Jupyter 的许可证名称。

            ## 开发方式

            Jupyter 社区在 GitHub 上公开开发，项目页面将其描述为非营利、开源项目。

            ## 许可证

            Jupyter 项目代码使用修改版 BSD，也称 BSD-3-Clause。该许可证结论来自项目治理页，不应因为 Jupyter 起源于 IPython 就推断它使用 IPython 的许可证。
            """
        ),
    },
    {
        "document_id": "os-doc-10",
        "filename": "10_numfocus_sponsored_projects.md",
        "format": "md",
        "material_type": "official_sponsorship_roster",
        "source_ids": ["S16"],
        "content": _clean(
            """
            # NumFOCUS 当前赞助项目名单快照

            ## 名单

            NumFOCUS 的 Sponsored Projects 页面列出 NumPy、pandas、SciPy、Project Jupyter 和 scikit-learn 等项目。此处的“赞助项目”是该页面的项目分类，不等同于项目的代码仓库归属。

            ## 法律与运营关系

            NumFOCUS 为受财政赞助的项目提供法律实体、财务管理以及运营和法律支持。项目仍由各自的开源社区和维护者开发。

            ## 名单边界

            本文件只记录页面抓取日的名单快照。它不据此声称名单之外的项目绝对没有任何其他资助关系。
            """
        ),
    },
    {
        "document_id": "os-doc-11",
        "filename": "11_cncf_kubernetes.md",
        "format": "md",
        "material_type": "official_foundation_summary",
        "source_ids": ["S17", "S18", "S19"],
        "content": _clean(
            """
            # CNCF 与 Kubernetes

            ## 托管关系

            Cloud Native Computing Foundation（CNCF）是云原生开源项目的中立协作组织，公开页面把 Kubernetes 列为其托管项目之一。

            ## 毕业与治理

            CNCF 在 2018 年公告中宣布 Kubernetes 成为其第一个毕业项目。公告同时说明 Kubernetes 建立了自己的治理结构和 Steering Committee。

            ## 许可证与上层组织

            同一公告说明 Kubernetes 以 Apache License 2.0 发布，并指出 CNCF 是 Linux Foundation 的一部分。项目托管、许可证和基金会层级是三条不同关系。
            """
        ),
    },
    {
        "document_id": "os-doc-12",
        "filename": "12_kubernetes_repository.txt",
        "format": "txt",
        "material_type": "github_repository_summary",
        "source_ids": ["S20"],
        "content": _clean(
            """
            Kubernetes GitHub 仓库快照

            仓库身份

            Kubernetes 的主 GitHub 仓库是 `kubernetes/kubernetes`。仓库页面把项目描述为跨多主机管理容器化应用的开源系统，并列出 Apache-2.0 许可证。

            简称与治理入口

            Kubernetes 常被简称为 K8s。仓库页面把 community、steering 和 enhancements 等位置作为治理或路线信息入口，但这些仓库不是主代码仓库的别名。
            """
        ),
    },
    {
        "document_id": "os-doc-13",
        "filename": "13_project_catalog.csv",
        "format": "csv",
        "material_type": "derived_snapshot_table",
        "source_ids": ["S04", "S06", "S12", "S20"],
        "content": _clean(
            """
            项目,生态类别,公开仓库,主许可证
            NumPy,Python 科学计算基础库,numpy/numpy,BSD-3-Clause
            pandas,Python 数据处理工具,pandas-dev/pandas,BSD-3-Clause
            SciPy,Python 科学算法库,scipy/scipy,BSD-3-Clause
            scikit-learn,Python 机器学习库,scikit-learn/scikit-learn,BSD-3-Clause
            Kubernetes,云原生容器编排系统,kubernetes/kubernetes,Apache-2.0
            """
        ),
    },
    {
        "document_id": "os-doc-14",
        "filename": "14_name_disambiguation.txt",
        "format": "txt",
        "material_type": "disambiguation_controls",
        "source_ids": ["S09", "S10", "S20"],
        "content": _clean(
            """
            名称消歧控制材料

            scikit-learn 与 sklearn

            `scikit-learn` 是 PyPI 的正式项目名；`sklearn` 是 Python 导入空间。测试问题若询问 pip 安装名，不能回答 `sklearn`。

            Kubernetes 与 K8s

            `K8s` 是 Kubernetes 的常见简称；`kubernetes/kubernetes` 是主仓库路径。简称、项目名和仓库名在检索中应归一到同一个项目实体。

            许可证近名陷阱

            Kubernetes 的主仓库页面标记 Apache-2.0，而 NumPy、pandas、SciPy 和 scikit-learn 的主项目页标记 BSD-3-Clause。不能因为这些项目都出现在同一个项目目录中，就把许可证相互迁移。
            """
        ),
    },
)


ENTITIES: tuple[dict[str, Any], ...] = (
    {"entity_id": "org:psf", "canonical_name": "Python Software Foundation", "aliases": ["PSF"], "type": "organization"},
    {"entity_id": "project:python", "canonical_name": "Python", "aliases": ["Python programming language"], "type": "project"},
    {"entity_id": "service:pypi", "canonical_name": "Python Package Index", "aliases": ["PyPI", "Python package repository"], "type": "service"},
    {"entity_id": "org:pypa", "canonical_name": "Python Packaging Authority", "aliases": ["PyPA"], "type": "organization"},
    {"entity_id": "project:pip", "canonical_name": "pip", "aliases": [], "type": "project"},
    {"entity_id": "project:virtualenv", "canonical_name": "virtualenv", "aliases": [], "type": "project"},
    {"entity_id": "project:numpy", "canonical_name": "NumPy", "aliases": ["numpy"], "type": "project"},
    {"entity_id": "repo:numpy", "canonical_name": "numpy/numpy", "aliases": [], "type": "repository"},
    {"entity_id": "license:bsd3", "canonical_name": "BSD-3-Clause", "aliases": ["3-Clause BSD", "modified BSD"], "type": "license"},
    {"entity_id": "license:numpy-expression", "canonical_name": "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0", "aliases": [], "type": "license_expression"},
    {"entity_id": "project:pandas", "canonical_name": "pandas", "aliases": [], "type": "project"},
    {"entity_id": "repo:pandas", "canonical_name": "pandas-dev/pandas", "aliases": [], "type": "repository"},
    {"entity_id": "org:pydata", "canonical_name": "PyData", "aliases": [], "type": "organization"},
    {"entity_id": "project:scikit-learn", "canonical_name": "scikit-learn", "aliases": ["sklearn", "scikit learn"], "type": "project"},
    {"entity_id": "entity:sklearn-namespace", "canonical_name": "sklearn import namespace", "aliases": ["sklearn"], "type": "namespace"},
    {"entity_id": "repo:scikit-learn", "canonical_name": "scikit-learn/scikit-learn", "aliases": [], "type": "repository"},
    {"entity_id": "project:scipy", "canonical_name": "SciPy", "aliases": ["scipy"], "type": "project"},
    {"entity_id": "repo:scipy", "canonical_name": "scipy/scipy", "aliases": [], "type": "repository"},
    {"entity_id": "project:jupyter", "canonical_name": "Project Jupyter", "aliases": ["Jupyter"], "type": "project"},
    {"entity_id": "project:ipython", "canonical_name": "IPython", "aliases": [], "type": "project"},
    {"entity_id": "org:numfocus", "canonical_name": "NumFOCUS", "aliases": [], "type": "organization"},
    {"entity_id": "org:cncf", "canonical_name": "Cloud Native Computing Foundation", "aliases": ["CNCF"], "type": "organization"},
    {"entity_id": "org:linux-foundation", "canonical_name": "Linux Foundation", "aliases": [], "type": "organization"},
    {"entity_id": "project:kubernetes", "canonical_name": "Kubernetes", "aliases": ["K8s"], "type": "project"},
    {"entity_id": "alias:k8s", "canonical_name": "K8s", "aliases": ["k8s"], "type": "alias"},
    {"entity_id": "repo:kubernetes", "canonical_name": "kubernetes/kubernetes", "aliases": [], "type": "repository"},
    {"entity_id": "license:apache2", "canonical_name": "Apache-2.0", "aliases": ["Apache License 2.0"], "type": "license"},
    {"entity_id": "entity:github", "canonical_name": "GitHub", "aliases": [], "type": "platform"},
)


def _relation(
    relation_id: str,
    subject_entity_id: str,
    predicate: str,
    object_entity_id: str,
    document_id: str,
    section_title: str,
    evidence: str,
    source_ids: Iterable[str],
) -> dict[str, Any]:
    return {
        "relation_id": relation_id,
        "subject_entity_id": subject_entity_id,
        "predicate": predicate,
        "object_entity_id": object_entity_id,
        "document_id": document_id,
        "source_locator": {"kind": "section", "section_title": section_title},
        "evidence_text": evidence,
        "source_ids": list(source_ids),
    }


RELATIONS: tuple[dict[str, Any], ...] = (
    _relation("OSR001", "org:psf", "stewards", "project:python", "os-doc-01", "知识产权职责", "PSF 负责 Python 相关知识产权的管理，并为大多数 Python 发布版本持有相应权利。", ["S02"]),
    _relation("OSR002", "org:psf", "hosts", "service:pypi", "os-doc-01", "包索引基础设施", "PSF 运营 Python Package Index，通常简称 PyPI。", ["S02"]),
    _relation("OSR003", "org:pypa", "maintains", "project:pip", "os-doc-02", "维护范围", "用来接手 pip 和 virtualenv 的维护工作。", ["S03"]),
    _relation("OSR004", "org:pypa", "maintains", "project:virtualenv", "os-doc-02", "维护范围", "用来接手 pip 和 virtualenv 的维护工作。", ["S03"]),
    _relation("OSR005", "project:numpy", "has_repository", "repo:numpy", "os-doc-03", "仓库身份", "NumPy 的公开源代码仓库是 `numpy/numpy`。", ["S04"]),
    _relation("OSR006", "project:numpy", "distributed_under", "license:bsd3", "os-doc-03", "许可证", "GitHub 项目页将 NumPy 标为 BSD-3-Clause。", ["S04"]),
    _relation("OSR007", "service:pypi", "lists", "project:numpy", "os-doc-04", "项目归属", "PyPI 项目页的 owner 显示为 NumPy。", ["S05"]),
    _relation("OSR008", "project:numpy", "has_license_expression", "license:numpy-expression", "os-doc-04", "发行版许可证表达式", "当前快照中的许可证表达式为 `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`。", ["S05"]),
    _relation("OSR009", "project:pandas", "has_repository", "repo:pandas", "os-doc-05", "代码仓库与许可证", "pandas 的源代码仓库是 `pandas-dev/pandas`。", ["S06"]),
    _relation("OSR010", "project:pandas", "distributed_under", "license:bsd3", "os-doc-05", "代码仓库与许可证", "项目构建元数据把许可写为 BSD-3-Clause。", ["S06"]),
    _relation("OSR011", "project:pandas", "documentation_hosted_at", "org:pydata", "os-doc-05", "文档入口", "pandas 的 PyPI 项目页把官方文档入口指向 PyData。", ["S07"]),
    _relation("OSR012", "service:pypi", "lists", "project:pandas", "os-doc-05", "文档入口", "pandas 的 PyPI 项目页把官方文档入口指向 PyData。", ["S07"]),
    _relation("OSR013", "project:scikit-learn", "has_repository", "repo:scikit-learn", "os-doc-06", "代码仓库", "scikit-learn 的公开源代码仓库是 `scikit-learn/scikit-learn`。", ["S08"]),
    _relation("OSR014", "project:scikit-learn", "distributed_under", "license:bsd3", "os-doc-06", "代码仓库", "仓库使用 3-Clause BSD 许可证。", ["S08"]),
    _relation("OSR015", "project:scikit-learn", "uses_import_namespace", "entity:sklearn-namespace", "os-doc-06", "名称与导入空间", "Python 代码中的导入空间使用 `sklearn`。", ["S09"]),
    _relation("OSR016", "service:pypi", "lists", "project:scikit-learn", "os-doc-07", "分发元数据", "PyPI 记录的正式项目名称是 `scikit-learn`。", ["S11"]),
    _relation("OSR017", "project:scikit-learn", "requires", "project:numpy", "os-doc-07", "直接依赖", "该快照列出的核心依赖包括 NumPy 和 SciPy。", ["S11"]),
    _relation("OSR018", "project:scikit-learn", "requires", "project:scipy", "os-doc-07", "直接依赖", "该快照列出的核心依赖包括 NumPy 和 SciPy。", ["S11"]),
    _relation("OSR019", "project:scipy", "has_repository", "repo:scipy", "os-doc-08", "源码与许可证", "SciPy 的公开 GitHub 仓库是 `scipy/scipy`。", ["S12"]),
    _relation("OSR020", "project:scipy", "distributed_under", "license:bsd3", "os-doc-08", "源码与许可证", "项目页标注 BSD-3-Clause。", ["S12"]),
    _relation("OSR021", "project:scipy", "built_on", "project:numpy", "os-doc-08", "科学计算关系", "建立在 NumPy 之上的科学计算库。", ["S13"]),
    _relation("OSR022", "project:scipy", "foundation_for", "project:scikit-learn", "os-doc-08", "科学计算关系", "scikit-learn 等 Python 包的重要基础。", ["S13"]),
    _relation("OSR023", "project:jupyter", "originated_from", "project:ipython", "os-doc-09", "起源", "Project Jupyter 于 2014 年从 IPython 项目演化而来。", ["S14"]),
    _relation("OSR024", "project:jupyter", "distributed_under", "license:bsd3", "os-doc-09", "许可证", "Jupyter 项目代码使用修改版 BSD，也称 BSD-3-Clause。", ["S15"]),
    _relation("OSR025", "project:jupyter", "developed_on", "entity:github", "os-doc-09", "开发方式", "Jupyter 社区在 GitHub 上公开开发。", ["S14"]),
    _relation("OSR026", "org:numfocus", "sponsors", "project:numpy", "os-doc-10", "名单", "NumPy、pandas、SciPy、Project Jupyter 和 scikit-learn 等项目", ["S16"]),
    _relation("OSR027", "org:numfocus", "sponsors", "project:pandas", "os-doc-10", "名单", "pandas、SciPy、Project Jupyter", ["S16"]),
    _relation("OSR028", "org:numfocus", "sponsors", "project:scipy", "os-doc-10", "名单", "SciPy、Project Jupyter", ["S16"]),
    _relation("OSR029", "org:numfocus", "sponsors", "project:jupyter", "os-doc-10", "名单", "Project Jupyter", ["S16"]),
    _relation("OSR030", "org:numfocus", "sponsors", "project:scikit-learn", "os-doc-10", "名单", "scikit-learn 等项目", ["S16"]),
    _relation("OSR031", "org:cncf", "hosts", "project:kubernetes", "os-doc-11", "托管关系", "Kubernetes 列为其托管项目之一", ["S17"]),
    _relation("OSR032", "project:kubernetes", "graduated_project_of", "org:cncf", "os-doc-11", "毕业与治理", "2018 年公告中宣布 Kubernetes 成为其第一个毕业项目", ["S18"]),
    _relation("OSR033", "project:kubernetes", "distributed_under", "license:apache2", "os-doc-11", "许可证与上层组织", "Kubernetes 以 Apache License 2.0 发布", ["S18"]),
    _relation("OSR034", "org:cncf", "part_of", "org:linux-foundation", "os-doc-11", "许可证与上层组织", "CNCF 是 Linux Foundation 的一部分", ["S18"]),
    _relation("OSR035", "project:kubernetes", "has_repository", "repo:kubernetes", "os-doc-12", "仓库身份", "Kubernetes 的主 GitHub 仓库是 `kubernetes/kubernetes`。", ["S20"]),
    _relation("OSR036", "project:kubernetes", "has_short_name", "alias:k8s", "os-doc-12", "简称与治理入口", "Kubernetes 常被简称为 K8s。", ["S20"]),
    _relation("OSR037", "project:kubernetes", "distributed_under", "license:apache2", "os-doc-12", "仓库身份", "Apache-2.0 许可证", ["S20"]),
)


def _case(
    case_id: str,
    category: str,
    question: str,
    expected_answer: str,
    answer_relation_ids: list[str],
    valid_paths: list[list[str]],
    *,
    split: str = "dev",
    semantic_intent: str = "simple",
    outcome: str = "answered",
    answer_variants: list[str] | None = None,
    query_only_terms: list[str] | None = None,
    answer_only_terms: list[str] | None = None,
    negative_kind: str | None = None,
    forbidden_claims: list[str] | None = None,
    absence_evidence_documents: list[str] | None = None,
    absence_scope: str | None = None,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "category": category,
        "split": split,
        "question": question,
        "expected_answer": expected_answer,
        "answerable": outcome == "answered",
        "expected_outcome": outcome,
        "expected_answer_aspects": [
            {
                "aspect_id": "answer",
                "answer_variants": answer_variants or [expected_answer],
            }
        ]
        if outcome == "answered"
        else [],
        "answer_gold_relation_ids": answer_relation_ids,
        "valid_paths": valid_paths,
        "semantic_intent": semantic_intent,
        "route_label_status": "candidate_requires_host_baseline"
        if semantic_intent == "graph"
        else "not_applicable",
        "query_only_terms": query_only_terms or [],
        "answer_only_terms": answer_only_terms or [],
        "negative_control_kind": negative_kind,
        "absence_evidence_documents": absence_evidence_documents or [],
        "absence_scope": absence_scope,
        "forbidden_claims": forbidden_claims or [],
        "notes": notes,
    }


CASES: tuple[dict[str, Any], ...] = (
    _case(
        "graph-001",
        "graph_multi_hop",
        "由 PSF 托管的 Python 包索引里，NumPy 的许可证组合是什么？",
        "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        ["OSR008"],
        [["OSR002", "OSR007", "OSR008"]],
        semantic_intent="graph",
        query_only_terms=["PSF", "Python 包索引"],
        answer_only_terms=["BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0"],
        notes="PSF→PyPI→NumPy→许可证表达式；三跳，答案事实与桥接事实分散。",
    ),
    _case(
        "graph-002",
        "graph_multi_hop",
        "NumFOCUS 支持的项目中，哪个项目的官方文档入口指向 PyData？",
        "pandas",
        ["OSR011"],
        [["OSR027", "OSR011"]],
        semantic_intent="graph",
        query_only_terms=["NumFOCUS"],
        answer_only_terms=["pandas"],
    ),
    _case(
        "graph-003",
        "graph_multi_hop",
        "NumFOCUS 支持的科学计算库所支撑的机器学习项目是什么？",
        "scikit-learn",
        ["OSR022"],
        [["OSR028", "OSR022"]],
        semantic_intent="graph",
        query_only_terms=["NumFOCUS"],
        answer_only_terms=["scikit-learn"],
    ),
    _case(
        "graph-004",
        "graph_multi_hop",
        "CNCF 托管且已毕业的容器项目，其 GitHub 主仓库是什么？",
        "kubernetes/kubernetes",
        ["OSR035"],
        [["OSR031", "OSR035"]],
        semantic_intent="graph",
        query_only_terms=["CNCF", "已毕业", "容器项目"],
        answer_only_terms=["kubernetes/kubernetes"],
    ),
    _case(
        "graph-005",
        "graph_multi_hop",
        "NumFOCUS 支持的交互式计算项目起源于哪个项目？",
        "IPython",
        ["OSR023"],
        [["OSR029", "OSR023"]],
        semantic_intent="graph",
        query_only_terms=["NumFOCUS", "交互式计算项目"],
        answer_only_terms=["IPython"],
    ),
    _case(
        "graph-006",
        "graph_multi_hop",
        "PSF 托管的 Python 包索引中，scikit-learn 依赖的科学计算库是什么？",
        "SciPy",
        ["OSR018"],
        [["OSR002", "OSR016", "OSR018"]],
        semantic_intent="graph",
        query_only_terms=["PSF", "Python 包索引"],
        answer_only_terms=["SciPy"],
    ),
    _case(
        "graph-007",
        "graph_multi_hop",
        "NumFOCUS 支持的基础数组项目，其 GitHub 仓库路径是什么？",
        "numpy/numpy",
        ["OSR005"],
        [["OSR026", "OSR005"]],
        semantic_intent="graph",
        query_only_terms=["NumFOCUS", "基础数组项目"],
        answer_only_terms=["numpy/numpy"],
    ),
    _case(
        "graph-008",
        "graph_multi_hop",
        "PSF 负责的包索引中，pandas 项目的文档站点属于哪个生态组织？",
        "PyData",
        ["OSR011"],
        [["OSR002", "OSR012", "OSR011"]],
        semantic_intent="graph",
        query_only_terms=["PSF", "包索引"],
        answer_only_terms=["PyData"],
    ),
    _case(
        "graph-009",
        "graph_multi_hop",
        "NumFOCUS 支持的科学计算库建立在什么数组库之上？",
        "NumPy",
        ["OSR021"],
        [["OSR028", "OSR021"]],
        semantic_intent="graph",
        query_only_terms=["NumFOCUS"],
        answer_only_terms=["NumPy"],
    ),
    _case(
        "graph-010",
        "graph_multi_hop",
        "由 CNCF 托管的 Kubernetes 项目采用哪种许可证？",
        "Apache-2.0",
        ["OSR037"],
        [["OSR031", "OSR037"]],
        semantic_intent="graph",
        query_only_terms=["CNCF"],
        answer_only_terms=["Apache-2.0"],
    ),
    _case(
        "graph-011",
        "graph_multi_hop",
        "PSF 托管的包索引中，pandas 项目的许可证是什么？",
        "BSD-3-Clause",
        ["OSR010"],
        [["OSR002", "OSR012", "OSR010"]],
        semantic_intent="graph",
        query_only_terms=["PSF", "包索引"],
        answer_only_terms=["BSD-3-Clause"],
    ),
    _case(
        "graph-012",
        "graph_multi_hop",
        "NumFOCUS 支持的机器学习项目依赖哪一个科学计算库？",
        "SciPy",
        ["OSR018"],
        [["OSR030", "OSR018"]],
        split="locked_test",
        semantic_intent="graph",
        query_only_terms=["NumFOCUS"],
        answer_only_terms=["SciPy"],
    ),
    _case("simple-001", "single_hop", "PSF 主要围绕哪一种编程语言开展工作？", "Python", ["OSR001"], [["OSR001"]], answer_only_terms=["Python"]),
    _case("simple-002", "single_hop", "PyPA 接手维护的两个工具是什么？", "pip 和 virtualenv", ["OSR003", "OSR004"], [["OSR003"], ["OSR004"]], answer_variants=["pip 和 virtualenv", "pip、virtualenv"], answer_only_terms=["pip", "virtualenv"]),
    _case("simple-003", "single_hop", "NumPy GitHub 项目页标注的主许可证是什么？", "BSD-3-Clause", ["OSR006"], [["OSR006"]], answer_only_terms=["BSD-3-Clause"]),
    _case("simple-004", "single_hop", "pandas 的官方文档入口指向哪里？", "PyData", ["OSR011"], [["OSR011"]], answer_only_terms=["PyData"]),
    _case("simple-005", "alias_resolution", "scikit-learn 在 Python 中使用的导入空间叫什么？", "sklearn", ["OSR015"], [["OSR015"]], answer_variants=["sklearn", "sklearn import namespace"], answer_only_terms=["sklearn"]),
    _case("simple-006", "single_hop", "Project Jupyter 起源于哪个项目？", "IPython", ["OSR023"], [["OSR023"]], answer_only_terms=["IPython"]),
    _case("simple-007", "single_hop", "Kubernetes 主仓库页面标注的许可证是什么？", "Apache-2.0", ["OSR033"], [["OSR033"]], answer_only_terms=["Apache-2.0"]),
    _case("simple-008", "single_hop_table", "项目目录中 SciPy 的公开仓库路径是什么？", "scipy/scipy", ["OSR019"], [["OSR019"]], answer_only_terms=["scipy/scipy"]),
    _case("negative-001", "negative_control", "本测试集是否记录 Kubernetes 是 NumFOCUS 的赞助项目？", "无充分证据", [], [], outcome="refused", negative_kind="open_world_unanswerable", forbidden_claims=["Kubernetes 是 NumFOCUS 赞助项目", "Kubernetes 不是任何形式的 NumFOCUS 项目"], notes="名单快照不支持把缺席推成绝对否定。"),
    _case("negative-002", "negative_control", "Jupyter 的项目代码是否采用 Apache-2.0？", "不是，资料标注为 BSD-3-Clause", ["OSR024"], [["OSR024"]], negative_kind="contradicted", forbidden_claims=["Jupyter 使用 Apache-2.0"], answer_variants=["不是，资料标注为 BSD-3-Clause", "BSD-3-Clause"]),
    _case("negative-003", "negative_control", "官方 PyPI 项目名是不是 sklearn？", "不是，官方项目名是 scikit-learn", ["OSR016"], [["OSR016"]], negative_kind="contradicted", forbidden_claims=["sklearn 是官方 PyPI 项目名"], answer_variants=["不是，官方项目名是 scikit-learn", "scikit-learn"]),
    _case("negative-004", "negative_control", "NumPy 是否由 CNCF 托管？", "无充分证据", [], [], outcome="refused", negative_kind="open_world_unanswerable", forbidden_claims=["CNCF 托管 NumPy", "PSF 托管 NumPy" ]),
    _case("negative-005", "negative_control", "scikit-image 是否就是 scikit-learn 的另一个名称？", "无充分证据，不能把两个名称合并", [], [], outcome="refused", negative_kind="open_world_unanswerable", forbidden_claims=["scikit-image 等于 scikit-learn"]),
    _case("negative-006", "negative_control", "SciPy 是否托管 NumPy 的 GitHub 仓库？", "无充分证据", [], [], outcome="refused", negative_kind="open_world_unanswerable", forbidden_claims=["SciPy 托管 NumPy 仓库", "NumPy 仓库是 scipy/scipy"]),
    _case("negative-007", "negative_control", "Kubernetes 是否使用 BSD-3-Clause？", "不是，资料标注为 Apache-2.0", ["OSR033"], [["OSR033"]], negative_kind="contradicted", forbidden_claims=["Kubernetes 使用 BSD-3-Clause"], answer_variants=["不是，资料标注为 Apache-2.0", "Apache-2.0"]),
    _case("negative-008", "negative_control", "项目目录是否包含名为 TensorFlow 的项目？", "没有，目录快照中未列出该项目", [], [], negative_kind="closed_world_absence", answer_variants=["没有，目录快照中未列出该项目", "未列出 TensorFlow"], absence_evidence_documents=["os-doc-13"], absence_scope="project_catalog_complete_snapshot", notes="只在声明为完整的 derived snapshot table 上使用 closed-world 语义。"),
)


SOURCES: tuple[dict[str, Any], ...] = (
    {"source_id": "S01", "platform": "python.org", "title": "About the Python Software Foundation", "url": "https://www.python.org/psf/about/", "license_note": "本集仅使用事实摘要，不复制网页长段落。"},
    {"source_id": "S02", "platform": "python.org", "title": "PSF Mission", "url": "https://www.python.org/psf/mission/", "license_note": "本集仅使用事实摘要，不复制网页长段落。"},
    {"source_id": "S03", "platform": "pypa.io", "title": "Packaging History", "url": "https://www.pypa.io/en/latest/history/", "license_note": "本集仅使用事实摘要，不复制网页长段落。"},
    {"source_id": "S04", "platform": "GitHub", "title": "numpy/numpy", "url": "https://github.com/numpy/numpy", "license_note": "仓库页面标明 BSD-3-Clause；本集不再分发仓库代码。"},
    {"source_id": "S05", "platform": "PyPI", "title": "numpy project metadata", "url": "https://pypi.org/project/numpy/", "license_note": "使用 PyPI 页面中的公开元数据事实；版本字段按访问日期冻结。"},
    {"source_id": "S06", "platform": "GitHub", "title": "pandas pyproject.toml", "url": "https://github.com/pandas-dev/pandas/blob/main/pyproject.toml", "license_note": "仓库页面标明 BSD-3-Clause；本集不再分发仓库代码。"},
    {"source_id": "S07", "platform": "PyPI", "title": "pandas project", "url": "https://pypi.org/project/pandas/", "license_note": "使用 PyPI 页面中的项目链接和公开元数据事实。"},
    {"source_id": "S08", "platform": "GitHub", "title": "scikit-learn/scikit-learn", "url": "https://github.com/scikit-learn/scikit-learn", "license_note": "仓库页面标明 BSD-3-Clause；本集不再分发仓库代码。"},
    {"source_id": "S09", "platform": "GitHub", "title": "scikit-learn sklearn namespace", "url": "https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/__init__.py", "license_note": "仅使用导入空间这一事实摘要。"},
    {"source_id": "S10", "platform": "GitHub", "title": "sklearn PyPI package clarification", "url": "https://github.com/scikit-learn/sklearn-pypi-package", "license_note": "仅使用项目名与导入名区别的事实摘要。"},
    {"source_id": "S11", "platform": "PyPI", "title": "scikit-learn project", "url": "https://pypi.org/project/scikit-learn/", "license_note": "使用 PyPI 页面中的依赖和许可证元数据事实。"},
    {"source_id": "S12", "platform": "GitHub", "title": "scipy/scipy", "url": "https://github.com/scipy/scipy", "license_note": "仓库页面标明 BSD-3-Clause；本集不再分发仓库代码。"},
    {"source_id": "S13", "platform": "NumFOCUS", "title": "SciPy project", "url": "https://numfocus.org/project/scipy", "license_note": "本集使用项目关系的事实摘要，不复制页面长段落。"},
    {"source_id": "S14", "platform": "Jupyter", "title": "Project Jupyter About Us", "url": "https://jupyter.org/about", "license_note": "本集使用事实摘要，不复制网页长段落。"},
    {"source_id": "S15", "platform": "Jupyter", "title": "Project Jupyter licensing terms", "url": "https://jupyter.org/governance/projectlicense/", "license_note": "仅使用许可证名称和关系事实，不复制完整许可证文本。"},
    {"source_id": "S16", "platform": "NumFOCUS", "title": "Sponsored Projects", "url": "https://numfocus.org/sponsored-projects", "license_note": "名单是访问日快照；不据缺席推导绝对否定。"},
    {"source_id": "S17", "platform": "CNCF", "title": "CNCF home", "url": "https://www.cncf.io/", "license_note": "本集使用托管关系的事实摘要。"},
    {"source_id": "S18", "platform": "CNCF", "title": "Kubernetes first graduated project", "url": "https://www.cncf.io/announcements/2018/03/06/cncf-announces-kubernetes-first-graduated-project/", "license_note": "本集使用历史事件和许可证事实摘要。"},
    {"source_id": "S19", "platform": "CNCF", "title": "CNCF governance overview", "url": "https://contribute.cncf.io/community/governance/", "license_note": "本集使用治理关系的事实摘要。"},
    {"source_id": "S20", "platform": "GitHub", "title": "kubernetes/kubernetes", "url": "https://github.com/kubernetes/kubernetes", "license_note": "仓库页面标明 Apache-2.0；本集不再分发仓库代码。"},
)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _manifest(output: Path) -> dict[str, Any]:
    docs = [
        {
            "document_id": row["document_id"],
            "filename": row["filename"],
            "format": row["format"],
            "material_type": row["material_type"],
            "source_ids": row["source_ids"],
            "sha256": hashlib.sha256(row["content"].encode("utf-8")).hexdigest(),
        }
        for row in DOCUMENTS
    ]
    return {
        "schema": SCHEMA,
        "dataset_id": DATASET_ID,
        "language": "zh-CN with canonical English project names",
        "synthetic": False,
        "derived_summary": True,
        "retrieved_at": ACCESS_DATE,
        "source_count": len(SOURCES),
        "document_count": len(DOCUMENTS),
        "relation_count": len(RELATIONS),
        "case_count": len(CASES),
        "case_counts": {
            "graph_candidate": sum(case["semantic_intent"] == "graph" for case in CASES),
            "simple_answerable": sum(
                case["semantic_intent"] == "simple" and case["category"] != "negative_control"
                for case in CASES
            ),
            "negative_control": sum(case["category"] == "negative_control" for case in CASES),
        },
        "semantic_intent_counts": {
            "graph": sum(case["semantic_intent"] == "graph" for case in CASES),
            "simple": sum(case["semantic_intent"] == "simple" for case in CASES),
        },
        "format_counts": {
            extension: sum(row["format"] == extension for row in DOCUMENTS)
            for extension in sorted({row["format"] for row in DOCUMENTS})
        },
        "route_label_policy": {
            "static_field": "semantic_intent",
            "graph_status": "candidate_requires_host_baseline",
            "primary_metric_ready": False,
            "reason": "Simple baseline and Graph benefit have not been run against this frozen corpus.",
        },
        "documents": docs,
        "artifacts": {
            "entities": "entities.jsonl",
            "relations": "relations.jsonl",
            "cases": "cases.jsonl",
            "sources": "SOURCE_NOTICES.md",
        },
    }


def _read_documents(output: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in DOCUMENTS:
        path = output / "documents" / row["filename"]
        if not path.exists():
            raise ValueError(f"missing document {row['filename']}")
        result[row["document_id"]] = path.read_text(encoding="utf-8")
    return result


def _validate_csv_snapshot(document_text: dict[str, str]) -> None:
    csv_text = document_text["os-doc-13"]
    rows = list(csv.reader(io.StringIO(csv_text)))
    if len(rows) != 6 or len(rows[0]) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError("project catalog CSV must have one header and five four-column rows")
    if rows[0] != ["项目", "生态类别", "公开仓库", "主许可证"]:
        raise ValueError("project catalog CSV header changed")


def validate(output: Path) -> None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_id") != DATASET_ID:
        raise ValueError("unexpected dataset id")
    document_text = _read_documents(output)
    _validate_csv_snapshot(document_text)
    document_ids = {row["document_id"] for row in DOCUMENTS}
    entity_ids = {row["entity_id"] for row in ENTITIES}
    relations_by_id = {row["relation_id"]: row for row in RELATIONS}
    if len(relations_by_id) != len(RELATIONS):
        raise ValueError("duplicate relation ids")
    for entity in ENTITIES:
        if not entity["canonical_name"] or not isinstance(entity["aliases"], list):
            raise ValueError(f"invalid entity {entity['entity_id']}")
    for relation in RELATIONS:
        if relation["document_id"] not in document_ids:
            raise ValueError(f"unknown relation document {relation['relation_id']}")
        if relation["subject_entity_id"] not in entity_ids or relation["object_entity_id"] not in entity_ids:
            raise ValueError(f"unknown relation endpoint {relation['relation_id']}")
        if relation["subject_entity_id"] == relation["object_entity_id"]:
            raise ValueError(f"self-loop relation is not allowed {relation['relation_id']}")
        if relation["relation_id"] in document_text[relation["document_id"]]:
            raise ValueError(f"relation id leaked into document {relation['relation_id']}")
        evidence = relation["evidence_text"]
        evidence_without_terminal_punctuation = evidence.rstrip("。.!！")
        if (
            evidence not in document_text[relation["document_id"]]
            and evidence_without_terminal_punctuation
            not in document_text[relation["document_id"]]
        ):
            raise ValueError(f"relation evidence missing from document {relation['relation_id']}")
        section_title = relation["source_locator"]["section_title"]
        if section_title not in document_text[relation["document_id"]]:
            raise ValueError(f"relation section missing {relation['relation_id']}")

    cases_path = output / "cases.jsonl"
    cases = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(cases) != len(CASES):
        raise ValueError("case count mismatch")
    case_ids: set[str] = set()
    negative_kinds: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise ValueError(f"duplicate or missing case id {case_id}")
        case_ids.add(case_id)
        outcome = case.get("expected_outcome")
        if outcome not in {"answered", "refused"}:
            raise ValueError(f"invalid outcome {case_id}")
        if case.get("answerable") != (outcome == "answered"):
            raise ValueError(f"answerable mismatch {case_id}")
        negative_kind = case.get("negative_control_kind")
        if case["category"] == "negative_control":
            if negative_kind not in {"contradicted", "closed_world_absence", "open_world_unanswerable"}:
                raise ValueError(f"invalid negative control {case_id}")
            negative_kinds.add(negative_kind)
            if negative_kind == "closed_world_absence":
                absence_docs = case.get("absence_evidence_documents")
                if not case.get("absence_scope") or not isinstance(absence_docs, list) or not absence_docs:
                    raise ValueError(f"closed-world case needs an explicit complete-snapshot scope {case_id}")
                if any(doc_id not in document_ids for doc_id in absence_docs):
                    raise ValueError(f"closed-world case references an unknown document {case_id}")
        elif negative_kind is not None:
            raise ValueError(f"non-negative case has negative kind {case_id}")
        paths = case.get("valid_paths")
        if not isinstance(paths, list):
            raise ValueError(f"missing valid paths {case_id}")
        if case["semantic_intent"] == "graph":
            if not paths or any(len(path) < 2 for path in paths):
                raise ValueError(f"Graph case needs at least a two-edge path {case_id}")
            answer_ids = set(case["answer_gold_relation_ids"])
            if not answer_ids:
                raise ValueError(f"Graph case needs answer relations {case_id}")
            for path in paths:
                if not set(path).issuperset(answer_ids):
                    raise ValueError(f"path does not cover answer relations {case_id}")
                for relation_id in path:
                    if relation_id not in relations_by_id:
                        raise ValueError(f"unknown path relation {case_id}:{relation_id}")
                for left_id, right_id in zip(path, path[1:]):
                    left = relations_by_id[left_id]
                    right = relations_by_id[right_id]
                    if left["object_entity_id"] != right["subject_entity_id"]:
                        raise ValueError(f"path is not connected {case_id}:{left_id}->{right_id}")
            for term in case.get("query_only_terms", []):
                if not isinstance(term, str) or not term.strip():
                    raise ValueError(f"invalid query-only term {case_id}")
            for relation_id in answer_ids:
                doc = relations_by_id[relation_id]["document_id"]
                if any(term in document_text[doc] for term in case.get("query_only_terms", [])):
                    raise ValueError(f"answer document contains query-only term {case_id}")
        for relation_id in case.get("answer_gold_relation_ids", []):
            if relation_id not in relations_by_id:
                raise ValueError(f"unknown answer relation {case_id}:{relation_id}")
        if case["category"] == "negative_control" and outcome == "refused" and case.get("answer_gold_relation_ids"):
            raise ValueError(f"refused case cannot have answer gold {case_id}")

    if negative_kinds != {"contradicted", "closed_world_absence", "open_world_unanswerable"}:
        raise ValueError("negative controls must cover all semantic kinds")
    if manifest.get("case_count") != len(cases) or manifest.get("relation_count") != len(RELATIONS):
        raise ValueError("manifest count mismatch")
    print(
        f"validated {DATASET_ID}: {len(DOCUMENTS)} documents, "
        f"{len(RELATIONS)} relations, {len(cases)} cases; "
        "Graph labels remain candidate until host baseline verification"
    )


def _write_readme(output: Path) -> None:
    (output / "README.md").write_text(
        _clean(
            f"""
            # routing-rag-v3-open-source

            这是一个基于公开项目元数据和官方页面事实摘要的 RAG/Graph 评测语料，快照日期为 {ACCESS_DATE}。文档正文是重新组织的短摘要，不复制来源页面的长段落；来源、访问日期和许可证说明见 `SOURCE_NOTICES.md`。

            ## 语料构成

            - 14 份可上传文档，包含 Markdown、纯文本和 CSV；主题覆盖 Python/科学计算生态以及 CNCF/Kubernetes 独立网络。
            - `entities.jsonl` 和 `relations.jsonl` 是关系抽取与路径金标准，不上传到知识库。
            - `cases.jsonl` 包含 12 个跨文档 Graph 候选、8 个单跳/别名/表格题和 8 个拒答或矛盾控制题。
            - Graph case 同时保留 `semantic_intent=graph` 与 `route_label_status=candidate_requires_host_baseline`。在冻结的 Simple 配置和明确的 Graph benefit 运行完成前，不能把它们当作已验证的路由主指标。

            ## 校验

            ```bash
            PYTHONPATH=src:. .venv/bin/python tools/build_open_source_rag_corpus.py build --force
            PYTHONPATH=src:. .venv/bin/python tools/build_open_source_rag_corpus.py validate
            ```

            builder 是离线脚本，不访问网络，不调用模型，不启动容器。运行主模型基线时应使用隔离的、已明确提供的 host 依赖；不可把个人 `rag` 数据库或 Docker 容器作为测试前提。

            ## 评测前置条件

            1. 只摄取 `documents/` 下的文件。
            2. 用 semantic v4 解析和固定索引配置完成一次 Simple baseline；记录每个 Graph case 是否覆盖所有 `valid_paths` 中的一条路径。
            3. 再执行 Graph replay，只有 source-backed 新答案证据补齐路径时才记录 Graph benefit。
            4. 将 baseline 配置 hash 写入独立的 locked evaluation manifest，不修改本目录中的事实 gold。

            ## Locked evaluation 合同

            `tools/evaluate_open_source_rag_v3.py` 是纯离线锁定与计分器。`--dry-run` 校验语料身份和冻结配置；`--write-observation-template` 生成待 host runner 填充的观察合同；`--observations ... --output ...` 只接受覆盖全部 case、绑定明确 runtime/index/build 身份的完整观察，并生成独立 locked manifest。

            冻结配置固定 semantic v4、text-only embedding、Simple exact-vector/classic `top_k=10`、Graph `adaptive_graphiti_v2`/`graphiti_v3`、classic rerank 和 `edge_limit=8`。观察文件分别保存 full Graph 与 incremental packed；locked manifest 以 Simple 是否完整覆盖任一 `valid_paths` 动态生成 Graph-needed 标签，分别报告 Simple-only、full Graph-only、Simple ∪ Graph augmented、Graph incremental、Graph benefit、Auto TP/FP、类型化关系抽取对齐和负例拒答；不会把 `semantic_intent=graph` 直接提升为 route gold，也不会把 augmented recall 标为 Graph-only recall。

            观察模板本身不是已完成基线，也不能作为质量结果。填充它需要外部 Provider 或隔离数据库/Graph 时，必须先满足仓库的 host-only 和授权边界；依赖缺失时保持未验证，不能用 Docker 补环境。

            `tools/run_open_source_rag_v3.py` 是对应的 host observation runner。它只读取已经绑定到 v3 KB/index/Graph build 的 owner-only `rag-eval` runtime，不创建 KB、不管理容器；逐 case 原子保存 content-safe checkpoint，完成后直接生成 immutable locked manifest。真实执行会调用 Provider，必须显式传入 `--confirm RUN_ROUTING_RAG_V3_EXTERNAL_CALLS`；`--dry-run` 仍保持纯离线。

            已运行且身份匹配的 `rag-eval` 可通过现有 provisioner 的 v3 冻结 spec 建立并绑定 KB：`--dataset routing-rag-v3-open-source --confirm PROVISION_ROUTING_RAG_V3_OPEN_SOURCE`。该入口会产生索引与 Graph Provider 流量，不属于测试准备步骤；没有单独授权时不得执行。
            """
        ),
        encoding="utf-8",
    )


def _write_sources(output: Path) -> None:
    lines = [
        f"# SOURCE_NOTICES（访问日期：{ACCESS_DATE}）",
        "",
        "本目录中的 documents/ 是基于公开来源重新撰写的事实摘要，不复制来源页面的长篇原文或仓库代码。每条 relation 通过 source_ids 追溯到下表。GitHub 仓库的许可证只适用于相应仓库内容；本测试集仅保留仓库名、许可证标签和事实性摘要。",
        "",
        "| source_id | 平台 | 来源 | URL | 处理说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for source in SOURCES:
        lines.append(
            f"| {source['source_id']} | {source['platform']} | {source['title']} | {source['url']} | {source['license_note']} |"
        )
    lines.extend(
        [
            "",
            "## 许可和时间边界",
            "",
            "这是一个来源快照，不承诺网页未来内容不变。对 PyPI 版本、项目名单和依赖字段，答案必须结合本目录的 retrieved_at 日期。若要重新抓取，应生成新的数据集身份和新的来源清单，不能覆盖本快照。",
            "",
            "本集没有把来源网页、许可证全文、README 长文或仓库代码作为上传语料；如果后续需要收录原文，必须单独核验原作者许可和再分发边界。",
        ]
    )
    (output / "SOURCE_NOTICES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(output: Path, *, force: bool = False) -> None:
    if output.exists():
        if not force:
            raise ValueError(f"output exists; pass --force to regenerate: {output}")
        if output == ROOT or output == ROOT / "evaluation":
            raise ValueError("refusing to replace a broad repository directory")
    output.mkdir(parents=True, exist_ok=True)
    documents_dir = output / "documents"
    documents_dir.mkdir(parents=True, exist_ok=True)
    for row in DOCUMENTS:
        (documents_dir / row["filename"]).write_text(row["content"], encoding="utf-8")
    _write_jsonl(output / "entities.jsonl", ENTITIES)
    _write_jsonl(output / "relations.jsonl", RELATIONS)
    _write_jsonl(output / "cases.jsonl", CASES)
    (output / "manifest.json").write_text(
        json.dumps(_manifest(output), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_readme(output)
    _write_sources(output)
    validate(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.command == "build":
        build(args.output, force=args.force)
    else:
        validate(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
