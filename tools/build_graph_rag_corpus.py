#!/usr/bin/env python3
"""Build and validate the deterministic Graph RAG hard-case corpus.

The corpus is deliberately synthetic.  Its facts are short, explicit
sentences spread over multiple narrative documents so that an evaluator can
compare lexical/hybrid retrieval with an entity-path traversal without relying
on a network download or a provider response.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Iterable

from rag_kb.domain import (
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
    ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
    GRAPH_EXTRACTOR_VERSION,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "graph-rag-v1"
CORPUS_SCHEMA = "graph_rag_corpus_v1"
CASE_SCHEMA = "graph_rag_case_v1"
TARGET_SECTIONS_PER_DOCUMENT = 14


@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: tuple[str, ...] = ()
    alias_for: str | None = None


@dataclass(frozen=True)
class Edge:
    relation_id: str
    document_id: str
    subject: str
    predicate: str
    object: str
    subject_surface: str | None = None
    object_surface: str | None = None
    note: str = ""


@dataclass(frozen=True)
class DocumentSpec:
    document_id: str
    filename: str
    title: str
    format: str


DOCUMENTS = (
    DocumentSpec("doc01", "01_group_governance.md", "霁岳集团治理与股权叙事", "md"),
    DocumentSpec("doc02", "02_subsidiary_registry.md", "子公司登记与业务隶属", "md"),
    DocumentSpec("doc03", "03_people_and_roles.md", "人员任职与法定代表人备案", "md"),
    DocumentSpec("doc04", "04_acquisitions.md", "并购、改名与历史沿革", "md"),
    DocumentSpec("doc05", "05_products_and_business.md", "产品线与研发归属", "md"),
    DocumentSpec("doc06", "06_partnerships.md", "合作、签约与供应关系", "md"),
    DocumentSpec("doc07", "07_research_and_certifications.md", "研究机构与技术许可", "md"),
    DocumentSpec("doc08", "08_regional_projects.md", "区域项目与落地城市", "md"),
    DocumentSpec("doc09", "09_supply_chain.txt", "供应链运营记录", "txt"),
    DocumentSpec("doc10", "10_aliases_and_disambiguation.txt", "别名、旧称与同名消歧", "txt"),
    DocumentSpec("doc11", "11_board_and_investment.md", "董事会决议与投资网络", "md"),
    DocumentSpec("doc12", "12_project_contracts.md", "项目承建与合同履约", "md"),
    DocumentSpec("doc13", "13_regional_branches.md", "区域分支与物流枢纽", "md"),
    DocumentSpec("doc14", "14_technology_roadmap.txt", "技术路线与产品认证", "txt"),
    DocumentSpec("doc15", "15_noise_vendor_notes.md", "无关供应商网络档案", "md"),
    DocumentSpec("doc16", "16_noise_consumer_brand.txt", "无关消费品牌周报", "txt"),
)


BASE_ENTITIES = (
    Entity("jiyue_holdings", "霁岳产业控股有限公司", "company", ("霁岳控股", "霁岳集团")),
    Entity("chenyue_precision", "辰岳精密有限公司", "company", ("辰岳精密",)),
    Entity("xinglan_manufacturing", "星澜智造有限公司", "company", ("星澜制造",)),
    Entity("yunxiu_digital", "云岫数字科技有限公司", "company", ("云岫数科",)),
    Entity("chenghai_energy", "澄海能源设备有限公司", "company", ("澄海能源",)),
    Entity("yuanxi_supply", "远汐供应链有限公司", "company", ("远汐物流",)),
    Entity("ju_chuan_robot", "炬川机器人有限公司", "company", ("炬川机器人",)),
    Entity("qingyu_storage", "青屿储能有限公司", "company", ("青屿储能",)),
    Entity("lanqiao_data", "岚桥数据服务有限公司", "company", ("岚桥数据",)),
    Entity("chaoxi_coldchain", "潮汐冷链有限公司", "company", ("潮汐冷链",)),
    Entity("wutong_chip", "梧桐芯片有限公司", "company", ("梧桐芯片",)),
    Entity("beichen_precision", "北辰精工有限公司", "company", ("北辰精工",)),
    Entity("beichen_energy", "北辰能源有限公司", "company", ("北辰能源",)),
    Entity("chenyue_old", "辰岳实业（旧称）", "alias", ("辰岳实业",), "chenyue_precision"),
    Entity("xinglan_factory_alias", "星澜工厂", "alias", (), "xinglan_manufacturing"),
    Entity("jiyue_group_alias", "霁岳集团", "alias", (), "jiyue_holdings"),
    Entity("yunxiu_alias", "云岫数科", "alias", (), "yunxiu_digital"),
    Entity("chenghai_alias", "澄海能源", "alias", (), "chenghai_energy"),
    Entity("beichen_legacy", "北辰精工（前称）", "alias", ("北辰精工旧名",), "beichen_precision"),
    Entity("han_qiming", "韩启明", "person", ()),
    Entity("su_wanzhou", "苏晚舟", "person", ()),
    Entity("liang_qinghe", "梁青禾", "person", ()),
    Entity("zhou_jibai", "周既白", "person", ()),
    Entity("shen_yan", "沈砚", "person", ()),
    Entity("xuan_yu_array", "玄羽阵列", "product", ()),
    Entity("xing_gui_controller", "星轨控制器", "product", ()),
    Entity("wu_lan_platform", "雾岚平台", "product", ()),
    Entity("qingyu_bms", "青屿储能管理模块", "product", ()),
    Entity("wutong_chiplet", "梧桐芯片模组", "product", ("WTC-7",)),
    Entity("nanqi_research", "南栖研究院", "institute", ()),
    Entity("cangyuan_university", "沧渊大学", "institute", ()),
    Entity("beicen_lab", "北岑实验室", "institute", ()),
    Entity("yuelu_city", "岳麓市", "place", ()),
    Entity("yuchuan_city", "榆川市", "place", ()),
    Entity("jialan_city", "嘉澜市", "place", ()),
    Entity("hailan_city", "海岚市", "place", ()),
    Entity("haidung_park", "海东零碳园区", "project", ()),
    Entity("qingyu_station", "青屿储能示范站", "project", ()),
    Entity("yuanxi_hub", "远汐冷链枢纽", "project", ()),
    Entity("gu_nanqiao", "顾南乔", "person", ()),
    Entity("tang_ming", "唐铭", "person", ()),
    Entity("hezhou_innovation", "合舟创新有限公司", "company", ("合舟创新",)),
    Entity("baizhi_fund", "百栀产业基金", "fund", ()),
    Entity("muyu_capital", "木语资本管理有限公司", "company", ("木语资本",)),
    Entity("hezhou_old", "合舟智造（旧称）", "alias", ("合舟智造",), "hezhou_innovation"),
    Entity("luming_project", "鹿鸣工业互联网园", "project", ()),
    Entity("rongcheng_city", "容城市", "place", ()),
    Entity("xunhai_logistics", "巡海物流有限公司", "company", ("巡海物流",)),
    Entity("xunhai_hub", "巡海冷链枢纽", "project", ()),
    Entity("suyin_energy", "素隐能源有限公司", "company", ("素隐能源",)),
    Entity("yuhe_platform", "玉衡工业平台", "product", ()),
    Entity("suyin_cell", "素隐电芯", "product", ()),
    Entity("yulu_power", "云鹭电力设备有限公司", "company", ("云鹭电力",)),
    Entity("donggang_storage", "东港储能科技有限公司", "company", ("东港储能",)),
    Entity("qingshan_microgrid", "青山微网项目", "project", ()),
    Entity("donggang_city", "东港市", "place", ()),
    Entity("mingzhou_logistics", "明舟物流有限公司", "company", ("明舟物流",)),
    Entity("yulu_dispatch", "云鹭调度系统", "product", ()),
    Entity("luanxing_retail", "鸾星零售有限公司", "company", ("鸾星零售",)),
    Entity("ximu_commerce", "西木商业管理有限公司", "company", ("西木商业",)),
    Entity("qingshan_mall", "青山生活广场", "project", ()),
    Entity("luohe_city", "洛河市", "place", ()),
    Entity("lanhu_membership", "蓝狐会员系统", "product", ()),
    Entity("beichen_energy_east", "北辰能源（东港）有限公司", "company", ("北辰能源东港分部",)),
)


def _edge(
    relation_id: str,
    document_id: str,
    subject: str,
    predicate: str,
    object_: str,
    *,
    subject_surface: str | None = None,
    object_surface: str | None = None,
    note: str = "",
) -> Edge:
    return Edge(
        relation_id,
        document_id,
        subject,
        predicate,
        object_,
        subject_surface,
        object_surface,
        note,
    )


# Five primary facts per document form the reusable cross-document paths.
CORE_EDGES = (
    _edge("R001", "doc01", "jiyue_holdings", "控股", "chenyue_precision", subject_surface="霁岳集团"),
    _edge("R002", "doc01", "jiyue_holdings", "全资设立", "yuanxi_supply", subject_surface="霁岳控股", object_surface="远汐物流"),
    _edge("R003", "doc01", "jiyue_holdings", "设立", "yunxiu_digital", subject_surface="霁岳产业控股有限公司", object_surface="云岫数科"),
    _edge("R004", "doc01", "jiyue_holdings", "控股", "chenghai_energy", subject_surface="霁岳集团", object_surface="澄海能源"),
    _edge("R005", "doc01", "jiyue_holdings", "注册地位于", "yuelu_city"),

    _edge("R006", "doc02", "chenyue_precision", "控股", "xinglan_factory_alias", subject_surface="辰岳精密有限公司", object_surface="星澜工厂"),
    _edge("R007", "doc02", "yunxiu_digital", "参与设立", "lanqiao_data", subject_surface="云岫数字科技有限公司", object_surface="岚桥数据服务有限公司"),
    _edge("R008", "doc02", "chenghai_energy", "控股", "qingyu_storage", subject_surface="澄海能源设备有限公司", object_surface="青屿储能有限公司"),
    _edge("R009", "doc02", "yuanxi_supply", "运营", "chaoxi_coldchain", subject_surface="远汐物流", object_surface="潮汐冷链有限公司"),
    _edge("R010", "doc04", "beichen_precision", "控股", "wutong_chip", subject_surface="北辰精工有限公司", object_surface="梧桐芯片有限公司"),

    _edge("R011", "doc03", "han_qiming", "担任法定代表人", "chenyue_precision"),
    _edge("R012", "doc03", "su_wanzhou", "担任首席执行官", "xinglan_manufacturing"),
    _edge("R013", "doc03", "liang_qinghe", "担任首席技术官", "yunxiu_digital"),
    _edge("R014", "doc03", "zhou_jibai", "担任董事长", "chenghai_energy"),
    _edge("R015", "doc03", "shen_yan", "担任总经理", "yuanxi_supply"),

    _edge("R016", "doc04", "chenyue_old", "更名为", "chenyue_precision", object_surface="辰岳精密有限公司"),
    _edge("R017", "doc04", "chenyue_precision", "收购", "beichen_precision"),
    _edge("R018", "doc04", "beichen_energy", "独立于", "beichen_precision"),
    _edge("R019", "doc04", "jiyue_holdings", "并入", "lanqiao_data"),
    _edge("R020", "doc04", "qingyu_storage", "承接", "qingyu_station"),

    _edge("R021", "doc05", "xuan_yu_array", "产品归属", "xinglan_manufacturing", object_surface="星澜制造"),
    _edge("R022", "doc05", "xing_gui_controller", "产品归属", "chenghai_energy"),
    _edge("R023", "doc05", "wu_lan_platform", "产品归属", "yunxiu_digital"),
    _edge("R024", "doc05", "qingyu_bms", "产品归属", "qingyu_storage"),
    _edge("R025", "doc05", "wutong_chiplet", "产品归属", "wutong_chip", subject_surface="WTC-7"),

    _edge("R026", "doc06", "xinglan_manufacturing", "合作", "ju_chuan_robot"),
    _edge("R027", "doc06", "ju_chuan_robot", "合作", "qingyu_storage"),
    _edge("R028", "doc07", "chenghai_energy", "签约", "lanqiao_data"),
    _edge("R029", "doc06", "yuanxi_supply", "服务", "chaoxi_coldchain"),
    _edge("R030", "doc06", "wutong_chip", "合作", "yunxiu_digital"),

    _edge("R031", "doc07", "wutong_chip", "获得专利许可于", "nanqi_research"),
    _edge("R032", "doc07", "nanqi_research", "隶属", "cangyuan_university"),
    _edge("R033", "doc07", "yunxiu_digital", "共建", "beicen_lab"),
    _edge("R034", "doc08", "beicen_lab", "位于", "jialan_city"),
    _edge("R035", "doc07", "xinglan_manufacturing", "研发", "xing_gui_controller"),

    _edge("R036", "doc08", "xinglan_manufacturing", "注册地位于", "yuchuan_city"),
    _edge("R037", "doc08", "haidung_park", "使用", "wu_lan_platform"),
    _edge("R038", "doc08", "haidung_park", "归属", "jialan_city"),
    _edge("R039", "doc08", "qingyu_station", "由...运营", "qingyu_storage"),
    _edge("R040", "doc08", "yuanxi_hub", "位于", "hailan_city"),

    _edge("R041", "doc09", "yuanxi_supply", "服务", "haidung_park"),
    _edge("R042", "doc06", "chaoxi_coldchain", "使用", "wu_lan_platform"),
    _edge("R043", "doc09", "wutong_chip", "提供模组给", "xinglan_manufacturing"),
    _edge("R044", "doc09", "qingyu_storage", "采购电芯自", "chenghai_energy"),
    _edge("R045", "doc09", "lanqiao_data", "提供接口给", "ju_chuan_robot"),

    _edge("R046", "doc10", "jiyue_group_alias", "别名为", "jiyue_holdings", subject_surface="霁岳集团", object_surface="霁岳产业控股有限公司"),
    _edge("R047", "doc10", "yunxiu_alias", "别名为", "yunxiu_digital", subject_surface="云岫数科", object_surface="云岫数字科技有限公司"),
    _edge("R048", "doc10", "xinglan_factory_alias", "别名为", "xinglan_manufacturing", subject_surface="星澜工厂", object_surface="星澜智造有限公司"),
    _edge("R049", "doc10", "chenghai_alias", "别名为", "chenghai_energy", subject_surface="澄海能源", object_surface="澄海能源设备有限公司"),
    _edge("R050", "doc10", "beichen_legacy", "更名为", "beichen_precision", subject_surface="北辰精工（前称）", object_surface="北辰精工有限公司"),

    _edge("R051", "doc11", "jiyue_holdings", "委派", "gu_nanqiao"),
    _edge("R052", "doc11", "gu_nanqiao", "担任董事长", "hezhou_innovation"),
    _edge("R053", "doc11", "jiyue_holdings", "设立", "hezhou_innovation"),
    _edge("R054", "doc11", "muyu_capital", "参股", "hezhou_innovation", subject_surface="木语资本"),
    _edge("R055", "doc11", "baizhi_fund", "投资", "hezhou_innovation"),

    _edge("R056", "doc12", "hezhou_innovation", "承建", "luming_project"),
    _edge("R057", "doc12", "luming_project", "位于", "rongcheng_city"),
    _edge("R058", "doc12", "yunxiu_digital", "提供平台给", "luming_project"),
    _edge("R059", "doc12", "hezhou_innovation", "合作", "xunhai_logistics"),
    _edge("R060", "doc12", "xunhai_logistics", "签署服务合同", "luming_project"),
    _edge("R063", "doc12", "hezhou_innovation", "控股", "suyin_energy"),

    _edge("R061", "doc13", "xunhai_logistics", "运营", "xunhai_hub"),
    _edge("R064", "doc13", "suyin_energy", "供应电芯给", "qingyu_storage"),
    _edge("R065", "doc13", "beicen_lab", "支持建设", "luming_project"),
    _edge("R067", "doc11", "hezhou_innovation", "研发", "yuhe_platform"),

    _edge("R062", "doc14", "xunhai_hub", "位于", "rongcheng_city"),
    _edge("R066", "doc14", "hezhou_old", "更名为", "hezhou_innovation", object_surface="合舟创新有限公司"),
    _edge("R068", "doc14", "yuhe_platform", "部署于", "luming_project"),
    _edge("R069", "doc14", "suyin_cell", "产品归属", "suyin_energy"),
    _edge("R070", "doc14", "yuhe_platform", "获得认证于", "cangyuan_university"),

    # Deliberately disconnected noise network A.
    _edge("R071", "doc15", "yulu_power", "控股", "donggang_storage"),
    _edge("R072", "doc15", "donggang_storage", "运营", "qingshan_microgrid"),
    _edge("R073", "doc15", "qingshan_microgrid", "位于", "donggang_city"),
    _edge("R074", "doc15", "mingzhou_logistics", "服务", "qingshan_microgrid"),
    _edge("R075", "doc15", "yulu_dispatch", "产品归属", "yulu_power"),

    # Deliberately disconnected noise network B, including a near-name entity.
    _edge("R076", "doc16", "luanxing_retail", "合作", "ximu_commerce"),
    _edge("R077", "doc16", "ximu_commerce", "运营", "qingshan_mall"),
    _edge("R078", "doc16", "qingshan_mall", "位于", "luohe_city"),
    _edge("R079", "doc16", "lanhu_membership", "产品归属", "ximu_commerce"),
    _edge("R080", "doc16", "beichen_energy_east", "供应设备给", "donggang_storage"),
)


FILLER_NAMES = {
    "doc01": ("霁岳审计委员会", "霁岳战略办公室", "霁岳投资档案", "霁岳风控中心", "霁岳产业园", "霁岳合规专班", "霁岳年度会议", "霁岳预算会", "霁岳档案库", "霁岳授权台账", "霁岳内控台"),
    "doc02": ("辰岳东区工厂", "云岫交付中心", "澄海装配基地", "远汐调度中心", "北辰材料基地", "青屿试制线", "岚桥接入站", "辰岳质量室", "云岫运维组", "子公司备案台", "业务排班室"),
    "doc03": ("治理委员会", "经营管理部", "技术委员会", "法务联络组", "财务共享中心", "人事备案室", "董事会办公室", "任职审阅组", "授权签字库", "干部轮岗表", "人员档案台"),
    "doc04": ("收购整合组", "历史档案室", "资产交割专班", "品牌迁移组", "法务复核组", "股权登记室", "旧名清理台账", "并购评估室", "交割凭证库", "沿革核验组", "交易复盘台"),
    "doc05": ("玄羽测试线", "星轨验证台", "雾岚运营台", "青屿电控仓", "梧桐封装线", "产品认证组", "研发样机库", "版本发布室", "产品安全组", "试验数据台", "产品归档室"),
    "doc06": ("机器人联合项目", "储能协作组", "数据服务合同", "冷链服务单", "芯片联合实验", "采购协同台", "客户联络室", "供应商会签台", "交付验收组", "联合排产室", "合作复盘室"),
    "doc07": ("南栖许可档案", "沧渊联合课题", "北岑试验台", "技术转移室", "专利复审组", "学术合作部", "认证资料库", "许可续期室", "研究伦理组", "成果转化台", "研究档案室"),
    "doc08": ("海东一期工程", "青屿示范线", "远汐干线项目", "榆川制造园", "嘉澜数字园", "海岚储运区", "区域协调组", "园区招商室", "城市服务台", "工程验收组", "项目审计室"),
    "doc09": ("远汐干线", "潮汐分拨站", "梧桐交付批次", "青屿采购单", "岚桥接口单", "海东仓配区", "供应商评审组", "冷链调度台", "芯片收货室", "物流结算组", "运输风控室"),
    "doc10": ("旧称索引一", "简称索引二", "同名排除项", "别名复核单", "历史证券简称", "主体消歧组", "登记口径说明", "旧证照目录", "名称变更台", "主体校验库", "历史核验台"),
    "doc11": ("董事会秘书处", "投资审议会", "百栀出资档案", "木语投后组", "合舟筹建组", "授权决议库", "战略投资台", "关联交易室", "基金合规组", "董事签章库", "投后报告室"),
    "doc12": ("鹿鸣施工组", "容城项目办", "合同履约台", "平台交付室", "巡海签约组", "园区验收组", "承建质量台", "里程碑档案", "项目法务室", "现场调度组", "合同复核室"),
    "doc13": ("巡海调度中心", "冷链枢纽站", "素隐供应组", "青屿电芯仓", "北岑支持台", "区域售后组", "仓储盘点室", "干线排班台", "能源联络组", "物流安全组", "区域结算室"),
    "doc14": ("玉衡研发台", "合舟旧名库", "容城部署组", "素隐电芯线", "认证申报室", "平台兼容组", "产品路线会", "技术白皮书库", "软件验收台", "认证复核组", "技术交付室"),
    "doc15": ("云鹭采购室", "东港电站台账", "青山施工组", "明舟调度台", "云鹭测试线", "供应商准入室", "电力安全组", "微网验收台", "设备保养库", "区域服务组", "合同归档室"),
    "doc16": ("鸾星营销组", "西木招商台", "青山商场档案", "洛河运营组", "蓝狐客服台", "品牌活动室", "会员数据室", "门店巡检组", "消费研究台", "商业合同库", "周报核验室"),
}


FILLER_PREDICATES = ("负责", "覆盖", "配套", "登记于", "服务于", "纳入", "支撑")


def _all_entities() -> dict[str, Entity]:
    entities = {entity.entity_id: entity for entity in BASE_ENTITIES}
    for document_id, names in FILLER_NAMES.items():
        for index, name in enumerate(names, start=1):
            entity_id = f"{document_id}_unit_{index:02d}"
            entities[entity_id] = Entity(entity_id, name, "unit")
    return entities


def _core_by_document() -> dict[str, list[Edge]]:
    result = {document.document_id: [] for document in DOCUMENTS}
    for edge in CORE_EDGES:
        result[edge.document_id].append(edge)
    return result


def _make_edges() -> tuple[Edge, ...]:
    """Return core edges plus enough explicit narrative relations per document."""

    entities = _all_entities()
    by_doc = _core_by_document()
    filler_id = 1
    for document in DOCUMENTS:
        core = by_doc[document.document_id]
        core_count = len(core)
        anchor = core[0].subject if core else f"{document.document_id}_unit_01"
        names = FILLER_NAMES[document.document_id]
        while len(by_doc[document.document_id]) < TARGET_SECTIONS_PER_DOCUMENT:
            index = len(by_doc[document.document_id]) - core_count
            target_id = f"{document.document_id}_unit_{index + 1:02d}"
            predicate = FILLER_PREDICATES[(filler_id - 1) % len(FILLER_PREDICATES)]
            subject_id = anchor
            if index and index % 3 == 0:
                subject_id = f"{document.document_id}_unit_{index:02d}"
            by_doc[document.document_id].append(
                _edge(
                    f"F{filler_id:03d}",
                    document.document_id,
                    subject_id,
                    predicate,
                    target_id,
                    note=f"背景叙事关系 {names[index]}",
                )
            )
            filler_id += 1
    return tuple(edge for document in DOCUMENTS for edge in by_doc[document.document_id])


EDGES = _make_edges()
EDGE_BY_ID = {edge.relation_id: edge for edge in EDGES}
ENTITY_BY_ID = _all_entities()


def _surface(entity_id: str, override: str | None = None) -> str:
    return override or ENTITY_BY_ID[entity_id].canonical_name


def _render_edge(edge: Edge) -> str:
    subject = _surface(edge.subject, edge.subject_surface)
    object_ = _surface(edge.object, edge.object_surface)
    if edge.predicate in {"担任法定代表人", "担任首席执行官", "担任首席技术官", "担任董事长", "担任总经理"}:
        sentence = f"任职备案明确记载，{subject}{edge.predicate}{object_}。"
    elif edge.predicate == "注册地位于":
        sentence = f"登记叙事明确记载，{subject}{edge.predicate}{object_}。"
    elif edge.predicate == "位于":
        sentence = f"项目档案明确记载，{subject}{edge.predicate}{object_}。"
    else:
        sentence = f"业务档案明确记载，{subject}{edge.predicate}{object_}。"
    context = (
        "该事实记录在集团档案的叙事章节中，业务人员先说明关系发生的背景，"
        "再说明它对后续协作、权责或产品流转的影响。档案中的时间线以连续文字保存，"
        "相邻章节会把同一网络的其他节点拆开记录。"
        "本段只保留可以由文字直接核对的主体、关系和客体，不依赖表格编号或外部常识；"
        "因此解析器可以把它作为完整证据单元，图抽取器也能从同一段找到两个实体和明确谓词。"
    )
    if edge.note:
        context += f"本段属于{edge.note}，用于保持同一网络中的实体密度。"
    return f"{sentence}{context}"


def _section_text(document: DocumentSpec, index: int, edge: Edge) -> str:
    body = _render_edge(edge)
    if document.format == "md":
        return f"### {document.document_id}-sec-{index:02d} · {edge.relation_id}\n\n{body}"
    return f"SECTION {document.document_id}-sec-{index:02d} [{edge.relation_id}]\n{body}\n"


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _case(
    case_id: str,
    category: str,
    question: str,
    expected_answer: str,
    *,
    seed_entity_id: str,
    path: tuple[str, ...],
    path_entities: tuple[str, ...],
    answer_relation_ids: tuple[str, ...],
    query_only_terms: tuple[str, ...],
    answer_only_terms: tuple[str, ...],
    answer_document_id: str | None,
    current_graph_support: str,
    answerable: bool = True,
    notes: str = "",
) -> dict[str, object]:
    return {
        "schema": CASE_SCHEMA,
        "case_id": case_id,
        "category": category,
        "question": question,
        "expected_answer": expected_answer,
        "answerable": answerable,
        "seed_entity_id": seed_entity_id,
        "seed_surface_terms": list(query_only_terms),
        "query_only_terms": list(query_only_terms),
        "answer_only_terms": list(answer_only_terms),
        "gold_path": list(path),
        "path_entities": list(path_entities),
        "required_hops": len(path),
        "answer_relation_ids": list(answer_relation_ids),
        "answer_document_id": answer_document_id,
        "answer_chunk_must_not_contain_query_terms": True,
        "current_graph_support": current_graph_support,
        "hybrid_expectation": "likely_miss" if answerable and category != "negative_control" else "control",
        "notes": notes,
    }


CASES = (
    _case(
        "graph-001", "graph_only_hard",
        "星澜工厂的控股方安排了哪位人士担任辰岳精密的法定代表人？", "韩启明",
        seed_entity_id="xinglan_factory_alias", path=("R006", "R011"),
        path_entities=("xinglan_factory_alias", "chenyue_precision", "han_qiming"),
        answer_relation_ids=("R011",), query_only_terms=("星澜工厂",), answer_only_terms=("韩启明",),
        answer_document_id="doc03", current_graph_support="1_3_hop",
        notes="起点使用别名，答案段只出现桥接企业和负责人。",
    ),
    _case(
        "graph-002", "graph_only_hard",
        "玄羽阵列所属制造企业的注册地在哪座城市？", "榆川市",
        seed_entity_id="xuan_yu_array", path=("R021", "R036"),
        path_entities=("xuan_yu_array", "xinglan_manufacturing", "yuchuan_city"),
        answer_relation_ids=("R036",), query_only_terms=("玄羽阵列",), answer_only_terms=("榆川市",),
        answer_document_id="doc08", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-003", "graph_only_hard",
        "云岫数字科技有限公司共建的实验室落在哪座城市？", "嘉澜市",
        seed_entity_id="yunxiu_digital", path=("R033", "R034"),
        path_entities=("yunxiu_digital", "beicen_lab", "jialan_city"),
        answer_relation_ids=("R034",), query_only_terms=("云岫数字科技有限公司",), answer_only_terms=("嘉澜市",),
        answer_document_id="doc08", current_graph_support="1_3_hop",
    ),
    _case(
        "alias-001", "alias_resolution",
        "辰岳实业（旧称）对应公司的法定代表人是谁？", "韩启明",
        seed_entity_id="chenyue_old", path=("R016", "R011"),
        path_entities=("chenyue_old", "chenyue_precision", "han_qiming"),
        answer_relation_ids=("R011",), query_only_terms=("辰岳实业（旧称）",), answer_only_terms=("韩启明",),
        answer_document_id="doc03", current_graph_support="1_3_hop",
        notes="别名与规范名分处历史沿革和任职文档。",
    ),
    _case(
        "graph-004", "graph_only_hard",
        "星轨控制器所属企业的董事长姓名是什么？", "周既白",
        seed_entity_id="xing_gui_controller", path=("R022", "R014"),
        path_entities=("xing_gui_controller", "chenghai_energy", "zhou_jibai"),
        answer_relation_ids=("R014",), query_only_terms=("星轨控制器",), answer_only_terms=("周既白",),
        answer_document_id="doc03", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-005", "graph_only_hard",
        "远汐供应链有限公司服务的海东园区归属哪座城市？", "嘉澜市",
        seed_entity_id="yuanxi_supply", path=("R041", "R038"),
        path_entities=("yuanxi_supply", "haidung_park", "jialan_city"),
        answer_relation_ids=("R038",), query_only_terms=("远汐供应链有限公司",), answer_only_terms=("嘉澜市",),
        answer_document_id="doc08", current_graph_support="1_3_hop",
    ),
    _case(
        "alias-002", "alias_resolution",
        "云岫数科的技术负责人是谁？", "梁青禾",
        seed_entity_id="yunxiu_alias", path=("R047", "R013"),
        path_entities=("yunxiu_alias", "yunxiu_digital", "liang_qinghe"),
        answer_relation_ids=("R013",), query_only_terms=("云岫数科",), answer_only_terms=("梁青禾",),
        answer_document_id="doc03", current_graph_support="1_3_hop",
    ),
    _case(
        "alias-003", "alias_resolution",
        "星澜工厂合作的机器人企业叫什么？", "炬川机器人有限公司",
        seed_entity_id="xinglan_factory_alias", path=("R048", "R026"),
        path_entities=("xinglan_factory_alias", "xinglan_manufacturing", "ju_chuan_robot"),
        answer_relation_ids=("R026",), query_only_terms=("星澜工厂",), answer_only_terms=("炬川机器人有限公司",),
        answer_document_id="doc06", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-006", "graph_only_hard",
        "代号 WTC-7 的芯片模组交付给哪家制造商？", "星澜智造有限公司",
        seed_entity_id="wutong_chiplet", path=("R025", "R043"),
        path_entities=("wutong_chiplet", "wutong_chip", "xinglan_manufacturing"),
        answer_relation_ids=("R043",), query_only_terms=("WTC-7",), answer_only_terms=("星澜智造有限公司",),
        answer_document_id="doc09", current_graph_support="1_3_hop",
        notes="答案段只写芯片公司，不重复问题中的产品代号。",
    ),
    _case(
        "graph-007", "graph_only_hard",
        "炬川机器人合作的储能企业，其控股方叫什么？", "澄海能源设备有限公司",
        seed_entity_id="ju_chuan_robot", path=("R027", "R008"),
        path_entities=("ju_chuan_robot", "qingyu_storage", "chenghai_energy"),
        answer_relation_ids=("R008",), query_only_terms=("炬川机器人",), answer_only_terms=("澄海能源设备有限公司",),
        answer_document_id="doc02", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-008", "graph_only_hard",
        "雾岚平台所属数科参与设立的服务商是谁？", "岚桥数据服务有限公司",
        seed_entity_id="wu_lan_platform", path=("R023", "R007"),
        path_entities=("wu_lan_platform", "yunxiu_digital", "lanqiao_data"),
        answer_relation_ids=("R007",), query_only_terms=("雾岚平台",), answer_only_terms=("岚桥数据服务有限公司",),
        answer_document_id="doc02", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-009", "graph_only_hard",
        "潮汐冷链背后的物流运营方负责哪个园区？", "海东零碳园区",
        seed_entity_id="chaoxi_coldchain", path=("R009", "R041"),
        path_entities=("chaoxi_coldchain", "yuanxi_supply", "haidung_park"),
        answer_relation_ids=("R041",), query_only_terms=("潮汐冷链",), answer_only_terms=("海东零碳园区",),
        answer_document_id="doc09", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-3hop-001", "graph_3hop",
        "霁岳集团投资体系中的实验室叫什么？", "北岑实验室",
        seed_entity_id="jiyue_group_alias", path=("R046", "R003", "R033"),
        path_entities=("jiyue_group_alias", "jiyue_holdings", "yunxiu_digital", "beicen_lab"),
        answer_relation_ids=("R033",), query_only_terms=("霁岳集团",), answer_only_terms=("北岑实验室",),
        answer_document_id="doc07", current_graph_support="1_3_hop",
        notes="用于验收当前 3-hop Graph 路径能力。",
    ),
    _case(
        "graph-3hop-002", "graph_3hop",
        "星澜工厂沿并购链最终关联到哪家芯片企业？", "梧桐芯片有限公司",
        seed_entity_id="xinglan_factory_alias", path=("R006", "R017", "R010"),
        path_entities=("xinglan_factory_alias", "chenyue_precision", "beichen_precision", "wutong_chip"),
        answer_relation_ids=("R010",), query_only_terms=("星澜工厂",), answer_only_terms=("梧桐芯片有限公司",),
        answer_document_id="doc04", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-3hop-003", "graph_3hop",
        "北辰精工（前称）获得许可的研究机构是哪一家？", "南栖研究院",
        seed_entity_id="beichen_legacy", path=("R050", "R010", "R031"),
        path_entities=("beichen_legacy", "beichen_precision", "wutong_chip", "nanqi_research"),
        answer_relation_ids=("R031",), query_only_terms=("北辰精工（前称）",), answer_only_terms=("南栖研究院",),
        answer_document_id="doc07", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-3hop-004", "graph_3hop",
        "炬川机器人合作的储能企业，其控股方签约了哪家数据服务商？", "岚桥数据服务有限公司",
        seed_entity_id="ju_chuan_robot", path=("R027", "R008", "R028"),
        path_entities=("ju_chuan_robot", "qingyu_storage", "chenghai_energy", "lanqiao_data"),
        answer_relation_ids=("R028",), query_only_terms=("炬川机器人",), answer_only_terms=("岚桥数据服务有限公司",),
        answer_document_id="doc07", current_graph_support="1_3_hop",
        notes="这是 3 条边的压力样例，服务必须提供完整路径而非静默截断。",
    ),
    _case(
        "graph-010", "graph_only_hard",
        "合舟创新有限公司合作的物流枢纽叫什么？", "巡海冷链枢纽",
        seed_entity_id="hezhou_innovation", path=("R059", "R061"),
        path_entities=("hezhou_innovation", "xunhai_logistics", "xunhai_hub"),
        answer_relation_ids=("R061",), query_only_terms=("合舟创新有限公司",), answer_only_terms=("巡海冷链枢纽",),
        answer_document_id="doc13", current_graph_support="1_3_hop",
        notes="新增文档组的跨项目桥接样例，答案文档不出现合舟主体。",
    ),
    _case(
        "graph-011", "graph_only_hard",
        "木语资本参股的企业承建了哪个园区？", "鹿鸣工业互联网园",
        seed_entity_id="muyu_capital", path=("R054", "R056"),
        path_entities=("muyu_capital", "hezhou_innovation", "luming_project"),
        answer_relation_ids=("R056",), query_only_terms=("木语资本",), answer_only_terms=("鹿鸣工业互联网园",),
        answer_document_id="doc12", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-012", "graph_only_hard",
        "玉衡工业平台部署的园区位于哪座城市？", "容城市",
        seed_entity_id="yuhe_platform", path=("R068", "R057"),
        path_entities=("yuhe_platform", "luming_project", "rongcheng_city"),
        answer_relation_ids=("R057",), query_only_terms=("玉衡工业平台",), answer_only_terms=("容城市",),
        answer_document_id="doc12", current_graph_support="1_3_hop",
    ),
    _case(
        "alias-004", "alias_resolution",
        "合舟智造（旧称）对应公司的董事长是谁？", "顾南乔",
        seed_entity_id="hezhou_old", path=("R066", "R052"),
        path_entities=("hezhou_old", "hezhou_innovation", "gu_nanqiao"),
        answer_relation_ids=("R052",), query_only_terms=("合舟智造（旧称）",), answer_only_terms=("顾南乔",),
        answer_document_id="doc11", current_graph_support="1_3_hop",
        notes="新增历史名称与董事会文档之间的别名消歧样例。",
    ),
    _case(
        "graph-013", "graph_only_hard",
        "巡海物流负责的枢纽位于哪座城市？", "容城市",
        seed_entity_id="xunhai_logistics", path=("R061", "R062"),
        path_entities=("xunhai_logistics", "xunhai_hub", "rongcheng_city"),
        answer_relation_ids=("R062",), query_only_terms=("巡海物流",), answer_only_terms=("容城市",),
        answer_document_id="doc14", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-014", "graph_only_hard",
        "百栀产业基金投资的企业承建了哪个园区？", "鹿鸣工业互联网园",
        seed_entity_id="baizhi_fund", path=("R055", "R056"),
        path_entities=("baizhi_fund", "hezhou_innovation", "luming_project"),
        answer_relation_ids=("R056",), query_only_terms=("百栀产业基金",), answer_only_terms=("鹿鸣工业互联网园",),
        answer_document_id="doc12", current_graph_support="1_3_hop",
    ),
    _case(
        "alias-005", "alias_resolution",
        "合舟智造（旧称）研发的工业平台叫什么？", "玉衡工业平台",
        seed_entity_id="hezhou_old", path=("R066", "R067"),
        path_entities=("hezhou_old", "hezhou_innovation", "yuhe_platform"),
        answer_relation_ids=("R067",), query_only_terms=("合舟智造（旧称）",), answer_only_terms=("玉衡工业平台",),
        answer_document_id="doc11", current_graph_support="1_3_hop",
    ),
    _case(
        "graph-015", "graph_only_hard",
        "素隐电芯所属企业向哪家储能公司供应电芯？", "青屿储能有限公司",
        seed_entity_id="suyin_cell", path=("R069", "R064"),
        path_entities=("suyin_cell", "suyin_energy", "qingyu_storage"),
        answer_relation_ids=("R064",), query_only_terms=("素隐电芯",), answer_only_terms=("青屿储能有限公司",),
        answer_document_id="doc13", current_graph_support="1_3_hop",
    ),
    _case(
        "negative-001", "negative_control",
        "星澜工厂是否直接控股南栖研究院？", "无直接关系证据",
        seed_entity_id="xinglan_factory_alias", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("星澜工厂", "直接控股"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
    ),
    _case(
        "negative-002", "negative_control",
        "北辰能源是否被北辰精工收购？", "无直接关系证据",
        seed_entity_id="beichen_energy", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("北辰能源", "被北辰精工收购"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
        notes="同名主体为独立公司，不能把名称相似误当成收购边。",
    ),
    _case(
        "negative-003", "negative_control",
        "玄羽阵列是否由潮汐冷链研发？", "无直接关系证据",
        seed_entity_id="xuan_yu_array", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("玄羽阵列", "潮汐冷链研发"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
    ),
    _case(
        "negative-004", "negative_control",
        "合舟创新有限公司是否直接控股鹿鸣工业互联网园？", "无直接关系证据",
        seed_entity_id="hezhou_innovation", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("合舟创新有限公司", "直接控股鹿鸣工业互联网园"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
        notes="存在承建边但不存在控股边，检查谓词不能被图连通性替代。",
    ),
    _case(
        "negative-005", "negative_control",
        "北辰能源有限公司是否控股东港储能科技有限公司？", "无直接关系证据",
        seed_entity_id="beichen_energy", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("北辰能源有限公司", "东港储能科技有限公司"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
        notes="噪声文档中的近似主体不能与主网络的北辰能源合并。",
    ),
    _case(
        "negative-006", "negative_control",
        "玄羽阵列是否由云鹭电力设备有限公司提供？", "无直接关系证据",
        seed_entity_id="xuan_yu_array", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("玄羽阵列", "云鹭电力设备有限公司"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
        notes="噪声企业拥有自己的产品与供应链，和霁岳产品网络没有边。",
    ),
    _case(
        "negative-007", "negative_control",
        "霁岳集团是否与鸾星零售有限公司存在合作？", "无直接关系证据",
        seed_entity_id="jiyue_holdings", path=(), path_entities=(), answer_relation_ids=(),
        query_only_terms=("霁岳集团", "鸾星零售有限公司"), answer_only_terms=("无直接关系证据",),
        answer_document_id=None, current_graph_support="control", answerable=False,
        notes="测试两个彼此独立的企业网络不会因通用业务词产生虚假路径。",
    ),
)


def _document_edges() -> dict[str, tuple[Edge, ...]]:
    result: dict[str, list[Edge]] = {document.document_id: [] for document in DOCUMENTS}
    for edge in EDGES:
        result[edge.document_id].append(edge)
    return {key: tuple(value) for key, value in result.items()}


def build(output: Path, *, force: bool = False) -> None:
    if output.exists() and force:
        for child in output.iterdir():
            if child.name in {".gitkeep", "README.md"}:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    documents_dir = output / "documents"
    documents_dir.mkdir(exist_ok=True)
    by_doc = _document_edges()
    document_rows: list[dict[str, object]] = []
    section_lookup: dict[str, dict[str, object]] = {}
    for document in DOCUMENTS:
        edges = by_doc[document.document_id]
        sections = []
        for index, edge in enumerate(edges, start=1):
            section_id = f"{document.document_id}-sec-{index:02d}"
            text = _section_text(document, index, edge)
            sections.append(text)
            section_lookup[edge.relation_id] = {
                "document_id": document.document_id,
                "section_id": section_id,
                "text": text,
            }
        path = documents_dir / document.filename
        path.write_text(
            (f"# {document.title}\n\n" if document.format == "md" else f"{document.title}\n\n")
            + "\n\n".join(sections),
            encoding="utf-8",
        )
        document_rows.append(
            {
                "document_id": document.document_id,
                "filename": document.filename,
                "format": document.format,
                "title": document.title,
                "logical_section_count": len(edges),
                "relation_ids": [edge.relation_id for edge in edges],
                "narrative_only": True,
            }
        )

    entity_rows = []
    for entity in ENTITY_BY_ID.values():
        row = {
            "entity_id": entity.entity_id,
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "aliases": list(entity.aliases),
        }
        if entity.alias_for:
            row["alias_for"] = entity.alias_for
        entity_rows.append(row)

    relation_rows = []
    for edge in EDGES:
        section = section_lookup[edge.relation_id]
        relation_rows.append(
            {
                "relation_id": edge.relation_id,
                "subject_entity_id": edge.subject,
                "predicate": edge.predicate,
                "object_entity_id": edge.object,
                "document_id": edge.document_id,
                "section_id": section["section_id"],
                "subject_surface": _surface(edge.subject, edge.subject_surface),
                "object_surface": _surface(edge.object, edge.object_surface),
                "explicit_text": section["text"],
                "is_filler": edge.relation_id.startswith("F"),
            }
        )

    _write_jsonl(output / "entities.jsonl", entity_rows)
    _write_jsonl(output / "relations.jsonl", relation_rows)
    _write_jsonl(output / "cases.jsonl", CASES)
    manifest = {
        "schema": CORPUS_SCHEMA,
        "dataset_id": "graph-rag-v1",
        "language": "zh-CN",
        "synthetic": True,
        "recommended_schema_profile": {
            "key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
            "digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
            "extractor_version": GRAPH_EXTRACTOR_VERSION,
        },
        "document_count": len(DOCUMENTS),
        "logical_section_count": len(EDGES),
        "entity_count": len(entity_rows),
        "relation_count": len(relation_rows),
        "case_count": len(CASES),
        "noise_document_count": 2,
        "noise_relation_count": 10,
        "supported_case_count": sum(case["current_graph_support"] == "1_3_hop" for case in CASES),
        "stretch_case_count": 0,
        "negative_control_count": sum(case["category"] == "negative_control" for case in CASES),
        "documents": document_rows,
        "properties": {
            "narrative_not_tables": True,
            "explicit_relations": True,
            "cross_document_answer_chunks": True,
            "query_anchor_forbidden_in_answer_chunk": True,
            "aliases_and_historical_names": True,
            "disconnected_noise_networks": True,
            "near_name_distractors": True,
        },
        "current_graph_contract": {
            "expected_online_max_hops": 3,
            "graph_path_kind": "GRAPH_PATH",
            "answer_evidence_source": "raw_chunk",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"built {manifest['dataset_id']}: {len(DOCUMENTS)} documents, "
        f"{len(EDGES)} logical sections, {len(CASES)} cases at {output}"
    )


def _case_path_valid(case: dict[str, object]) -> list[str]:
    errors: list[str] = []
    path = tuple(case["gold_path"])
    nodes = tuple(case["path_entities"])
    if len(path) != case["required_hops"]:
        errors.append("required_hops does not equal path length")
    if path and len(nodes) != len(path) + 1:
        errors.append("path_entities must have one more node than edges")
    for index, relation_id in enumerate(path):
        edge = EDGE_BY_ID.get(relation_id)
        if edge is None:
            errors.append(f"unknown relation {relation_id}")
            continue
        if index + 1 >= len(nodes):
            continue
        pair = {nodes[index], nodes[index + 1]}
        if pair != {edge.subject, edge.object}:
            errors.append(f"relation {relation_id} does not connect path nodes")
    return errors


def validate(output: Path) -> int:
    errors: list[str] = []
    manifest_path = output / "manifest.json"
    cases_path = output / "cases.jsonl"
    relations_path = output / "relations.jsonl"
    if not manifest_path.exists() or not cases_path.exists() or not relations_path.exists():
        print(f"missing generated corpus files under {output}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CORPUS_SCHEMA:
        errors.append("manifest schema mismatch")
    if manifest.get("recommended_schema_profile") != {
        "key": ENTERPRISE_GRAPH_SCHEMA_PROFILE_KEY,
        "digest": ENTERPRISE_GRAPH_SCHEMA_PROFILE_DIGEST,
        "extractor_version": GRAPH_EXTRACTOR_VERSION,
    }:
        errors.append("enterprise schema profile identity mismatch")
    expected_relation_count = len(DOCUMENTS) * TARGET_SECTIONS_PER_DOCUMENT
    if manifest.get("logical_section_count") != expected_relation_count:
        errors.append(f"expected exactly {expected_relation_count} logical sections")
    if manifest.get("document_count") != len(DOCUMENTS):
        errors.append("document count mismatch")
    relation_rows = [json.loads(line) for line in relations_path.read_text(encoding="utf-8").splitlines() if line]
    relation_ids = {row["relation_id"] for row in relation_rows}
    if len(relation_rows) != expected_relation_count or len(relation_ids) != expected_relation_count:
        errors.append(
            f"relation manifest must contain {expected_relation_count} unique relations"
        )
    for row in relation_rows:
        text = str(row["explicit_text"])
        for field in ("subject_surface", "predicate", "object_surface"):
            value = str(row[field])
            if value not in text:
                errors.append(f"{row['relation_id']}: {field} is not explicit in source text")
    for document in DOCUMENTS:
        path = output / "documents" / document.filename
        if not path.exists():
            errors.append(f"missing document {document.filename}")
            continue
        text = path.read_text(encoding="utf-8")
        if "|" in text or "\t" in text:
            errors.append(f"table-like delimiter found in {document.filename}")
        if len(re.findall(r"(?:^### |^SECTION )", text, flags=re.MULTILINE)) != TARGET_SECTIONS_PER_DOCUMENT:
            errors.append(
                f"{document.filename} does not contain {TARGET_SECTIONS_PER_DOCUMENT} logical sections"
            )
    case_rows = [json.loads(line) for line in cases_path.read_text(encoding="utf-8").splitlines() if line]
    if len(case_rows) != len(CASES):
        errors.append("case count mismatch")
    for case in case_rows:
        errors.extend(f"{case['case_id']}: {error}" for error in _case_path_valid(case))
        for relation_id in case["gold_path"]:
            if relation_id not in relation_ids:
                errors.append(f"{case['case_id']}: missing relation {relation_id}")
        answer_document_id = case.get("answer_document_id")
        if not case["answerable"]:
            if case["gold_path"] or answer_document_id is not None:
                errors.append(f"{case['case_id']}: negative control has a gold answer path")
            continue
        if answer_document_id is None:
            errors.append(f"{case['case_id']}: answerable case has no answer document")
            continue
        answer_document = next(
            (document for document in DOCUMENTS if document.document_id == answer_document_id),
            None,
        )
        if answer_document is None:
            errors.append(f"{case['case_id']}: unknown answer document {answer_document_id}")
            continue
        answer_document_text = (
            output / "documents" / answer_document.filename
        ).read_text(encoding="utf-8")
        for term in case["query_only_terms"]:
            if term in answer_document_text:
                errors.append(
                    f"{case['case_id']}: query-only term {term!r} leaked into answer document"
                )
        seed_edge = EDGE_BY_ID[case["gold_path"][0]]
        seed_document = next(
            document for document in DOCUMENTS if document.document_id == seed_edge.document_id
        )
        seed_document_text = (
            output / "documents" / seed_document.filename
        ).read_text(encoding="utf-8")
        for term in case["query_only_terms"]:
            if term not in seed_document_text:
                errors.append(
                    f"{case['case_id']}: query-only term {term!r} is absent from seed document"
                )
        answer_relations = [EDGE_BY_ID[relation_id] for relation_id in case["answer_relation_ids"]]
        if any(edge.document_id != answer_document_id for edge in answer_relations):
            errors.append(f"{case['case_id']}: answer relation is not in answer document")
        answer_text = "\n".join(
            str(row["explicit_text"])
            for row in relation_rows
            if row["relation_id"] in set(case["answer_relation_ids"])
        )
        question = str(case["question"])
        for term in case["query_only_terms"]:
            if term in answer_text:
                errors.append(f"{case['case_id']}: query-only term {term!r} leaked into answer chunk")
        for term in case["answer_only_terms"]:
            if term not in answer_text:
                errors.append(f"{case['case_id']}: answer-only term {term!r} missing from answer chunk")
            if term in question:
                errors.append(f"{case['case_id']}: answer-only term {term!r} appears in question")
        if case["required_hops"] >= 1 and case["answer_document_id"] == EDGE_BY_ID[case["gold_path"][0]].document_id:
            errors.append(f"{case['case_id']}: seed and answer documents must differ")
    if errors:
        print("validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    supported = sum(case["current_graph_support"] == "1_3_hop" for case in case_rows)
    stretch = sum(case["category"] == "graph_3hop" for case in case_rows)
    negatives = sum(case["category"] == "negative_control" for case in case_rows)
    print(
        f"validated graph-rag-v1: {len(DOCUMENTS)} docs, "
        f"{len(DOCUMENTS) * TARGET_SECTIONS_PER_DOCUMENT} logical sections, "
        f"{len(case_rows)} cases ({supported} current 1-3 hop, {stretch} 3-hop, {negatives} controls)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="replace generated files under output")
    args = parser.parse_args()
    if args.command == "build":
        build(args.output, force=args.force)
        return validate(args.output)
    return validate(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
