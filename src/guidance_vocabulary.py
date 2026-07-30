"""Jurisdiction-aware vocabulary for management-guidance retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from src.ticker_policy import get_ticker_suffix


@dataclass(frozen=True)
class GuidanceLocalePolicy:
    results_terms: str
    bridge_terms: str
    excerpt_priority_terms: tuple[str, ...]
    transient_tax_terms: tuple[str, ...]


_ENGLISH = GuidanceLocalePolicy(
    results_terms=(
        "latest results release investor presentation guidance revenue "
        "operating profit pretax profit net income"
    ),
    bridge_terms=(
        "earnings call transcript Q&A guidance income tax tax credit subsidy "
        "regulatory accounting change net income"
    ),
    excerpt_priority_terms=(
        "wage-increase tax credit",
        "tax credit",
        "tax incentive",
        "deferred tax",
        "net income",
        "operating profit",
    ),
    transient_tax_terms=(
        "tax credit",
        "tax incentive",
        "tax benefit",
        "wage-increase tax credit",
    ),
)

_POLICIES: dict[str, GuidanceLocalePolicy] = {
    ".T": GuidanceLocalePolicy(
        results_terms="最新 決算短信 決算説明資料 業績予想 売上高 営業利益 経常利益 当期純利益",
        bridge_terms=(
            "決算説明会 質疑応答 書き起こし 業績予想 営業利益 当期純利益 "
            "法人税 税額控除 税負担 賃上げ促進税制 当期純利益"
        ),
        excerpt_priority_terms=(
            "賃上げ促進税制",
            "税額控除",
            "法人税等調整額",
            "当期純利益",
            "営業利益",
        ),
        transient_tax_terms=("税額控除", "賃上げ促進税制", "税制優遇"),
    ),
    ".KS": GuidanceLocalePolicy(
        results_terms="최신 실적발표 자료 사업보고서 가이던스 매출 영업이익 당기순이익",
        bridge_terms=(
            "컨퍼런스콜 질의응답 법인세 세액공제 보조금 규제 변경 "
            "당기순이익 가이던스"
        ),
        excerpt_priority_terms=("세액공제", "법인세", "당기순이익", "영업이익"),
        transient_tax_terms=("세액공제", "조세혜택"),
    ),
    ".HK": GuidanceLocalePolicy(
        results_terms="最新 年報 業績公告 業績說明會 財測 營業收入 營業利益 稅前利益 淨利",
        bridge_terms="業績說明會 問答 法人稅 稅收抵免 補助金 監管變更 淨利 財測",
        excerpt_priority_terms=("稅收抵免", "法人稅", "淨利", "營業利益"),
        transient_tax_terms=("稅收抵免",),
    ),
    ".SS": GuidanceLocalePolicy(
        results_terms="最新 年报 业绩公告 业绩说明会 业绩预告 营业收入 营业利润 税前利润 净利润",
        bridge_terms="业绩说明会 问答 企业所得税 税收抵免 政府补助 监管变化 净利润 业绩预告",
        excerpt_priority_terms=("税收抵免", "企业所得税", "净利润", "营业利润"),
        transient_tax_terms=("税收抵免",),
    ),
    ".WA": GuidanceLocalePolicy(
        results_terms=(
            "najnowszy raport okresowy wyniki finansowe prezentacja inwestorska "
            "prognoza przychody zysk operacyjny zysk netto"
        ),
        bridge_terms=(
            "konferencja wynikowa transkrypcja prognoza zysk operacyjny zysk netto "
            "podatek dochodowy ulga podatkowa dotacja zmiana przepisów"
        ),
        excerpt_priority_terms=(
            "ulga podatkowa",
            "podatek dochodowy",
            "zysk netto",
            "zysk operacyjny",
        ),
        transient_tax_terms=("ulga podatkowa", "preferencja podatkowa"),
    ),
    ".KL": GuidanceLocalePolicy(
        results_terms=(
            "keputusan kewangan laporan tahunan suku tahunan unjuran hasil "
            "keuntungan operasi keuntungan bersih company announcement annual "
            "report quarterly results guidance Bursa Malaysia"
        ),
        bridge_terms=(
            "taklimat keputusan transkrip unjuran cukai insentif cukai subsidi "
            "perubahan peraturan keuntungan bersih results briefing earnings "
            "guidance tax incentive Bursa Malaysia"
        ),
        excerpt_priority_terms=(
            "insentif cukai",
            "cukai",
            "keuntungan bersih",
            "keuntungan operasi",
        ),
        transient_tax_terms=("insentif cukai", "pelepasan cukai"),
    ),
}

for _suffix in (".KQ",):
    _POLICIES[_suffix] = _POLICIES[".KS"]
for _suffix in (".TW", ".TWO"):
    _POLICIES[_suffix] = _POLICIES[".HK"]
for _suffix in (".SZ",):
    _POLICIES[_suffix] = _POLICIES[".SS"]


def guidance_locale_policy(ticker: str) -> GuidanceLocalePolicy:
    """Return retrieval vocabulary for the ticker's listing jurisdiction."""
    return _POLICIES.get(get_ticker_suffix(ticker), _ENGLISH)


def all_transient_tax_terms() -> tuple[str, ...]:
    """Return deduplicated tax terms used by every supported locale policy."""
    terms = _ENGLISH.transient_tax_terms + tuple(
        term for policy in _POLICIES.values() for term in policy.transient_tax_terms
    )
    return tuple(dict.fromkeys(terms))
