import argparse
import html
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    import yfinance as yf
except Exception:
    yf = None

QUARTERS = [f"{y}Q{q}" for y in range(2023, 2026) for q in range(1, 5)]
SEC_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
REVENUE_SCALE = 1_000_000
REVENUE_UNIT_LABEL = "million"
NVIDIA_SITEMAP_URL = "https://nvidianews.nvidia.com/sitemap.xml"

COMMON_NAME_MAP = {
    "台積電": ("TW", "2330", "Taiwan Semiconductor Manufacturing"),
    "台積": ("TW", "2330", "Taiwan Semiconductor Manufacturing"),
    "台達電": ("TW", "2308", "Delta Electronics"),
    "delta": ("TW", "2308", "Delta Electronics"),
    "apple": ("US", "AAPL", "Apple Inc."),
    "蘋果": ("US", "AAPL", "Apple Inc."),
    "nvidia": ("US", "NVDA", "NVIDIA Corp."),
    "輝達": ("US", "NVDA", "NVIDIA Corp."),
    "toyota": ("JP", "7203", "Toyota Motor Corp."),
    "豐田": ("JP", "7203", "Toyota Motor Corp."),
}

US_REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
]

TW_MONTHLY_REVENUE_KEYS = ["營業收入-當月營收", "當月營收", "月營收", "Revenue"]
SEGMENT_HEADINGS = {
    "Data Center": [r"Data Center"],
    "Gaming": [r"Gaming and AI PC", r"Gaming"],
    "Professional Visualization": [r"Professional Visualization"],
    "Automotive and Robotics": [r"Automotive and Robotics", r"Automotive"],
    "OEM and Other": [r"OEM and Other", r"OEM"],
}

FISCAL_QUARTER_MAP = {"first": 1, "second": 2, "third": 3, "fourth": 4}


def env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class ResolvedCompany:
    company_input: str
    market: str
    symbol: str
    name: str
    cik: Optional[str] = None


@dataclass
class SourceFetchResult:
    status: str
    message: str
    quarter_data: Dict[str, Dict[str, Any]]


class Resolver:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._sec_ticker_map: Optional[Dict[str, Dict[str, Any]]] = None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[resolver] {msg}")

    def _load_sec_ticker_map(self) -> Dict[str, Dict[str, Any]]:
        if self._sec_ticker_map is not None:
            return self._sec_ticker_map

        headers = {"User-Agent": os.getenv("SEC_USER_AGENT", "RevenueFetcher/1.0 (contact@example.com)")}
        data = http_get_json(SEC_TICKER_URL, headers=headers, retries=3, backoff_base=1.0)
        ticker_map: Dict[str, Dict[str, Any]] = {}

        for _, row in data.items():
            ticker = str(row.get("ticker", "")).upper().strip()
            if not ticker:
                continue
            ticker_map[ticker] = {
                "name": row.get("title") or ticker,
                "cik": str(row.get("cik_str", "")).zfill(10),
            }

        self._sec_ticker_map = ticker_map
        self._log(f"SEC ticker map loaded: {len(ticker_map)} records")
        return ticker_map

    def resolve_candidates(self, raw_query: str) -> List[ResolvedCompany]:
        query = raw_query.strip()
        if not query:
            return []

        prefixed = self._resolve_prefixed(query)
        if prefixed:
            return [prefixed]

        key = query.lower()
        if key in COMMON_NAME_MAP:
            market, symbol, name = COMMON_NAME_MAP[key]
            return [ResolvedCompany(raw_query, market, symbol, name)]

        if query.isdigit() and len(query) == 4:
            return [
                ResolvedCompany(raw_query, "TW", query, f"TW-{query}"),
                ResolvedCompany(raw_query, "JP", query, f"JP-{query}"),
            ]

        if re.fullmatch(r"[A-Za-z]{1,5}", query):
            ticker = query.upper()
            try:
                ticker_map = self._load_sec_ticker_map()
                info = ticker_map.get(ticker)
                if info:
                    return [ResolvedCompany(raw_query, "US", ticker, str(info["name"]), str(info["cik"]))]
            except Exception as exc:
                self._log(f"SEC lookup failed: {exc}")
            return [ResolvedCompany(raw_query, "US", ticker, ticker)]

        try:
            ticker_map = self._load_sec_ticker_map()
            hits: List[ResolvedCompany] = []
            for ticker, info in ticker_map.items():
                if key in str(info["name"]).lower():
                    hits.append(ResolvedCompany(raw_query, "US", ticker, str(info["name"]), str(info["cik"])))
                if len(hits) >= 5:
                    break
            return hits
        except Exception:
            return []

    @staticmethod
    def _resolve_prefixed(query: str) -> Optional[ResolvedCompany]:
        m = re.match(r"^(TW|US|JP)\s*[:\s]\s*(.+)$", query, re.IGNORECASE)
        if not m:
            return None
        market = m.group(1).upper()
        symbol = m.group(2).strip().upper() if market == "US" else m.group(2).strip()
        return ResolvedCompany(query, market, symbol, f"{market}-{symbol}")


def http_get_json(
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    timeout: int = 20,
    retries: int = 3,
    backoff_base: float = 1.0,
    verify: bool = True,
) -> Any:
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout, verify=verify)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_err = exc
            if attempt == retries:
                break
            time.sleep(backoff_base * (2 ** (attempt - 1)))
    raise RuntimeError(f"HTTP request failed after retries: {url}; error={last_err}")


def quarter_from_date_text(date_text: str) -> Optional[str]:
    try:
        dt = datetime.fromisoformat(str(date_text)[:10])
    except Exception:
        return None
    q = f"{dt.year}Q{((dt.month - 1) // 3) + 1}"
    return q if q in QUARTERS else None


def parse_iso_date(date_text: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(date_text)[:10])
    except Exception:
        return None


def days_between(start_date: str, end_date: str) -> Optional[int]:
    s = parse_iso_date(start_date)
    e = parse_iso_date(end_date)
    if not s or not e:
        return None
    return (e - s).days


def pick_latest(old: Optional[Tuple[str, float]], key_date: str, value: float) -> Tuple[str, float]:
    if old is None or key_date > old[0]:
        return key_date, value
    return old


def fetch_us_official(company: ResolvedCompany, verbose: bool = False) -> SourceFetchResult:
    cik = company.cik
    if not cik:
        try:
            ticker_map = Resolver(verbose=False)._load_sec_ticker_map()
            info = ticker_map.get(company.symbol.upper())
            cik = info["cik"] if info else None
        except Exception as exc:
            return SourceFetchResult("connection_error", f"Unable to resolve CIK for {company.symbol}: {exc}", {})

    if not cik:
        return SourceFetchResult("no_data", "CIK not found", {})

    headers = {"User-Agent": os.getenv("SEC_USER_AGENT", "RevenueFetcher/1.0 (contact@example.com)")}
    url = SEC_COMPANYFACTS_URL.format(cik=str(cik).zfill(10))

    try:
        data = http_get_json(url, headers=headers, retries=3, backoff_base=1.0)
    except Exception as exc:
        return SourceFetchResult("connection_error", str(exc), {})

    us_gaap = (((data or {}).get("facts") or {}).get("us-gaap") or {})
    selected_tag = None
    unit_name = None
    entries = None
    best_score = -1
    target_years = {"2023", "2024", "2025"}

    for tag in US_REVENUE_TAGS:
        node = us_gaap.get(tag)
        if not node:
            continue
        units = node.get("units", {})
        candidate_unit = "USD" if "USD" in units else (next(iter(units.keys()), None))
        if not candidate_unit:
            continue
        candidate_entries = units[candidate_unit]
        score = 0
        for entry in candidate_entries:
            frame = str(entry.get("frame") or "")
            end_date = str(entry.get("end") or "")
            if re.match(r"^CY(2023|2024|2025)Q[1-4]", frame):
                score += 2
            elif re.match(r"^CY(2023|2024|2025)$", frame):
                score += 1
            elif end_date[:4] in target_years:
                score += 1
        if score > best_score:
            best_score = score
            selected_tag = tag
            unit_name = candidate_unit
            entries = candidate_entries

    if not selected_tag or entries is None:
        return SourceFetchResult("mapping_error", "SEC data available but revenue field mapping not found", {})

    q_frame_values: Dict[str, Tuple[str, float]] = {}
    annual_values: Dict[str, Tuple[str, float]] = {}

    for entry in entries:
        frame = str(entry.get("frame") or "")
        start = str(entry.get("start") or "")
        end_date = str(entry.get("end") or "")
        filed = str(entry.get("filed") or "")
        form = str(entry.get("form") or "")
        value = entry.get("val")
        if value is None:
            continue
        val = float(value)
        key_date = filed or end_date or "0000-00-00"

        m_q = re.match(r"^CY(\d{4})Q([1-4])", frame)
        if m_q:
            q = f"{m_q.group(1)}Q{m_q.group(2)}"
            if q in QUARTERS:
                q_frame_values[q] = pick_latest(q_frame_values.get(q), key_date, val)
            continue

        m_y = re.match(r"^CY(\d{4})$", frame)
        if m_y:
            y = m_y.group(1)
            annual_values[y] = pick_latest(annual_values.get(y), key_date, val)
            continue

    quarter_values: Dict[str, float] = {q: v for q, (_, v) in q_frame_values.items()}

    for y in ["2023", "2024", "2025"]:
        q4 = f"{y}Q4"
        if q4 in quarter_values:
            continue
        annual = annual_values.get(y)
        q1 = quarter_values.get(f"{y}Q1")
        q2 = quarter_values.get(f"{y}Q2")
        q3 = quarter_values.get(f"{y}Q3")
        if annual and q1 is not None and q2 is not None and q3 is not None:
            q4_val = annual[1] - q1 - q2 - q3
            quarter_values[q4] = q4_val

    quarter_data: Dict[str, Dict[str, Any]] = {}
    for q in QUARTERS:
        if q not in quarter_values:
            continue
        quarter_data[q] = {
            "revenue": quarter_values[q],
            "currency": unit_name,
            "field_name": selected_tag,
            "source_url": url,
            "source_tier": "official",
            "status": "ok",
        }

    if verbose:
        print(f"[us] {company.symbol} via SEC tag={selected_tag}, quarters={len(quarter_data)}")

    if not quarter_data:
        return SourceFetchResult("no_data", "No SEC quarter values in target range", {})
    return SourceFetchResult("ok", "ok", quarter_data)


def tw_month_candidates() -> List[str]:
    return [f"{y}{m:02d}" for y in range(2023, 2026) for m in range(1, 13)]


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]", "", str(key).lower())


def pick_value_by_keys(row: Dict[str, Any], candidates: List[str]) -> Optional[Any]:
    for k in candidates:
        if k in row and row.get(k) not in (None, ""):
            return row.get(k)
    return None


def extract_company_code(row: Dict[str, Any]) -> str:
    val = pick_value_by_keys(row, ["公司代號", "SecuritiesCompanyCode", "公司代號CompanyCode"])
    if val is not None:
        return str(val).strip()

    for k, v in row.items():
        nk = normalize_key(k)
        if ("公司代號" in str(k)) or ("companycode" in nk):
            return str(v).strip()
    return ""


def extract_year_month(row: Dict[str, Any], fallback_yyyymm: str) -> str:
    val = pick_value_by_keys(row, ["資料年月", "YearMonth", "年/月"])
    text = str(val).strip() if val is not None else fallback_yyyymm
    ym = re.sub(r"\D", "", text)
    if len(ym) >= 6:
        return ym[:6]
    return fallback_yyyymm


def extract_tw_revenue(row: Dict[str, Any]) -> Optional[float]:
    raw = pick_value_by_keys(row, TW_MONTHLY_REVENUE_KEYS)
    if raw is not None:
        try:
            return float(str(raw).replace(",", "").strip())
        except ValueError:
            pass

    for k, v in row.items():
        key_text = str(k)
        if ("營收" in key_text) or ("revenue" in normalize_key(key_text)):
            try:
                return float(str(v).replace(",", "").strip())
            except ValueError:
                continue
    return None


def fetch_tw_official(company: ResolvedCompany, verbose: bool = False) -> SourceFetchResult:
    # MOPS direct endpoints may block non-browser automation; use official TWSE open data as operational source.
    endpoint_patterns = [
        "https://openapi.twse.com.tw/v1/opendata/t187ap05_L_{yyyymm}",
        "https://openapi.twse.com.tw/v1/opendata/t187ap06_L_{yyyymm}",
    ]
    insecure_tw_tls = env_true("ALLOW_INSECURE_TW_TLS", default=True)

    month_values: Dict[str, float] = {}
    got_any_response = False
    mapping_issue = False

    if insecure_tw_tls:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")

    for yyyymm in tw_month_candidates():
        fetched_this_month = False
        for pattern in endpoint_patterns:
            url = pattern.format(yyyymm=yyyymm)
            try:
                rows = http_get_json(
                    url,
                    retries=1,
                    timeout=8,
                    backoff_base=0.5,
                    verify=not insecure_tw_tls,
                )
            except Exception:
                continue

            got_any_response = True
            fetched_this_month = True
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue
                code = extract_company_code(row)
                if code != company.symbol:
                    continue
                ym = extract_year_month(row, yyyymm)
                value = extract_tw_revenue(row)
                if value is None:
                    mapping_issue = True
                    continue
                month_values[ym] = value

        if not fetched_this_month and verbose:
            print(f"[tw] no response for {yyyymm}")

    if not got_any_response:
        return SourceFetchResult("connection_error", "TW official endpoint unreachable", {})
    if mapping_issue and not month_values:
        return SourceFetchResult("mapping_error", "TW official data found but revenue field mapping failed", {})
    if not month_values:
        return SourceFetchResult("no_data", "No TW monthly revenue data found", {})

    quarter_data: Dict[str, Dict[str, Any]] = {}
    for q in QUARTERS:
        y = int(q[:4])
        qq = int(q[-1])
        months = [f"{y}{m:02d}" for m in range((qq - 1) * 3 + 1, (qq - 1) * 3 + 4)]
        vals = [month_values[m] for m in months if m in month_values]
        if len(vals) == 3:
            quarter_data[q] = {
                "revenue": float(sum(vals)),
                "currency": "TWD",
                "field_name": "MonthlyRevenueSum",
                "source_url": "https://mops.twse.com.tw/",
                "source_tier": "official",
                "status": "ok",
            }

    if verbose:
        print(f"[tw] {company.symbol} official quarters={len(quarter_data)}")

    if not quarter_data:
        return SourceFetchResult("no_data", "No complete TW quarter values in target range", {})
    return SourceFetchResult("ok", "ok", quarter_data)


def fetch_jp_official(company: ResolvedCompany, verbose: bool = False) -> SourceFetchResult:
    code_map_text = os.getenv("EDINET_CODE_MAP", "").strip()
    api_key = os.getenv("EDINET_API_KEY", "").strip()

    if not code_map_text:
        return SourceFetchResult(
            "no_data",
            "EDINET_CODE_MAP not configured; cannot map JP ticker to EDINET code",
            {},
        )

    code_map: Dict[str, str] = {}
    for item in code_map_text.split(","):
        if ":" in item:
            k, v = item.split(":", 1)
            code_map[k.strip()] = v.strip()

    edinet_code = code_map.get(company.symbol)
    if not edinet_code:
        return SourceFetchResult("no_data", f"No EDINET mapping for {company.symbol}", {})

    params = {"date": datetime.now().strftime("%Y-%m-%d"), "type": 2}
    if api_key:
        params["Subscription-Key"] = api_key

    url = "https://disclosure2dl.edinet-fsa.go.jp/api/v2/documents.json"
    try:
        data = http_get_json(url, params=params, retries=2, backoff_base=0.7)
    except Exception as exc:
        return SourceFetchResult("connection_error", str(exc), {})

    if verbose:
        print(f"[jp] EDINET listing fetched, rows={len((data or {}).get('results') or [])}")

    return SourceFetchResult(
        "no_data",
        "EDINET integration requires document download + XBRL parse by EDINET code",
        {},
    )


def quarter_sort_key(q: str) -> int:
    return int(q[:4]) * 10 + int(q[-1])


def parse_amount_to_million(text: str) -> Optional[float]:
    m = re.search(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(billion|million)", text, re.I)
    if not m:
        return None
    num = float(m.group(1).replace(",", ""))
    unit = m.group(2).lower()
    return num * 1000 if unit == "billion" else num


def extract_cy_quarter_from_fiscal_release(url: str, html_text: str) -> Optional[str]:
    for text in [url, html_text]:
        m = re.search(r"(first|second|third|fourth)[\s-]+quarter(?:[\s-]+and)?[\s-]+fiscal[\s-]+(\d{4})", text, re.I)
        if not m:
            continue
        fq = FISCAL_QUARTER_MAP.get(m.group(1).lower())
        fiscal_year = int(m.group(2))
        if fq is None:
            continue
        cy_year = fiscal_year - 1
        q = f"{cy_year}Q{fq}"
        if q in QUARTERS:
            return q
    return None


def extract_segment_value_from_release(html_text: str, heading_pattern: str) -> Optional[float]:
    pattern = rf"<p[^>]*>\s*<strong[^>]*>\s*{heading_pattern}\s*</strong>\s*</p>.*?<li[^>]*>(.*?)</li>"
    m = re.search(pattern, html_text, re.I | re.S)
    if not m:
        return None
    li_text = re.sub(r"<[^>]+>", " ", m.group(1))
    li_text = html.unescape(re.sub(r"\s+", " ", li_text)).strip()
    return parse_amount_to_million(li_text)


def fetch_nvda_segment_revenue_from_company_site(verbose: bool = False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    try:
        xml = requests.get(NVIDIA_SITEMAP_URL, timeout=25).text
    except Exception:
        return pd.DataFrame()

    locs = re.findall(r"<loc>([^<]+)</loc>", xml)
    urls = [
        u.replace("http://", "https://")
        for u in locs
        if "nvidia" in u.lower()
        and "financial-results-for" in u.lower()
        and re.search(r"fiscal-(2024|2025|2026)", u, re.I)
    ]
    urls = sorted(set(urls), reverse=True)

    by_quarter_segment: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for url in urls:
        try:
            page = requests.get(url, timeout=25)
            page.raise_for_status()
            html_text = page.text
        except Exception:
            continue

        quarter = extract_cy_quarter_from_fiscal_release(url, html_text)
        if quarter not in QUARTERS:
            continue

        for segment, heading_patterns in SEGMENT_HEADINGS.items():
            val = None
            for hp in heading_patterns:
                val = extract_segment_value_from_release(html_text, hp)
                if val is not None:
                    break
            if val is None:
                continue

            key = (quarter, segment)
            old = by_quarter_segment.get(key)
            if old is None:
                by_quarter_segment[key] = {
                    "quarter": quarter,
                    "segment": segment,
                    "revenue": val,
                    "revenue_unit": REVENUE_UNIT_LABEL,
                    "currency": "USD",
                    "source_tier": "company_official_site",
                    "source_url": url,
                    "status": "ok",
                }

    for q in QUARTERS:
        for seg in SEGMENT_HEADINGS.keys():
            key = (q, seg)
            item = by_quarter_segment.get(key)
            if item:
                rows.append(item)
            else:
                rows.append(
                    {
                        "quarter": q,
                        "segment": seg,
                        "revenue": pd.NA,
                        "revenue_unit": REVENUE_UNIT_LABEL,
                        "currency": "USD",
                        "source_tier": pd.NA,
                        "source_url": pd.NA,
                        "status": "missing",
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["quarter_order"] = df["quarter"].map(quarter_sort_key)
        df = df.sort_values(by=["quarter_order", "segment"]).drop(columns=["quarter_order"])
    if verbose:
        ok_n = int((df["status"] == "ok").sum()) if not df.empty else 0
        print(f"[segment] NVDA company site rows={len(df)} ok={ok_n}")
    return df


def fetch_company_site_segment_revenue(company: ResolvedCompany, verbose: bool = False) -> pd.DataFrame:
    if company.market == "US" and company.symbol.upper() == "NVDA":
        df = fetch_nvda_segment_revenue_from_company_site(verbose=verbose)
        if df.empty:
            return df
        df.insert(0, "company_input", company.company_input)
        df.insert(1, "resolved_market", company.market)
        df.insert(2, "resolved_symbol", company.symbol)
        df.insert(3, "resolved_name", company.name)
        return df
    return pd.DataFrame()


def fetch_fallback_ir_like(company: ResolvedCompany, verbose: bool = False) -> SourceFetchResult:
    if yf is None:
        return SourceFetchResult("unsupported", "yfinance not installed", {})

    ticker = company.symbol.upper() if company.market == "US" else (f"{company.symbol}.TW" if company.market == "TW" else f"{company.symbol}.T")

    try:
        t = yf.Ticker(ticker)
        q_stmt = t.quarterly_income_stmt
    except Exception as exc:
        return SourceFetchResult("connection_error", str(exc), {})

    if q_stmt is None or q_stmt.empty:
        return SourceFetchResult("no_data", "Fallback source has no quarterly income statement", {})

    selected_row = None
    for candidate in ["Total Revenue", "Revenues", "Revenue", "Net Sales"]:
        if candidate in q_stmt.index:
            selected_row = candidate
            break

    if not selected_row:
        return SourceFetchResult("mapping_error", "Fallback source has no revenue-like field", {})

    row = q_stmt.loc[selected_row]
    quarter_data: Dict[str, Dict[str, Any]] = {}
    for col, val in row.items():
        if pd.isna(val):
            continue
        try:
            dt = col.to_pydatetime() if hasattr(col, "to_pydatetime") else pd.to_datetime(col).to_pydatetime()
        except Exception:
            continue

        q = f"{dt.year}Q{((dt.month - 1) // 3) + 1}"
        if q not in QUARTERS:
            continue

        source_tier = "fallback_company_site" if company.market == "US" else "fallback_ir"
        source_url = f"https://finance.yahoo.com/quote/{ticker}/financials"
        quarter_data[q] = {
            "revenue": float(val),
            "currency": "UNKNOWN",
            "field_name": selected_row,
            "source_url": source_url,
            "source_tier": source_tier,
            "status": "ok",
        }

    if verbose:
        print(f"[fallback] {company.symbol} via {ticker}, quarters={len(quarter_data)}")

    if not quarter_data:
        return SourceFetchResult("no_data", "Fallback has no target quarter values", {})
    return SourceFetchResult("ok", "ok", quarter_data)


def get_official_revenue(company: ResolvedCompany, verbose: bool = False) -> SourceFetchResult:
    if company.market == "US":
        return fetch_us_official(company, verbose=verbose)
    if company.market == "TW":
        return fetch_tw_official(company, verbose=verbose)
    if company.market == "JP":
        return fetch_jp_official(company, verbose=verbose)
    return SourceFetchResult("unsupported", f"Unsupported market: {company.market}", {})


def build_company_dataframe(company: ResolvedCompany, verbose: bool = False) -> Tuple[pd.DataFrame, List[str]]:
    notes: List[str] = []
    official = get_official_revenue(company, verbose=verbose)
    merged = dict(official.quarter_data)

    allow_tw_fallback = env_true("ALLOW_TW_FALLBACK", default=False)

    if official.status in {"connection_error", "no_data"}:
        if company.market == "TW" and not allow_tw_fallback:
            notes.append(f"official={official.status}; fallback=disabled_for_tw")
            if official.message != "ok":
                notes.append(f"official_message={official.message}")
            merged = dict(official.quarter_data)
        else:
            fallback = fetch_fallback_ir_like(company, verbose=verbose)
            if fallback.status == "ok":
                for q, payload in fallback.quarter_data.items():
                    if q not in merged:
                        merged[q] = payload
                notes.append(f"official={official.status}; fallback=ok")
            else:
                notes.append(f"official={official.status}; fallback={fallback.status}")
                notes.append(f"fallback_message={fallback.message}")
    elif official.status == "mapping_error":
        notes.append("official mapping error; fallback intentionally skipped")
    elif official.status != "ok":
        notes.append(f"official={official.status}")

    if official.message != "ok":
        notes.append(f"official_message={official.message}")

    rows: List[Dict[str, Any]] = []
    for q in QUARTERS:
        payload = merged.get(q)
        if payload:
            revenue_raw = payload.get("revenue")
            revenue_scaled = float(revenue_raw) / REVENUE_SCALE if revenue_raw is not None else pd.NA
            rows.append(
                {
                    "company_input": company.company_input,
                    "resolved_market": company.market,
                    "resolved_symbol": company.symbol,
                    "resolved_name": company.name,
                    "quarter": q,
                    "revenue": revenue_scaled,
                    "revenue_unit": REVENUE_UNIT_LABEL,
                    "currency": payload.get("currency"),
                    "revenue_field_used": payload.get("field_name"),
                    "source_tier": payload.get("source_tier"),
                    "source_url": payload.get("source_url"),
                    "status": payload.get("status", "ok"),
                }
            )
        else:
            rows.append(
                {
                    "company_input": company.company_input,
                    "resolved_market": company.market,
                    "resolved_symbol": company.symbol,
                    "resolved_name": company.name,
                    "quarter": q,
                    "revenue": pd.NA,
                    "revenue_unit": REVENUE_UNIT_LABEL,
                    "currency": pd.NA,
                    "revenue_field_used": pd.NA,
                    "source_tier": pd.NA,
                    "source_url": pd.NA,
                    "status": "missing",
                }
            )

    notes = list(dict.fromkeys(notes))
    return pd.DataFrame(rows), notes


def print_company_table(df: pd.DataFrame) -> None:
    label = f"{df.iloc[0]['resolved_market']}:{df.iloc[0]['resolved_symbol']} ({df.iloc[0]['resolved_name']})"
    print(f"\n=== {label} ===")
    print(df[["quarter", "revenue", "revenue_unit", "currency", "source_tier", "status"]].to_string(index=False))


def print_segment_table(segment_df: pd.DataFrame) -> None:
    if segment_df.empty:
        print("No segment revenue found from company official site.")
        return
    label = f"{segment_df.iloc[0]['resolved_market']}:{segment_df.iloc[0]['resolved_symbol']} ({segment_df.iloc[0]['resolved_name']})"
    print(f"\n--- Segment Revenue ({label}) ---")
    print(segment_df[["quarter", "segment", "revenue", "revenue_unit", "currency", "status"]].to_string(index=False))


def sanitize_sheet_name(name: str) -> str:
    bad = r"[]:*?/\\"
    cleaned = "".join("_" if ch in bad else ch for ch in name)
    return cleaned[:31]


def export_outputs(all_dfs: List[pd.DataFrame], outdir: Path) -> Tuple[Optional[Path], List[Path], Optional[str]]:
    outdir.mkdir(parents=True, exist_ok=True)

    csv_paths: List[Path] = []
    summary_rows: List[Dict[str, Any]] = []
    for df in all_dfs:
        market = str(df.iloc[0]["resolved_market"])
        symbol = str(df.iloc[0]["resolved_symbol"])
        csv_path = outdir / f"{market}_{symbol}_revenue_2023Q1_2025Q4.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        csv_paths.append(csv_path)

        summary_rows.append(
            {
                "resolved_market": market,
                "resolved_symbol": symbol,
                "resolved_name": str(df.iloc[0]["resolved_name"]),
                "ok_quarters": int((df["status"] == "ok").sum()),
                "missing_quarters": int((df["status"] == "missing").sum()),
                "csv_file": str(csv_path),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    xlsx_path = outdir / "revenue_2023Q1_2025Q4.xlsx"

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, index=False, sheet_name="summary")
            for df in all_dfs:
                market = str(df.iloc[0]["resolved_market"])
                symbol = str(df.iloc[0]["resolved_symbol"])
                df.to_excel(writer, index=False, sheet_name=sanitize_sheet_name(f"{market}_{symbol}"))
        return xlsx_path, csv_paths, None
    except ModuleNotFoundError as exc:
        return None, csv_paths, f"Excel skipped: {exc}"


def export_segment_outputs(segment_dfs: List[pd.DataFrame], outdir: Path) -> List[Path]:
    if not segment_dfs:
        return []
    outdir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for df in segment_dfs:
        if df.empty:
            continue
        market = str(df.iloc[0]["resolved_market"])
        symbol = str(df.iloc[0]["resolved_symbol"])
        p = outdir / f"{market}_{symbol}_segment_revenue_2023Q1_2025Q4.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(p)
    return paths


def parse_query_list(query_text: str) -> List[str]:
    return [x.strip() for x in query_text.split(",") if x.strip()]


def choose_candidate_interactive(raw_query: str, candidates: List[ResolvedCompany]) -> Optional[ResolvedCompany]:
    print(f"\nQuery '{raw_query}' has multiple candidates:")
    for idx, c in enumerate(candidates, start=1):
        print(f"  {idx}. {c.market}:{c.symbol} - {c.name}")
    picked = input("Select number (blank to skip): ").strip()
    if not picked.isdigit():
        return None
    i = int(picked)
    if not (1 <= i <= len(candidates)):
        return None
    return candidates[i - 1]


def resolve_for_execution(resolver: Resolver, raw_query: str, interactive: bool) -> Tuple[Optional[ResolvedCompany], Optional[str]]:
    candidates = resolver.resolve_candidates(raw_query)
    if not candidates:
        return None, f"No match for query: {raw_query}"
    if len(candidates) == 1:
        return candidates[0], None

    if interactive:
        picked = choose_candidate_interactive(raw_query, candidates)
        if picked:
            return picked, None
        return None, f"Ambiguous query skipped: {raw_query}"

    lines = [f"Ambiguous query in CLI mode: {raw_query}", "Candidates:"]
    for c in candidates:
        lines.append(f"- {c.market}:{c.symbol} {c.name}")
    return None, "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch quarterly revenue (2023Q1-2025Q4) for TW/US/JP from official-first sources."
    )
    parser.add_argument("--query", help="Single query or comma-separated queries, e.g. '2330,AAPL,7203,台積電'")
    parser.add_argument("--outdir", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--verbose", action="store_true", help="Show resolver/source details")
    args = parser.parse_args()

    query_text = args.query
    interactive = False
    if not query_text:
        interactive = True
        query_text = input("Enter company query (single or comma-separated): ").strip()
        if not query_text:
            print("No query provided.")
            return 1

    queries = parse_query_list(query_text)
    resolver = Resolver(verbose=args.verbose)

    all_dfs: List[pd.DataFrame] = []
    segment_dfs: List[pd.DataFrame] = []
    had_error = False

    for raw in queries:
        company, err = resolve_for_execution(resolver, raw, interactive)
        if err:
            print(err)
            had_error = True
            continue
        assert company is not None

        if args.verbose:
            print(f"[resolved] input={raw} -> {company.market}:{company.symbol} ({company.name})")

        df, notes = build_company_dataframe(company, verbose=args.verbose)
        print_company_table(df)
        for note in notes:
            print(f"note: {note}")
        all_dfs.append(df)

        seg_df = fetch_company_site_segment_revenue(company, verbose=args.verbose)
        if not seg_df.empty:
            print_segment_table(seg_df)
            segment_dfs.append(seg_df)

    if not all_dfs:
        print("No company data generated.")
        return 1

    xlsx_path, csv_paths, excel_warning = export_outputs(all_dfs, Path(args.outdir))
    segment_paths = export_segment_outputs(segment_dfs, Path(args.outdir))
    for p in csv_paths:
        print(f"Saved CSV: {p}")
    for p in segment_paths:
        print(f"Saved Segment CSV: {p}")

    if xlsx_path:
        print(f"Saved Excel: {xlsx_path}")
    if excel_warning:
        print(excel_warning)

    success_count = sum(int((df["status"] == "ok").sum() > 0) for df in all_dfs)
    print(f"Summary: total={len(all_dfs)}, with_data={success_count}, no_data={len(all_dfs)-success_count}")

    return 1 if had_error else 0


if __name__ == "__main__":
    sys.exit(main())
