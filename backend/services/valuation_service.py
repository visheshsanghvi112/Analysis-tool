"""
Valuation Service
=================
Canonical, provenance-aware valuation service for StockIQ Pro.

Extracts valuation and DCF computation from router and adapter layers.
Ensures zero synthetic cash-flow fabrication:
- Does NOT manufacture cash flow from `price * shares * 0.04` or `revenue * 0.06`.
- Strictly requires legitimate, verified cash-flow or earnings inputs.
- Clearly separates valuation methodology from data completeness.
- Distinguishes standard Enterprise DCF from Financial Institution Equity Cashflow Proxy.
- Exposes explicit assumption provenance (data_status vs assumption_status).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import math


@dataclass
class ValuationResult:
    fair_value: Optional[float] = None
    methodology: str = "STANDARD_DCF"
    data_status: str = "INSUFFICIENT_DATA"  # "COMPLETE", "PARTIAL", "INSUFFICIENT_DATA"
    assumption_status: str = "DEFAULT_MODEL_ASSUMPTIONS"  # "DEFAULT_MODEL_ASSUMPTIONS", "CUSTOM_ASSUMPTIONS"
    valuation_status: str = "INSUFFICIENT_DATA"  # "OK", "PROXY", "INSUFFICIENT_DATA"
    valuation_reason: Optional[str] = None
    starting_flow: Optional[float] = None
    flow_type: Optional[str] = None
    growth_rate: Optional[float] = None
    discount_rate: Optional[float] = None
    terminal_growth: Optional[float] = 0.045
    enterprise_value: Optional[float] = None
    equity_value: Optional[float] = None
    shares_outstanding: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fair_value": self.fair_value,
            "methodology": self.methodology,
            "data_status": self.data_status,
            "assumption_status": self.assumption_status,
            "valuation_status": self.valuation_status,
            "valuation_reason": self.valuation_reason,
            "starting_flow": self.starting_flow,
            "flow_type": self.flow_type,
            "growth_rate": self.growth_rate,
            "discount_rate": self.discount_rate,
            "terminal_growth": self.terminal_growth,
            "enterprise_value": self.enterprise_value,
            "equity_value": self.equity_value,
            "shares_outstanding": self.shares_outstanding,
            "details": self.details,
        }


def _clean_num(val: Any) -> Optional[float]:
    """Safely cast numeric value, rejecting None, NaN, and Inf."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def is_financial_institution(info: Dict[str, Any], ticker: str = "") -> bool:
    """
    Identifies whether a company is a bank, NBFC, or financial institution
    where traditional debt-deducted enterprise DCF is mathematically inappropriate.
    """
    ticker_clean = (ticker or info.get("symbol") or "").upper()
    sector = str(info.get("sector") or "").lower()
    industry = str(info.get("industry") or "").lower()
    name = str(info.get("longName") or "").lower()

    fin_terms = ["finance", "financial", "bank", "insurance", "lending", "nbfc", "credit"]
    if any(term in sector or term in industry or term in name for term in fin_terms):
        return True

    # Common Indian / global financial ticker identifiers
    known_fins = ["BANK", "FINANCE", "FINSERV", "BAJFINANCE", "MUTHOOT", "IRFC", "PFC", "REC", "HDFC", "SBIN", "ICICIBANK"]
    if any(kw in ticker_clean for kw in known_fins):
        return True

    return False


def calculate_canonical_valuation(
    info: Dict[str, Any],
    current_price: Optional[float] = None,
    custom_growth_rate: Optional[float] = None,
    custom_discount_rate: Optional[float] = None,
    custom_terminal_growth: Optional[float] = None,
    custom_starting_flow: Optional[float] = None,
) -> ValuationResult:
    """
    Computes a canonical, deterministic DCF / Equity Cashflow Proxy intrinsic fair value.
    
    Data Integrity Guarantees:
    - Never manufactures cash flow from arbitrary market cap yields or revenue percentages.
    - If required cash flow / net income inputs are unavailable, returns fair_value = None
      and valuation_status = 'INSUFFICIENT_DATA'.
    - Clearly distinguishes STANDARD_DCF (using verified FCF) from OCF_PROXY_VALUATION (using OCF).
    - Accurately names financial institution model as FINANCIAL_INSTITUTION_EQUITY_CASHFLOW_PROXY.
    - Exposes assumption_status to separate data completeness from model parameters.
    """
    ticker = str(info.get("symbol") or "")
    is_financial = is_financial_institution(info, ticker)

    # Assumption provenance tracking
    is_custom_assumptions = any([
        custom_growth_rate is not None,
        custom_discount_rate is not None,
        custom_terminal_growth is not None,
        custom_starting_flow is not None,
    ])
    assumption_status = "CUSTOM_ASSUMPTIONS" if is_custom_assumptions else "DEFAULT_MODEL_ASSUMPTIONS"

    # Price & Shares
    price = _clean_num(current_price) or _clean_num(info.get("currentPrice")) or _clean_num(info.get("regularMarketPrice"))
    market_cap = _clean_num(info.get("marketCap"))
    shares = _clean_num(info.get("sharesOutstanding")) or 0.0

    if shares <= 0 and market_cap and price and price > 0:
        shares = market_cap / price

    if shares <= 0:
        return ValuationResult(
            fair_value=None,
            methodology="FINANCIAL_INSTITUTION_EQUITY_CASHFLOW_PROXY" if is_financial else "STANDARD_DCF",
            data_status="INSUFFICIENT_DATA",
            assumption_status=assumption_status,
            valuation_status="INSUFFICIENT_DATA",
            valuation_reason="Shares outstanding unavailable",
            details={"error": "Missing shares outstanding or market cap"},
        )

    # Financial data extraction
    fcf = _clean_num(info.get("freeCashflow"))
    ocf = _clean_num(info.get("operatingCashflow"))
    net_income = _clean_num(info.get("netIncomeToCommon"))
    cash = _clean_num(info.get("totalCash")) or 0.0
    debt = _clean_num(info.get("totalDebt")) or 0.0

    # Determine starting cash flow strictly without synthetic manufacture
    starting_flow: Optional[float] = None
    flow_type: Optional[str] = None
    methodology: str = "STANDARD_DCF"
    data_status: str = "INSUFFICIENT_DATA"
    valuation_status: str = "INSUFFICIENT_DATA"

    if custom_starting_flow is not None and custom_starting_flow > 0:
        starting_flow = custom_starting_flow
        flow_type = "User Defined Starting Flow"
        methodology = "CUSTOM_VALUATION"
        data_status = "COMPLETE"
        valuation_status = "OK"
    elif is_financial:
        methodology = "FINANCIAL_INSTITUTION_EQUITY_CASHFLOW_PROXY"
        if net_income is not None and net_income > 0:
            starting_flow = net_income
            flow_type = "Net Income (Equity Cashflow Proxy)"
            data_status = "COMPLETE"
            valuation_status = "OK"
        else:
            # Do NOT manufacture from rev * 0.15 or market_cap * 0.05
            return ValuationResult(
                fair_value=None,
                methodology=methodology,
                data_status="INSUFFICIENT_DATA",
                assumption_status=assumption_status,
                valuation_status="INSUFFICIENT_DATA",
                valuation_reason="Required net income input unavailable or non-positive for financial institution",
                shares_outstanding=shares,
                details={"net_income": net_income, "is_financial": True},
            )
    else:
        if fcf is not None and fcf > 0:
            starting_flow = fcf
            flow_type = "Free Cash Flow (source-provided)"
            methodology = "STANDARD_DCF"
            data_status = "COMPLETE"
            valuation_status = "OK"
        elif ocf is not None and ocf > 0:
            # Operating cash flow is an explicit proxy; do NOT present as standard DCF
            starting_flow = ocf
            flow_type = "Operating Cash Flow (OCF Proxy Baseline)"
            methodology = "OCF_PROXY_VALUATION"
            data_status = "PARTIAL"
            valuation_status = "PROXY"
        else:
            # Do NOT manufacture from rev * 0.06 or price * shares * 0.04
            return ValuationResult(
                fair_value=None,
                methodology="STANDARD_DCF",
                data_status="INSUFFICIENT_DATA",
                assumption_status=assumption_status,
                valuation_status="INSUFFICIENT_DATA",
                valuation_reason="Required cash-flow inputs unavailable (FCF/OCF <= 0 or missing)",
                shares_outstanding=shares,
                details={"fcf": fcf, "ocf": ocf, "is_financial": False},
            )

    # Growth rate
    if custom_growth_rate is not None:
        growth_rate = custom_growth_rate
    else:
        rev_growth = _clean_num(info.get("revenueGrowth"))
        earn_growth = _clean_num(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
        base_growth = rev_growth if rev_growth is not None else earn_growth
        if base_growth is not None:
            growth_rate = max(0.03, min(0.20, base_growth))
        else:
            growth_rate = 0.08  # Documented standard macro baseline

    # Discount rate (WACC / Cost of Equity Ke)
    if custom_discount_rate is not None:
        discount_rate = custom_discount_rate
    else:
        beta = _clean_num(info.get("beta")) or 1.0
        # CAPM baseline: Rf (6.5%) + Beta * ERP (6.0%)
        discount_rate = max(0.08, min(0.16, 0.065 + beta * 0.06))

    terminal_growth = custom_terminal_growth if custom_terminal_growth is not None else 0.045
    if discount_rate <= terminal_growth:
        discount_rate = terminal_growth + 0.02

    # 10-Year Discrete DCF Projection
    pv_sum = 0.0
    cf = starting_flow
    projected_flows = []
    for t in range(1, 11):
        cf *= (1.0 + growth_rate)
        discount_factor = (1.0 + discount_rate) ** t
        pv = cf / discount_factor
        pv_sum += pv
        projected_flows.append({"year": t, "cf": round(cf, 2), "pv": round(pv, 2)})

    # Terminal Value (Gordon Growth Model)
    terminal_cf = cf * (1.0 + terminal_growth)
    tv = terminal_cf / (discount_rate - terminal_growth)
    pv_tv = tv / ((1.0 + discount_rate) ** 10)

    enterprise_value = pv_sum + pv_tv

    if is_financial:
        # For financial institutions, equity value equals projected cashflow value directly.
        # Operating debt (deposits/borrowings) is not deducted as industrial debt.
        equity_value = enterprise_value
    else:
        # Standard DCF / OCF Proxy: Equity Value = Enterprise Value + Cash - Debt
        equity_value = enterprise_value + cash - debt

    if equity_value <= 0:
        fair_value_per_share = 0.0
    else:
        fair_value_per_share = round(equity_value / shares, 2)

    return ValuationResult(
        fair_value=fair_value_per_share,
        methodology=methodology,
        data_status=data_status,
        assumption_status=assumption_status,
        valuation_status=valuation_status,
        valuation_reason=None if valuation_status in ("OK", "PROXY") else "Valuation incomplete",
        starting_flow=round(starting_flow, 2),
        flow_type=flow_type,
        growth_rate=round(growth_rate, 4),
        discount_rate=round(discount_rate, 4),
        terminal_growth=round(terminal_growth, 4),
        enterprise_value=round(enterprise_value, 2),
        equity_value=round(equity_value, 2),
        shares_outstanding=round(shares, 2),
        details={
            "pv_discrete_flows": round(pv_sum, 2),
            "pv_terminal_value": round(pv_tv, 2),
            "cash_adjustment": round(cash, 2) if not is_financial else 0.0,
            "debt_adjustment": round(debt, 2) if not is_financial else 0.0,
            "is_financial": is_financial,
        },
    )
