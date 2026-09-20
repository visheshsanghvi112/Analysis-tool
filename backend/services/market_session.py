"""
StockIQ Pro — Market Session Service.
=====================================

Exchange-aware, timezone-governed session state machine.
Explicitly distinguishes PRE_MARKET, OPEN, POST_MARKET, CLOSED, WEEKEND, HOLIDAY, SPECIAL_SESSION, UNKNOWN.

Authoritative Calendars:
- NSE / BSE 2026 Trading Holidays (Official Exchange Calendar)
- US (NYSE / NASDAQ) 2026 Trading Holidays & Early Close Rules
- Year-aware bounds (returns UNKNOWN for uncataloged years)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from enum import Enum
from typing import Any, Dict, Optional, Set
import zoneinfo


class MarketStatus(str, Enum):
    PRE_MARKET = "PRE_MARKET"
    OPEN = "OPEN"
    POST_MARKET = "POST_MARKET"
    CLOSED = "CLOSED"
    WEEKEND = "WEEKEND"
    HOLIDAY = "HOLIDAY"
    SPECIAL_SESSION = "SPECIAL_SESSION"
    UNKNOWN = "UNKNOWN"


# ── Authoritative 2026 Exchange Holiday Calendars ────────────────────────────
# Official NSE/BSE Trading Holidays for 2026 (Excludes weekends)
HOLIDAYS_NSE_2026: Set[date] = {
    date(2026, 1, 15),   # Makar Sankranti / Pongal
    date(2026, 1, 26),   # Republic Day
    date(2026, 2, 19),   # Chhatrapati Shivaji Maharaj Jayanti
    date(2026, 3, 19),   # Id-Ul-Fitr (Ramzan Id)
    date(2026, 3, 26),   # Shri Ram Navami
    date(2026, 3, 31),   # Mahavir Jayanti
    date(2026, 4, 1),    # Annual Bank Closing
    date(2026, 4, 3),    # Good Friday
    date(2026, 4, 14),   # Dr. B.R. Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 5, 28),   # Bakri Id / Eid-ul-Adha
    date(2026, 6, 26),   # Muharram
    date(2026, 9, 14),   # Ganesh Chaturthi
    date(2026, 10, 2),   # Mahatma Gandhi Jayanti
    date(2026, 20, 10) if False else date(2026, 10, 20),  # Dussehra
    date(2026, 11, 10),  # Diwali Balipratipada
    date(2026, 11, 24),  # Prakash Gurpurb Sri Guru Nanak Dev
    date(2026, 12, 25),  # Christmas
}

# Special Trading Sessions (e.g. Diwali Muhurat Trading)
SPECIAL_SESSIONS_NSE_2026: Dict[date, Dict[str, Any]] = {
    date(2026, 11, 8): {
        "name": "Diwali Laxmi Pujan (Muhurat Trading)",
        "start": time(18, 0),
        "end": time(19, 15),
    }
}

# Official US (NYSE / NASDAQ) Trading Holidays for 2026
HOLIDAYS_US_2026: Set[date] = {
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # Martin Luther King Jr. Day
    date(2026, 2, 16),   # Washington's Birthday / Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (Observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving Day
    date(2026, 12, 25),  # Christmas Day
}

# US Early Close Schedule (Regular session closes at 13:00 ET)
EARLY_CLOSE_US_2026: Dict[date, time] = {
    date(2026, 11, 27): time(13, 0),  # Day after Thanksgiving (Black Friday)
    date(2026, 12, 24): time(13, 0),  # Christmas Eve
}

# Year-aware registry
SUPPORTED_CALENDAR_YEARS = {2026}


@dataclass
class MarketSessionState:
    status: MarketStatus
    market_open: Optional[bool]
    timezone: str
    exchange: str
    current_time_str: str
    phase_name: str
    phase_num: int
    directive: str
    as_of: datetime

    @property
    def is_open(self) -> Optional[bool]:
        return self.market_open

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "market_open": self.market_open,
            "is_open": self.market_open,
            "timezone": self.timezone,
            "exchange": self.exchange,
            "current_time_str": self.current_time_str,
            "phase_name": self.phase_name,
            "phase_num": self.phase_num,
            "directive": self.directive,
            "as_of": self.as_of.isoformat() if hasattr(self.as_of, "isoformat") else str(self.as_of),
        }


def is_indian_instrument(ticker: str) -> bool:
    t = ticker.strip().upper()
    return t.endswith(".NS") or t.endswith(".BO") or t.startswith("^NSE") or t.startswith("^BSE")


def get_market_session(
    ticker: str,
    as_of: Optional[datetime] = None,
    holidays: Optional[Set[date]] = None,
) -> MarketSessionState:
    """
    Computes the canonical session state for a given ticker and timestamp.
    Defaults to current time if as_of is None.
    
    Guarantees:
    - Year-aware: Returns UNKNOWN for uncataloged years without synthetic claims.
    - Exchange-aware: Uses official NSE/BSE and US holiday/early-close schedules.
    """
    is_in = is_indian_instrument(ticker)
    tz_str = "Asia/Kolkata" if is_in else "America/New_York"
    exchange = "NSE/BSE" if is_in else "US_EXCHANGES"

    try:
        tz = zoneinfo.ZoneInfo(tz_str)
    except Exception:
        tz = zoneinfo.ZoneInfo("UTC")

    if as_of is None:
        now = datetime.now(tz)
    else:
        if as_of.tzinfo is None:
            now = as_of.replace(tzinfo=tz)
        else:
            now = as_of.astimezone(tz)

    current_date = now.date()
    weekday = now.weekday()  # 0 = Monday, 5 = Saturday, 6 = Sunday
    time_str = now.strftime("%I:%M %p %Z")
    current_year = current_date.year

    # 1. Year Awareness Check
    if holidays is None and current_year not in SUPPORTED_CALENDAR_YEARS:
        # Weekend check still applies universally
        if weekday >= 5:
            return MarketSessionState(
                status=MarketStatus.WEEKEND,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Weekend Market Pause",
                phase_num=0,
                directive="Markets closed for the weekend.",
                as_of=now,
            )
        # For weekdays in uncataloged years, session state cannot be verified without authoritative calendar
        return MarketSessionState(
            status=MarketStatus.UNKNOWN,
            market_open=None,
            timezone=tz_str,
            exchange=exchange,
            current_time_str=time_str,
            phase_name="Uncataloged Calendar Year",
            phase_num=0,
            directive=f"Exchange holiday calendar uncataloged for year {current_year}. Market session state cannot be verified.",
            as_of=now,
        )

    # Resolve holiday calendar
    active_holidays = holidays
    if active_holidays is None:
        active_holidays = HOLIDAYS_NSE_2026 if is_in else HOLIDAYS_US_2026

    # 2. Holiday Check
    if active_holidays is not None and current_date in active_holidays:
        return MarketSessionState(
            status=MarketStatus.HOLIDAY,
            market_open=False,
            timezone=tz_str,
            exchange=exchange,
            current_time_str=time_str,
            phase_name="Exchange Trading Holiday",
            phase_num=0,
            directive="Markets closed for designated exchange holiday. Review macro context.",
            as_of=now,
        )

    # 3. Special Session Check (e.g. Diwali Muhurat Trading)
    if is_in and current_date in SPECIAL_SESSIONS_NSE_2026:
        spec = SPECIAL_SESSIONS_NSE_2026[current_date]
        spec_start = now.replace(hour=spec["start"].hour, minute=spec["start"].minute, second=0, microsecond=0)
        spec_end = now.replace(hour=spec["end"].hour, minute=spec["end"].minute, second=0, microsecond=0)

        if spec_start <= now <= spec_end:
            return MarketSessionState(
                status=MarketStatus.SPECIAL_SESSION,
                market_open=True,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name=spec["name"],
                phase_num=1,
                directive="Special festive trading session active.",
                as_of=now,
            )
        else:
            return MarketSessionState(
                status=MarketStatus.CLOSED,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name=f"{spec['name']} (Closed)",
                phase_num=0,
                directive=f"Special session scheduled for {spec['start'].strftime('%H:%M')} - {spec['end'].strftime('%H:%M')} IST.",
                as_of=now,
            )

    # 4. Weekend Check
    if weekday >= 5:
        return MarketSessionState(
            status=MarketStatus.WEEKEND,
            market_open=False,
            timezone=tz_str,
            exchange=exchange,
            current_time_str=time_str,
            phase_name="Weekend Market Pause",
            phase_num=0,
            directive="Markets closed for the weekend. Analyze multi-week setups and prepare watchlists.",
            as_of=now,
        )

    # 5. Session Hours
    if is_in:
        # India: Pre-market 09:00 - 09:15, Open 09:15 - 15:30, Post-market 15:30 - 16:00
        open_time = now.replace(hour=9, minute=15, second=0, microsecond=0)
        close_time = now.replace(hour=15, minute=30, second=0, microsecond=0)
        pre_market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        post_market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)

        if now < pre_market_start:
            return MarketSessionState(
                status=MarketStatus.CLOSED,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Overnight Session Pause",
                phase_num=0,
                directive="Markets closed. Pre-market opens at 09:00 IST.",
                as_of=now,
            )
        elif now < open_time:
            return MarketSessionState(
                status=MarketStatus.PRE_MARKET,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Pre-Market Price Discovery",
                phase_num=0,
                directive="Pre-market order matching active. Observe gap auctions.",
                as_of=now,
            )
        elif now <= close_time:
            if now < now.replace(hour=9, minute=45, second=0, microsecond=0):
                phase_name = "Opening Price Discovery / ORB"
                phase_num = 1
                directive = "Establish 15m opening range. Watch institutional order flow."
            elif now < now.replace(hour=11, minute=30, second=0, microsecond=0):
                phase_name = "Morning Momentum"
                phase_num = 2
                directive = "High institutional participation. Trend continuation favored."
            elif now < now.replace(hour=13, minute=30, second=0, microsecond=0):
                phase_name = "Midday Chop & Consolidation"
                phase_num = 3
                directive = "Volume thins out. False breakouts prevalent. Defend profits."
            elif now < now.replace(hour=15, minute=0, second=0, microsecond=0):
                phase_name = "European Crossover & Institutional Rebalancing"
                phase_num = 4
                directive = "Liquidity surges. High momentum breakout window."
            else:
                phase_name = "Closing Auction / Intraday Square-off"
                phase_num = 5
                directive = "Intraday margin square-offs accelerate. Avoid new low-timeframe risk."

            return MarketSessionState(
                status=MarketStatus.OPEN,
                market_open=True,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name=phase_name,
                phase_num=phase_num,
                directive=directive,
                as_of=now,
            )
        elif now <= post_market_end:
            return MarketSessionState(
                status=MarketStatus.POST_MARKET,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Post-Market Closing Session",
                phase_num=6,
                directive="Closing price determination active. Review day execution.",
                as_of=now,
            )
        else:
            return MarketSessionState(
                status=MarketStatus.CLOSED,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Market Closed",
                phase_num=0,
                directive="Session closed. Overnight analysis window.",
                as_of=now,
            )
    else:
        # US: Pre-market 04:00 - 09:30, Open 09:30 - 16:00 (or 13:00 on early close), Post-market to 20:00 (or 17:00)
        open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
        pre_market_start = now.replace(hour=4, minute=0, second=0, microsecond=0)

        # Early close handling (e.g. 13:00 ET on Black Friday and Christmas Eve)
        if current_date in EARLY_CLOSE_US_2026:
            close_time = now.replace(hour=13, minute=0, second=0, microsecond=0)
            post_market_end = now.replace(hour=17, minute=0, second=0, microsecond=0)
            is_early_close = True
        else:
            close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
            post_market_end = now.replace(hour=20, minute=0, second=0, microsecond=0)
            is_early_close = False

        if now < pre_market_start:
            return MarketSessionState(
                status=MarketStatus.CLOSED,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Overnight Pause",
                phase_num=0,
                directive="Markets closed. Pre-market opens at 04:00 EST.",
                as_of=now,
            )
        elif now < open_time:
            return MarketSessionState(
                status=MarketStatus.PRE_MARKET,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="US Pre-Market Session",
                phase_num=0,
                directive="Pre-market trading active. Monitor gap and macro earnings.",
                as_of=now,
            )
        elif now <= close_time:
            phase_desc = "US Early-Close Trading Session (Closes 13:00 ET)" if is_early_close else "US Regular Market Hours"
            return MarketSessionState(
                status=MarketStatus.OPEN,
                market_open=True,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name=phase_desc,
                phase_num=1,
                directive="Regular trading session active.",
                as_of=now,
            )
        elif now <= post_market_end:
            return MarketSessionState(
                status=MarketStatus.POST_MARKET,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="US After-Hours Session",
                phase_num=2,
                directive="After-hours trading active. Earnings releases pending.",
                as_of=now,
            )
        else:
            return MarketSessionState(
                status=MarketStatus.CLOSED,
                market_open=False,
                timezone=tz_str,
                exchange=exchange,
                current_time_str=time_str,
                phase_name="Closed",
                phase_num=0,
                directive="Markets closed.",
                as_of=now,
            )


def get_market_session_state(
    ticker: str,
    now_dt: Optional[datetime] = None,
    holidays: Optional[Set[date]] = None,
) -> MarketSessionState:
    """Convenience alias for get_market_session."""
    return get_market_session(ticker=ticker, as_of=now_dt, holidays=holidays)


__all__ = [
    "EARLY_CLOSE_US_2026",
    "HOLIDAYS_NSE_2026",
    "HOLIDAYS_US_2026",
    "MarketSessionState",
    "MarketStatus",
    "SPECIAL_SESSIONS_NSE_2026",
    "SUPPORTED_CALENDAR_YEARS",
    "get_market_session",
    "get_market_session_state",
    "is_indian_instrument",
]
