"""
NASDAQ market-calendar gate.

Reads dbo.market_calendar (shared across NASDAQ/NSE/FOREX) to decide whether the
daily ML pipeline should run for a given date. Used to skip prediction + DB inserts
on NASDAQ market holidays (and weekends), so we don't write predictions for days the
market never opened.

Table dbo.market_calendar columns:
    calendar_date (date), market (varchar: 'NASDAQ'|'NSE'|'FOREX'),
    is_trading_day (bit), is_holiday (bit), holiday_name (varchar),
    day_of_week (tinyint), is_options_expiry (bit)

Fail-open policy: if the calendar has no row for the date, or the lookup errors,
we return is_trading_day=True so a genuine trading day is never blocked by a gap in
the calendar data.
"""

import logging
from datetime import date, datetime
from typing import NamedTuple, Optional, Union

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from database.connection import SQLServerConnection

logger = logging.getLogger(__name__)

MARKET = 'NASDAQ'


class CalendarStatus(NamedTuple):
    is_trading_day: bool       # True = pipeline should run
    is_holiday: bool           # True = positively confirmed market holiday
    holiday_name: Optional[str]
    found: bool                # True = a calendar row existed for this date
    check_date: date
    reason: str                # human-readable explanation


def _coerce_date(check_date: Union[date, datetime, str, None]) -> date:
    if check_date is None:
        return date.today()
    if isinstance(check_date, datetime):
        return check_date.date()
    if isinstance(check_date, date):
        return check_date
    return datetime.strptime(str(check_date)[:10], '%Y-%m-%d').date()


def get_nasdaq_calendar_status(
    check_date: Union[date, datetime, str, None] = None,
    db: Optional[SQLServerConnection] = None,
) -> CalendarStatus:
    """
    Look up the NASDAQ calendar status for ``check_date`` (default: today).

    Fails open (is_trading_day=True) on missing calendar row or any DB error.
    """
    d = _coerce_date(check_date)
    db = db or SQLServerConnection()

    query = """
        SELECT TOP 1 is_trading_day, is_holiday, holiday_name
        FROM dbo.market_calendar
        WHERE market = :market AND calendar_date = :calendar_date
    """
    try:
        df = db.execute_query(query, {'market': MARKET, 'calendar_date': d})
    except Exception as e:
        logger.warning(f"market_calendar lookup failed for {d} ({e}); failing open (treating as trading day)")
        return CalendarStatus(True, False, None, False, d,
                              f"Calendar lookup failed ({e}); proceeding as trading day")

    if df is None or df.empty:
        logger.warning(f"No market_calendar row for {MARKET} on {d}; failing open (treating as trading day)")
        return CalendarStatus(True, False, None, False, d,
                              f"No calendar entry for {d}; proceeding as trading day")

    row = df.iloc[0]
    is_trading_day = bool(row['is_trading_day'])
    is_holiday = bool(row['is_holiday'])
    holiday_name = row['holiday_name'] if row['holiday_name'] not in (None, '') else None

    if is_trading_day:
        reason = f"{MARKET} is open on {d}"
    elif is_holiday:
        reason = f"{MARKET} market holiday on {d}: {holiday_name or 'unnamed holiday'}"
    else:
        reason = f"{MARKET} is closed on {d} (non-trading day / weekend)"

    return CalendarStatus(is_trading_day, is_holiday, holiday_name, True, d, reason)


def is_nasdaq_trading_day(
    check_date: Union[date, datetime, str, None] = None,
    db: Optional[SQLServerConnection] = None,
) -> bool:
    """Convenience boolean: True if the NASDAQ pipeline should run for the date."""
    return get_nasdaq_calendar_status(check_date, db).is_trading_day


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else None
    status = get_nasdaq_calendar_status(target)
    print(f"Date        : {status.check_date}")
    print(f"Trading day : {status.is_trading_day}")
    print(f"Holiday     : {status.is_holiday} ({status.holiday_name})")
    print(f"Calendar hit: {status.found}")
    print(f"Reason      : {status.reason}")
    # Exit non-zero when the market is closed, so .bat schedulers can branch on it.
    sys.exit(0 if status.is_trading_day else 10)
