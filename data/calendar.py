import datetime
from typing import Set

import nselib
from nsepython import nse_holidays, nse_marketStatus

# cached holiday date set — populated on first call, lives for pipeline lifetime
_holiday_cache: Set[datetime.date] | None = None


def _parse_holiday_date(raw: str) -> datetime.date:
    return datetime.datetime.strptime(raw, "%d-%b-%Y").date()


def load_holiday_calendar() -> Set[datetime.date]:
    global _holiday_cache

    if _holiday_cache is not None:
        return _holiday_cache

    holidays: Set[datetime.date] = set()

    # nselib returns a bulk calendar for the current year — one call for all segments
    try:
        df = nselib.trading_holiday_calendar()
        # first column is always tradingDate — name varies by segment
        for row in df.iloc[:, 0]:
            try:
                holidays.add(_parse_holiday_date(str(row)))
            except (ValueError, TypeError):
                continue
    except Exception:
        pass  # TODO: log warning — holiday calendar fetch failed

    # fallback — nsepython's api endpoint covers the same data with a different shape
    if not holidays:
        try:
            raw = nse_holidays(type="trading")
            for segment_list in raw.values():
                for entry in segment_list:
                    try:
                        holidays.add(_parse_holiday_date(entry["tradingDate"]))
                    except (ValueError, KeyError, TypeError):
                        continue
        except Exception:
            pass  # TODO: log warning — nse_holidays fallback also failed

    _holiday_cache = holidays
    return holidays


def is_trading_day(date: datetime.date | None = None) -> bool:
    if date is None:
        date = datetime.date.today()
    if date.weekday() >= 5:
        return False
    return date not in load_holiday_calendar()


def market_open() -> bool:
    try:
        status = nse_marketStatus()
        return status.get("marketState", "").upper() == "OPEN"
    except Exception:
        pass  # TODO: log warning — market status check failed
        return False


def next_trading_day(from_date: datetime.date | None = None) -> datetime.date:
    if from_date is None:
        from_date = datetime.date.today()
    candidate = from_date
    for _ in range(30):
        candidate += datetime.timedelta(days=1)
        if is_trading_day(candidate):
            return candidate
    raise RuntimeError("no trading day found within 30 days of %s" % from_date)


def nth_prior_trading_day(date: datetime.date, n: int) -> datetime.date:
    candidate = date
    found = 0
    for _ in range(n * 2):
        candidate -= datetime.timedelta(days=1)
        if is_trading_day(candidate):
            found += 1
            if found == n:
                return candidate
    raise RuntimeError("could not find %d prior trading days from %s" % (n, date))


def trading_days_between(start: datetime.date, end: datetime.date) -> int:
    count = 0
    current = start
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += datetime.timedelta(days=1)
    return count


def clear_cache() -> None:
    global _holiday_cache
    _holiday_cache = None
