"""Exchange-session checks; retrieval time is never a quote timestamp."""

import pandas as pd


def latest_completed_session(market: str, now=None) -> str:
    import exchange_calendars as calendars

    if market not in {"cn", "us"}:
        raise ValueError("market must be cn or us")
    instant = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if instant.tzinfo is None:
        raise ValueError("scan time must include timezone")
    instant = instant.tz_convert("UTC")
    calendar = calendars.get_calendar("XSHG" if market == "cn" else "XNYS")
    day = instant.tz_localize(None).normalize()
    if day < calendar.first_session or day > calendar.last_session:
        raise RuntimeError(
            "Trading calendar does not cover scan date; refresh calendar dependency"
        )
    schedule = calendar.schedule.loc[day - pd.Timedelta(days=40) : day]
    completed = schedule[schedule["close"] <= instant]
    if completed.empty:
        raise RuntimeError("Cannot establish latest completed session")
    return completed.index[-1].date().isoformat()
