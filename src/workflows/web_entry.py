# src/workflows/web_entry.py
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict

from src.workflows.workflow import run_analysis
from src.tools.daily_csv_tool import save_workflow_to_symbol_csv


def get_trading_dates(start_date: str, end_date: str) -> List[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


async def run_web_analysis(
    symbols: List[str],
    start_date: str,
    end_date: str,
    output_cb=None,   
) -> Dict:
    def log(msg: str):
        if output_cb:
            output_cb(msg)
        else:
            print(msg)

    log("Agent - AI Financial Analysis Platform")
    log("=" * 60)

    log(f"Symbols: {', '.join(symbols)}")
    log(f"Analysis Period: {start_date} to {end_date}")

    trading_dates = get_trading_dates(start_date, end_date)
    log(f"Trading Days: {len(trading_dates)}")
    log("Starting workflow execution...")

    success = 0
    failed = 0

    for i, date in enumerate(trading_dates, 1):
        log(f"\n{'='*60}")
        log(f"Day {i}/{len(trading_dates)}: {date}")
        log(f"{'='*60}")

        session_id = f"daily_analysis_{date.replace('-', '_')}"

        try:
            result = await run_analysis(symbols, session_id, date)
            if result.get("success"):
                success += 1
                save_workflow_to_symbol_csv(result, date, "./output/csv")
                log(f"{date} - Analysis completed successfully")
            else:
                failed += 1
                log(f"{date} - Analysis failed")
        except Exception as e:
            failed += 1
            log(f"{date} - Execution error: {e}")

    log("\nANALYSIS COMPLETE")
    log("=" * 60)
    log(f"Successful Runs: {success}")
    log(f"Failed Runs: {failed}")

    return {
        "success": success,
        "failed": failed,
        "total": len(trading_dates),
    }
