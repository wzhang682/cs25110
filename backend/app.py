# backend/app.py
from fastapi import FastAPI, WebSocket, Request
from fastapi.templating import Jinja2Templates
from pathlib import Path
import asyncio
from backend.terminal_manager import TerminalManager
from backend.message_parser import parse_output
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import requests
from fastapi import HTTPException
import os
from dotenv import load_dotenv

load_dotenv()
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

app = FastAPI()

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/results", response_class=HTMLResponse)
async def results(request: Request, symbols: str, start_date: str, end_date: str):
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "symbols": symbols,
            "start_date": start_date,
            "end_date": end_date,
        },
    )


@app.get("/api/market")
async def get_market_data():

    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=500, detail="Finnhub API key not set")

    symbols = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "NVDA",  # Nvidia
    "TSLA",  # Tesla
    "AMZN",  # Amazon
    "GOOGL", # Alphabet
    "META",  # Meta
    "AVGO",  # Broadcom
    "AMD",   # AMD
    "NFLX",  # Netflix
    "ADBE",  # Adobe
    "INTC"   # Intel
]
    results = []

    for symbol in symbols:

        url = (
            f"https://finnhub.io/api/v1/quote"
            f"?symbol={symbol}"
            f"&token={FINNHUB_API_KEY}"
        )

        response = requests.get(url)
        data = response.json()

        if "c" not in data:
            continue

        results.append({
            "symbol": symbol,
            "regularMarketPrice": data["c"],
            "regularMarketChange": data["d"],
            "regularMarketChangePercent": data["dp"]
        })

    return results


@app.websocket("/ws/run")
async def run_analysis_ws(ws: WebSocket):
    await ws.accept()

    try:
        data = await ws.receive_json()
        symbols = data.get("symbols")
        start_date = data.get("start_date")
        end_date = data.get("end_date")

        if not symbols or not start_date or not end_date:
            await ws.send_json({
                "type": "error",
                "msg": "Missing symbols / start_date / end_date"
            })
            await ws.close()
            return

        tm = TerminalManager()
        tm.start()

        tm.send_input(symbols)
        tm.send_input(start_date)
        tm.send_input(end_date)

        while True:
            await asyncio.sleep(0.05)
            lines = tm.get_output()

            for line in lines:
                parsed = parse_output(line)
                await ws.send_json(parsed)

            if tm.is_finished() and not tm.get_output():
                break

        await ws.send_json({"type": "done"})
        await ws.close()

    except Exception as e:
        await ws.send_json({
            "type": "error",
            "msg": str(e)
        })
        await ws.close()
