import ast
import re

SECTION_HEADERS = {
    "=== Technical Analysis ===": "technical_analysis",
    "=== SAMPLE NEWS ===": "news",
    "=== Trading Analysis ===": "trading_analysis",
    "=== Company Profile ===": "company_profile",
    "=== Portfolio Log ===": "portfolio",
}

_current_section = None

def _clean_numpy_repr(text: str) -> str:
    return re.sub(r"np\.float64\(([^)]+)\)", r"\1", text)

def parse_output(line: str) -> dict:
    global _current_section

    line = line.strip()
    print("[DEBUG] raw line:", repr(line))  
    

    for header, section_type in SECTION_HEADERS.items():
        if line == header:               
            _current_section = section_type
            return {"type": "section_start", "section": section_type}


    if not line:
        return {"type": "skip"}          


    if _current_section and (line.startswith("{") or line.startswith("[")):
        try:
            data = ast.literal_eval(_clean_numpy_repr(line))
            return {"type": _current_section, "data": data}
        except Exception:
            pass

  
    return {"type": "log", "msg": line}












