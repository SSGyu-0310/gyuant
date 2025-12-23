from typing import Dict, Iterable, List, Tuple

YAHOO_TO_FMP: Dict[str, str] = {
    "BTC-USD": "BTCUSD",
    "ETH-USD": "ETHUSD",
    "GC=F": "GCUSD",
    "CL=F": "CLUSD",
    "DX-Y.NYB": "DXY",
}


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def to_fmp_symbol(symbol: str) -> str:
    sym = normalize_symbol(symbol)
    mapped = YAHOO_TO_FMP.get(sym)
    if mapped:
        return mapped
    if "-" in sym and sym.count("-") == 1:
        left, right = sym.split("-")
        if right.isalpha() and len(right) == 1:
            return f"{left}.{right}"
    return sym


def from_fmp_symbol(symbol: str) -> str:
    sym = normalize_symbol(symbol)
    for yahoo, fmp in YAHOO_TO_FMP.items():
        if fmp == sym:
            return normalize_symbol(yahoo)
    if "." in sym and sym.count(".") == 1:
        left, right = sym.split(".")
        if right.isalpha() and len(right) == 1:
            return f"{left}-{right}"
    return sym


def map_symbols_to_fmp(symbols: Iterable[str]) -> Tuple[List[str], Dict[str, str]]:
    fmp_symbols: List[str] = []
    reverse: Dict[str, str] = {}
    for symbol in symbols:
        original = normalize_symbol(symbol)
        fmp_symbol = to_fmp_symbol(original)
        fmp_symbols.append(fmp_symbol)
        if fmp_symbol not in reverse:
            reverse[fmp_symbol] = original
    return fmp_symbols, reverse
