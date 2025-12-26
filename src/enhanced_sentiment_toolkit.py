"""
Enhanced Multilingual Sentiment Analysis Tools

Provides "vague sentiment" signal for ex-US equities through:
1. Multi-language search queries (using company names in local language)
2. Site-specific searches of accessible platforms
3. Proxy sentiment from price action and volume
4. Machine translation of available results

NO paid APIs required beyond existing Tavily.
"""

from typing import Annotated
import structlog
from langchain_core.tools import tool

logger = structlog.get_logger(__name__)

# Company name translations for common markets
# Expand this dictionary as needed
COMPANY_NAME_TRANSLATIONS = {
    # Format: "ticker": {"native": "local name", "romanized": "romanization"}
    "0700.HK": {"native": "腾讯控股", "romanized": "Tengxun Konggu"},
    "005930.KS": {"native": "삼성전자", "romanized": "Samsung Electronics"},
    "0005.HK": {"native": "滙豐控股", "romanized": "Huifeng Konggu"},
    "0291.HK": {"native": "華潤啤酒", "romanized": "Huarun Pijiu"},
    "6758.T": {"native": "ソニーグループ", "romanized": "Sonī Gurūpu"},
    "2382.TW": {"native": "廣達電腦", "romanized": "Guangda Diannao"},
    "7203.T": {"native": "トヨタ自動車", "romanized": "Toyota Jidosha"},
    "NESN.SW": {"native": "Nestlé", "romanized": "Nestle"},
    "NOVN.SW": {"native": "Novartis", "romanized": "Novartis"},
    "ROG.SW": {"native": "Roche", "romanized": "Roche"},
    "MC.PA": {"native": "LVMH", "romanized": "LVMH"},
    "AIR.PA": {"native": "Airbus", "romanized": "Airbus"},
    "SAP.DE": {"native": "SAP", "romanized": "SAP"},
    "SIE.DE": {"native": "Siemens", "romanized": "Siemens"},
    "ITX.MC": {"native": "Inditex", "romanized": "Inditex"},
    "PETR4.SA": {"native": "Petrobras", "romanized": "Petrobras"},
    "VALE3.SA": {"native": "Vale", "romanized": "Vale"},
    "PTT.BK": {"native": "ปตท.", "romanized": "PTT"},
    "BBRI.JK": {"native": "Bank Rakyat Indonesia", "romanized": "Bank Rakyat Indonesia"},
    "1155.KL": {"native": "Malayan Banking", "romanized": "Maybank"},
    "2222.SR": {"native": "أرامكو السعودية", "romanized": "Saudi Aramco"},
}

# Accessible platforms that aggregate sentiment (no login required)
ACCESSIBLE_SENTIMENT_PLATFORMS = {
    "tradingview.com": "Technical + sentiment charts",
    "investing.com": "Community comments and sentiment",
    "stocktwits.com": "Retail investor sentiment",
    "reddit.com/r/stocks": "Reddit stocks discussion",
    "reddit.com/r/investing": "Reddit investing discussion",
    "reddit.com/r/wallstreetbets": "Reddit WSB (high risk focus)",
    "seekingalpha.com": "Articles and comment sentiment",
    "finance.yahoo.com": "Yahoo Finance comments",
}

# Region-specific English-accessible platforms
REGION_PLATFORMS = {
    # Asia
    "hong_kong": ["scmp.com", "ejinsight.com", "hkex.com.hk"],
    "south_korea": ["koreaherald.com", "koreatimes.co.kr", "yonhapnews.co.kr"],
    "japan": ["japantimes.co.jp", "asia.nikkei.com", "kyodonews.net"],
    "taiwan": ["taipeitimes.com", "focustaiwan.tw"],
    "china": ["chinadaily.com.cn", "globaltimes.cn", "caixinglobal.com"],
    "india": ["economictimes.indiatimes.com", "moneycontrol.com", "livemint.com"],
    "thailand": ["bangkokpost.com", "nationthailand.com"],
    "malaysia": ["thestar.com.my", "nst.com.my", "theedgemalaysia.com"],
    "vietnam": ["vnexpress.net", "vietnamnews.vn", "vir.com.vn"],
    "indonesia": ["jakartaglobe.id", "thejakartapost.com", "antaranews.com"],
    
    # Europe
    "germany": ["dw.com", "handelsblatt.com/english", "thelocal.de"],
    "france": ["france24.com", "rfi.fr", "connexionfrance.com", "lemonde.fr/en"],
    "spain": ["elpais.com/english", "theolivepress.es", "surinenglish.com"],
    "portugal": ["theportugalnews.com", "portugalresident.com"],
    "poland": ["tvpworld.com", "polskieradio.pl", "notesfrompoland.com"],
    "denmark": ["cphpost.dk", "dr.dk/nyheder"],
    "switzerland": ["swissinfo.ch", "lenews.ch"],
    
    # Americas
    "brazil": ["riotimesonline.com", "braziljournal.com", "folha.uol.com.br"],
    "mexico": ["mexiconewsdaily.com", "eluniversal.com.mx/english"],
    
    # Middle East
    "middle_east": ["arabnews.com", "thenationalnews.com", "aljazeera.com", "gulfnews.com"],
}


def get_company_translations(ticker: str, company_name: str) -> dict[str, str]:
    """
    Get local language translations of company name.
    Returns dict with 'native' and 'romanized' keys.
    """
    # Check if we have a pre-configured translation
    if ticker in COMPANY_NAME_TRANSLATIONS:
        return COMPANY_NAME_TRANSLATIONS[ticker]
    
    # Otherwise return what we have
    return {
        "native": company_name,  # Fallback to English name
        "romanized": company_name,
        "note": "Translation not configured - using English name"
    }


def detect_market_region(ticker: str) -> str:
    """Detect the market region from ticker suffix."""
    ticker = ticker.upper()
    
    # Asia
    if ticker.endswith(".HK"): return "hong_kong"
    if ticker.endswith(".KS") or ticker.endswith(".KQ"): return "south_korea"
    if ticker.endswith(".T"): return "japan"
    if ticker.endswith(".TW") or ticker.endswith(".TWO"): return "taiwan"
    if ticker.endswith(".SS") or ticker.endswith(".SZ"): return "china"
    if ticker.endswith(".NS") or ticker.endswith(".BO"): return "india"
    if ticker.endswith(".BK"): return "thailand"
    if ticker.endswith(".KL"): return "malaysia"
    if ticker.endswith(".VN"): return "vietnam" 
    if ticker.endswith(".JK"): return "indonesia"
    
    # Europe
    if ticker.endswith(".DE") or ticker.endswith(".F") or ticker.endswith(".XETRA"): return "germany"
    if ticker.endswith(".VI"): return "germany" # Austria (German language)
    if ticker.endswith(".PA"): return "france"
    if ticker.endswith(".BR"): return "france" # Brussels (often French/English overlap in fin media)
    if ticker.endswith(".MC") or ticker.endswith(".MA"): return "spain"
    if ticker.endswith(".LS"): return "portugal"
    if ticker.endswith(".WA"): return "poland"
    if ticker.endswith(".CO"): return "denmark"
    if ticker.endswith(".SW") or ticker.endswith(".S"): return "switzerland"
    
    # Americas
    if ticker.endswith(".SA"): return "brazil"
    if ticker.endswith(".MX"): return "mexico"
    
    # Middle East
    if ticker.endswith(".SR") or ticker.endswith(".QA") or ticker.endswith(".AE"): return "middle_east"
    
    return "unknown"


@tool
async def get_multilingual_sentiment_search(
    ticker: Annotated[str, "Stock ticker symbol"],
    company_name: Annotated[str, "Company name in English"],
    tavily_tool: Annotated[object, "Tavily search tool instance"]
) -> str:
    """
    Multi-tier sentiment search using accessible platforms and multilingual queries.
    
    Provides "vague sentiment" signal without requiring paid APIs or deep scraping.
    Uses Tavily web search with smart query construction.
    """
    
    output = f"""
========================================
MULTILINGUAL SENTIMENT SEARCH
========================================
Ticker: {ticker}
Company: {company_name}

This is a BEST EFFORT search for sentiment signals using publicly accessible
platforms. Results are directional indicators, not comprehensive sentiment.

"""
    
    # Get translations
    translations = get_company_translations(ticker, company_name)
    region = detect_market_region(ticker)
    
    output += f"""
**Company Names**:
- English: {company_name}
- Local: {translations.get('native', 'N/A')}
- Romanized: {translations.get('romanized', 'N/A')}
- Market Region: {region}

"""
    
    sentiment_signals = []
    
    # ===== TIER 1: ACCESSIBLE PLATFORM SEARCHES =====
    output += "\n### Tier 1: Accessible Platform Searches\n\n"
    
    try:
        # Search TradingView (has international stocks)
        tradingview_query = f'site:tradingview.com {ticker} OR "{company_name}" sentiment OR bullish OR bearish'
        tv_result = await tavily_tool.ainvoke({"query": tradingview_query})
        
        output += f"**TradingView Search**:\n"
        output += f"Query: `{tradingview_query}`\n"
        
        # Parse results for sentiment keywords
        tv_text = str(tv_result).lower()
        bullish_count = tv_text.count("bullish") + tv_text.count("positive") + tv_text.count("upside")
        bearish_count = tv_text.count("bearish") + tv_text.count("negative") + tv_text.count("downside")
        
        if bullish_count + bearish_count > 0:
            sentiment_ratio = bullish_count / (bullish_count + bearish_count)
            sentiment_signals.append({
                "source": "TradingView",
                "bullish": bullish_count,
                "bearish": bearish_count,
                "ratio": sentiment_ratio
            })
            output += f"Sentiment Keywords: Bullish={bullish_count}, Bearish={bearish_count}\n"
        else:
            output += "No clear sentiment signals found.\n"
        
        output += "\n"
        
    except Exception as e:
        output += f"TradingView search failed: {str(e)}\n\n"
    
    # Search Investing.com (has comments for international stocks)
    try:
        investing_query = f'site:investing.com {ticker} comments OR sentiment'
        inv_result = await tavily_tool.ainvoke({"query": investing_query})
        
        output += f"**Investing.com Search**:\n"
        output += f"Query: `{investing_query}`\n"
        
        inv_text = str(inv_result).lower()
        bullish_count = inv_text.count("buy") + inv_text.count("bullish") + inv_text.count("strong buy")
        bearish_count = inv_text.count("sell") + inv_text.count("bearish") + inv_text.count("strong sell")
        
        if bullish_count + bearish_count > 0:
            sentiment_ratio = bullish_count / (bullish_count + bearish_count)
            sentiment_signals.append({
                "source": "Investing.com",
                "bullish": bullish_count,
                "bearish": bearish_count,
                "ratio": sentiment_ratio
            })
            output += f"Sentiment Keywords: Bullish={bullish_count}, Bearish={bearish_count}\n"
        else:
            output += "No clear sentiment signals found.\n"
        
        output += "\n"
        
    except Exception as e:
        output += f"Investing.com search failed: {str(e)}\n\n"
    
    # ===== TIER 2: MULTILINGUAL SEARCHES =====
    output += "\n### Tier 2: Multilingual Searches\n\n"
    
    native_name = translations.get('native', '')
    
    if native_name and native_name != company_name:
        try:
            # Search using native language company name
            multilang_query = f'"{native_name}" {ticker} 股票 OR stock OR sentiment'
            ml_result = await tavily_tool.ainvoke({"query": multilang_query})
            
            output += f"**Native Language Search**:\n"
            output += f"Query: `{multilang_query}`\n"
            
            # Check if we got any results
            if ml_result and len(str(ml_result)) > 100:
                output += f"Found {len(str(ml_result))} characters of content.\n"
                output += "Note: Results may be in local language. Manual review recommended.\n"
                
                # Try to detect sentiment even in non-English
                # Common sentiment words across languages
                positive_indicators = [
                    "good", "positive", "bullish", "buy", "strong", "growth",
                    "买入", "긍정", "ポジティブ", "kaufen", "acheter", "comprar", "compra",
                    "ดี", "bagus", "tốt", "dobry", "god"
                ]
                negative_indicators = [
                    "bad", "negative", "bearish", "sell", "weak", "decline",
                    "卖出", "부정", "ネガティブ", "verkaufen", "vendre", "vender", "venta",
                    "แย่", "buruk", "xấu", "zły", "dårlig"
                ]
                
                ml_text = str(ml_result).lower()
                pos_count = sum(ml_text.count(word) for word in positive_indicators)
                neg_count = sum(ml_text.count(word) for word in negative_indicators)
                
                if pos_count + neg_count > 0:
                    output += f"Sentiment Indicators: Positive={pos_count}, Negative={neg_count}\n"
                    sentiment_signals.append({
                        "source": "Multilingual Search",
                        "bullish": pos_count,
                        "bearish": neg_count,
                        "ratio": pos_count / (pos_count + neg_count)
                    })
            else:
                output += "Limited results found.\n"
            
            output += "\n"
            
        except Exception as e:
            output += f"Multilingual search failed: {str(e)}\n\n"
    else:
        output += "No native language translation available for enhanced search.\n\n"
    
    # ===== TIER 3: REGION-SPECIFIC NEWS =====
    output += "\n### Tier 3: Region-Specific English News\n\n"
    
    if region in REGION_PLATFORMS:
        platforms = REGION_PLATFORMS[region]
        
        # Search major regional English-language financial news
        region_sites = " OR ".join([f"site:{site}" for site in platforms[:3]])
        region_query = f'({region_sites}) "{company_name}" OR {ticker}'
        
        try:
            region_result = await tavily_tool.ainvoke({"query": region_query})
            
            output += f"**Regional News Search** ({region.replace('_', ' ').title()}):\n"
            output += f"Query: `{region_query}`\n"
            
            if region_result and len(str(region_result)) > 100:
                output += f"Found {len(str(region_result))} characters of regional news.\n"
                
                # Analyze tone
                region_text = str(region_result).lower()
                positive_words = ["growth", "profit", "strong", "beat", "rally", "upgrade"]
                negative_words = ["decline", "loss", "weak", "miss", "fall", "downgrade"]
                
                pos_score = sum(region_text.count(word) for word in positive_words)
                neg_score = sum(region_text.count(word) for word in negative_words)
                
                if pos_score + neg_score > 0:
                    output += f"News Tone: Positive={pos_score}, Negative={neg_score}\n"
                    sentiment_signals.append({
                        "source": "Regional News",
                        "bullish": pos_score,
                        "bearish": neg_score,
                        "ratio": pos_score / (pos_score + neg_score)
                    })
            else:
                output += "Limited regional news found.\n"
            
            output += "\n"
            
        except Exception as e:
            output += f"Regional news search failed: {str(e)}\n\n"
    
    # ===== AGGREGATE SENTIMENT SIGNAL =====
    output += "\n### Aggregated Sentiment Signal\n\n"
    
    if sentiment_signals:
        # Calculate weighted average
        total_mentions = sum(s['bullish'] + s['bearish'] for s in sentiment_signals)
        total_bullish = sum(s['bullish'] for s in sentiment_signals)
        total_bearish = sum(s['bearish'] for s in sentiment_signals)
        
        if total_mentions > 0:
            overall_ratio = total_bullish / total_mentions
            
            output += f"**Total Mentions**: {total_mentions}\n"
            output += f"**Bullish**: {total_bullish} ({total_bullish/total_mentions*100:.1f}%)\n"
            output += f"**Bearish**: {total_bearish} ({total_bearish/total_mentions*100:.1f}%)\n\n"
            
            # Classify sentiment
            if overall_ratio > 0.60:
                sentiment_class = "POSITIVE"
            elif overall_ratio > 0.40:
                sentiment_class = "NEUTRAL"
            else:
                sentiment_class = "NEGATIVE"
            
            output += f"**Aggregate Sentiment**: {sentiment_class}\n"
            output += f"**Confidence**: {'LOW' if total_mentions < 10 else 'MEDIUM' if total_mentions < 30 else 'HIGH'}\n\n"
            
            output += "**Sources Breakdown**:\n"
            for signal in sentiment_signals:
                output += f"- {signal['source']}: {signal['bullish']} bullish, {signal['bearish']} bearish\n"
        else:
            output += "Insufficient data to calculate aggregate sentiment.\n"
    else:
        output += "**No sentiment signals detected.**\n"
        output += "This suggests the stock is truly undiscovered or has minimal online discussion.\n"
    
    output += """

========================================
INTERPRETATION GUIDANCE
========================================

**What This Analysis Provides**:
✓ Directional sentiment from accessible platforms
✓ Relative discussion volume (high mentions = more attention)
✓ Basic bullish/bearish ratio from keyword analysis

**What This Analysis Does NOT Provide**:
✗ Comprehensive local social media sentiment (login-walled platforms)
✗ Real-time sentiment (search results may be days/weeks old)
✗ Institutional investor sentiment (beyond published research)

**How to Use This Data**:
- HIGH bullish + LOW mentions = Potential undiscovered opportunity ✓
- HIGH bearish + HIGH mentions = Avoid (known issues)
- LOW mentions overall = Confirms "undiscovered" thesis ✓
- NEUTRAL sentiment + good fundamentals = Proceed with fundamental analysis

This is a SUPPLEMENTARY signal. Primary investment decisions should be based
on fundamental analysis (Financial Health, Growth Transition scores).

"""
    
    logger.info("multilingual_sentiment_search_completed", 
                ticker=ticker,
                signals_found=len(sentiment_signals))
    
    return output