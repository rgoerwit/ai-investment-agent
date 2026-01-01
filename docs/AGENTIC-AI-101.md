# Agentic AI 101: A Practical Introduction

> **Or: How I Learned to Stop Worrying and Let AI Agents Debate My Stock Picks**

This guide explains what agentic AI is, why it matters, and how this repository demonstrates its potential to democratize fields with high barriers to entry—like sophisticated equity research.

---

## What Is Agentic AI?

### The Simple Definition

**Agentic AI** is an autonomous artificial intelligence system that plans, executes, and adapts actions to achieve complex goals without constant human intervention. Unlike a chatbot that simply responds to prompts, an AI agent can:

1. **Maintain state** - Remember context across multiple interactions
2. **Use tools** - Call functions, fetch data, perform calculations
3. **Make decisions** - Choose different paths based on conditions
4. **Collaborate** - Work with other agents to solve problems

Think of it this way: ChatGPT is like asking a very knowledgeable person a question. Agentic AI is like having a **team of specialists** who collaborate, debate, use tools, and deliver a synthesized decision.

### Multi-Agent Systems: The Team Approach

[Multi-agent systems](https://www.kubiya.ai/blog/what-are-multi-agent-systems-in-ai) comprise multiple autonomous AI agents collaborating to solve tasks that are difficult for a single model. Instead of one AI trying to do everything, you have:

- **Specialized agents** - Each with a specific role and expertise
- **Coordination** - Agents communicate and share information
- **Debate** - Different perspectives challenge each other
- **Synthesis** - A final decision emerges from collective reasoning

[Microsoft defines](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/12/04/multi-agentic-ai-unlocking-the-next-wave-of-business-transformation/) agentic AI as "the pairing of traditional software strengths—such as workflows, state, and tool use—with the adaptive reasoning capabilities of large language models."

In 2025, we're seeing rapid adoption: industry forecasts suggest that by 2028, [at least 15% of daily business decisions will be made autonomously by agentic AI](https://ioni.ai/post/multi-ai-agents-in-2025-key-insights-examples-and-challenges).

---

## Why Does This Matter?

### The Problem: Gatekeepers and Barriers

Consider equity research. A retail investor wants to analyze international stocks—say, a semiconductor company in Taiwan or a bank in Hong Kong. What are their options?

1. **Bloomberg Terminal** - $24,000/year, designed for institutions
2. **Premium research services** - $2,000-$10,000/year, often US-focused
3. **Free tools** - Yahoo Finance, limited depth, no synthesis
4. **DIY spreadsheets** - Time-consuming, inconsistent, expertise-dependent

This is gatekeeping. Sophisticated financial analysis has been a **luxury good**, accessible only to institutions or wealthy individuals. The same pattern exists in healthcare (specialist consultations), legal services (multi-lawyer teams), and enterprise software (dedicated analyst teams).

### The Solution: Democratization Through AI

What if you could simulate an institutional research team using AI?

- **Market Analyst** - Technical analysis, price trends, volume
- **Fundamentals Analyst** - Financial metrics, balance sheet health
- **News Analyst** - Recent events, catalysts, risks
- **Sentiment Analyst** - Social media, retail investor mood
- **Bull Researcher** - Best-case arguments for buying
- **Bear Researcher** - Worst-case arguments for selling
- **Risk Analysts** - Conservative, neutral, and aggressive perspectives
- **Portfolio Manager** - Synthesizes all viewpoints, makes final call

This isn't science fiction. This is what this repository does—using free or cheap-tier AI APIs and open-source tools.

---

## The Technology: LangGraph and Friends

### LangGraph: Orchestrating Agent Workflows

[LangGraph](https://blog.langchain.com/langgraph/) is a framework for building stateful, multi-agent applications. It models agent interactions as **directed graphs**, where:

- **Nodes** = Individual agents (functions that receive state, do work, return updated state)
- **Edges** = Connections between agents (who talks to whom)
- **Conditional routing** = Dynamic flow (if X happens, go to agent A; otherwise, agent B)
- **State** = Shared context that gets updated as the graph executes

[LangGraph operates through message passing](https://medium.com/@umang91999/beginners-guide-to-langchain-graphs-states-nodes-and-edges-3ca7f3de5bfe): when a node completes its operation, it sends messages along edges to other nodes, which execute their functions and pass results to the next set of nodes.

#### Why LangGraph? (vs AutoGen, CrewAI, etc.)

This project uses LangGraph because financial decisions require **precision**:

- **Explicit control** - You define exactly when validation happens, when debate stops, when to short-circuit (reject early)
- **Conditional routing** - Red-flag validator can skip debate if a stock has catastrophic leverage
- **Debugging** - LangSmith traces show exact state transitions (critical for production)
- **Scalability** - Stateful graphs can be deployed to cloud infrastructure

Other frameworks (AutoGen, CrewAI) trade control for ease of use. For a research tool where errors could cost real money, LangGraph's "low-level but powerful" approach wins.

### LangSmith: Observability and Debugging

[LangSmith](https://www.langchain.com/langsmith) provides LLM-native observability—tracing every step of your agent workflow so you can debug non-deterministic behavior.

**Key features:**

- **Tracing** - Record every LLM call, tool invocation, and state transition
- **Real-time monitoring** - Track latency, error rates, token usage
- **Debugging** - See exactly where an agent went wrong (which prompt, which tool call)
- **Production insights** - Dashboard for usage patterns, costs, performance

If you're building agentic systems, [LangSmith is essentially mandatory](https://murf.ai/blog/llm-observability-with-langsmith). Enable it with one environment variable: `LANGSMITH_TRACING=true`. The free tier is sufficient for development.

### ChromaDB: Memory That Doesn't Leak

[ChromaDB](https://www.trychroma.com/) is a vector database for storing agent memory—past analyses, learned patterns, context.

**The challenge:** If you analyze 100 stocks, you don't want agent memory from Samsung's chip shortage bleeding into HSBC's bank analysis. This is called **memory contamination**, and it's a common failure mode in RAG (Retrieval-Augmented Generation) systems.

**The solution (in this repo):** Ticker-specific collections. When analyzing `0005.HK`, create isolated ChromaDB collections:

- `0005_HK_bull_memory`
- `0005_HK_bear_memory`
- `0005_HK_portfolio_manager_memory`

Each analysis gets its own namespace. Simple but critical.

---

## This Repository: A Case Study in Democratization

### The Philosophy

I built this system because I was frustrated. As a retail investor interested in international equities, I had three bad options:

1. **Pay $24k/year for Bloomberg** (no thanks)
2. **Trust free stock screeners** (superficial, US-biased)
3. **Manually research each stock** (time-consuming, inconsistent)

What I wanted was an **institutional-quality research team** that could:

- Handle obscure tickers (Taiwan semiconductors, Hong Kong banks)
- Apply disciplined investment criteria (no emotional decisions)
- Synthesize multiple perspectives (bull case, bear case, risk assessment)
- Cost $0 marginal per analysis (free-tier APIs)

So I built one. Not because it's perfect—it's not. But because it's **accessible**.

### The Architecture: How Agents Collaborate

Here's how a single stock analysis flows through the system:

```
User: "Analyze 0005.HK"
    ↓
[Parallel Data Gathering]
  ├─ Market Analyst (technical analysis)
  ├─ Fundamentals Analyst (financial metrics)
  ├─ News Analyst (recent events)
  └─ Sentiment Analyst (social media)
    ↓
[Financial Validator]
    ├─ RED FLAGS? (D/E > 500%, earnings quality issues)
    │   ├─ YES → Skip debate, recommend SELL
    │   └─ NO → Continue to debate
    ↓
[Research Manager]
    └─ Synthesize findings, identify themes
    ↓
[Adversarial Debate - Round 1]
    ├─ Bull Researcher: "Here's the upside..."
    └─ Bear Researcher: "Wait, consider these risks..."
    ↓
[Adversarial Debate - Round 2]
    ├─ Bull: "I concede X, but counter with Y..."
    └─ Bear: "Agreed on Y, but Z is still concerning..."
    ↓
[Risk Assessment Team]
    ├─ Conservative Analyst: "Recommend 0% allocation"
    ├─ Neutral Analyst: "Recommend 2% allocation"
    └─ Aggressive Analyst: "Recommend 3% allocation"
    ↓
[Portfolio Manager]
    └─ Synthesize all perspectives
    └─ Apply thesis criteria (GARP strategy)
    └─ FINAL DECISION: BUY / HOLD / SELL + position size
```

**Why this works better than a single prompt:**

1. **Specialization** - Each agent focuses on one domain (no "jack of all trades")
2. **Adversarial debate** - Bull/Bear researchers catch each other's blind spots
3. **Multi-perspective risk** - Conservative/neutral/aggressive analysts represent different risk tolerances
4. **Deterministic gates** - Financial Validator uses hard thresholds (D/E > 500% = auto-reject, no rationalization)
5. **Thesis enforcement** - Portfolio Manager applies quantitative criteria (growth score ≥ 50%, liquidity ≥ $500k)

### Real-World Example: Red-Flag Detection

Consider a stock with these financials:

- **P/E ratio:** 12 (cheap!)
- **Revenue growth:** 15% YoY (strong!)
- **Debt-to-Equity ratio:** 650% (uh oh...)
- **Interest coverage:** 1.2x (barely paying interest)

**What a single-prompt LLM might say:**

> "This stock looks promising with strong revenue growth and an attractive P/E ratio. Consider buying."

**What this multi-agent system does:**

1. **Fundamentals Analyst** extracts metrics into structured DATA_BLOCK
2. **Financial Validator** detects:
   - ❌ EXTREME_LEVERAGE (D/E 650% > 500% threshold)
   - ❌ REFINANCING_RISK (interest coverage 1.2x < 2.0x with high leverage)
3. **Conditional routing** skips debate entirely
4. **Portfolio Manager** receives validator recommendation: AUTO_REJECT
5. **Final decision:** SELL - "Leverage risk makes this uninvestable per GARP criteria"

This saves ~60% of token costs (no debate needed for doomed stocks) and prevents "hope bias" (where debate might rationalize fatal flaws).

### The GARP Thesis: Disciplined Investing

This system isn't a black box. It enforces a specific investment strategy: **GARP (Growth at a Reasonable Price)**. The alignment with this thesis is visualized via a **6-Axis Radar Chart**:

1.  **Health** - Financial health composite (incorporating D/E and ROA).
2.  **Growth** - Growth transition score alignment.
3.  **Value** - Valuation metrics (P/E, PEG).
4.  **Regulatory** - Compliance risks (PFIC, VIE, CMIC).
5.  **Undiscovered** - Institutional/Analyst coverage gaps.
6.  **Jurisdiction** - Geopolitical and exchange stability.

**Designing for Reliability**: A common failure mode in LLM applications is "fragile parsing"—relying on regex to hunt through long narrative text for a single number. This system mitigates this by using a versioned **Structured DATA_BLOCK (v7.4)**. The LLM is forced to output critical metrics in a strict schema, which the Python plotting layer then extracts deterministically. This turns the LLM into a reliable data-producer for the visualization engine.

**Hard requirements** (automatic SELL if violated):

- Financial Health Score ≥ 50% (profitable, positive cash flow, manageable debt)
- Growth Score ≥ 50% (revenue/EPS growth, margin expansion, turnaround trajectory)
- Liquidity ≥ $500k USD daily volume (tradeable without slippage)
- Analyst Coverage < 15 (undiscovered by mainstream US research)

**Soft factors** (influence risk scoring):

- Valuation (P/E ≤ 18, PEG ≤ 1.2, P/B ≤ 1.4)
- US Revenue exposure (prefer 25-35% for diversification)
- Qualitative risks (geopolitical, industry headwinds, management issues)

**Philosophy:** Find mid-cap international stocks transitioning from value (undervalued) to growth (expansion phase) before US analysts discover them. This is one of the few ways retail investors can compete with institutional funds—by looking where others aren't.

### Performance: The Honest Truth

**Speed:**
- Quick mode: 2-4 minutes per ticker
- Standard mode: 5-10 minutes per ticker
- Batch (300 tickers): 12-24 hours on free-tier Gemini (slower with rate limits)

**Cost:**
- Free tier Gemini: $0 per analysis (15 RPM limit, lots of retries)
- Paid tier Gemini: ~$0.05-$0.15 per analysis (depending on model choice)

**Accuracy:**
- Quantitative metrics: ~80% match to Bloomberg/Yahoo Finance (when data available)
- Qualitative synthesis: Good for generating hypotheses, **not for final decisions**
- Weaknesses: Historical data only, free APIs have gaps, sentiment analysis limited

**Reality check:** This is a **research tool**, not an oracle. Use it to generate a shortlist for deeper due diligence. Always verify with SEC filings, earnings calls, and independent sources. Never blindly trust AI—or any analyst, human or machine.

---

## Learning from This Repository

### For AI Engineers: Production Patterns

This codebase demonstrates production-grade patterns often missing from tutorials:

1. **Error handling** - Multi-source data fallback (yfinance → YahooQuery → FMP → EODHD → Tavily)
2. **Rate limiting** - Exponential backoff, configurable RPM limits, graceful degradation
3. **Memory isolation** - Ticker-specific ChromaDB collections prevent contamination
4. **Prompt versioning** - JSON files with metadata tracking (`prompts/fundamentals_analyst.json`)
5. **Testing** - 37 test files covering unit, integration, edge cases (rate limits, missing data, malformed responses)
6. **Observability** - LangSmith tracing, structured logging, token usage tracking

### For Investors: A New Tool

This system provides:

- ✅ Systematic discipline (no emotional decisions)
- ✅ Multi-perspective analysis (bull/bear/risk synthesis)
- ✅ International coverage (HK, JP, TW, KR equities)
- ✅ Reproducible research (same inputs → same outputs)
- ✅ Full transparency (every decision explained with supporting data)

But remember:

- ❌ Not real-time (historical data, 15-min delays on free APIs)
- ❌ Not an execution engine (you must place trades manually)
- ❌ Not financial advice (this is a research tool, DYOR)
- ❌ Not a substitute for judgment (AI can hallucinate, data can be wrong)

### For Educators: A Teaching Case

This repository is designed to teach agentic AI through a **real-world problem**:

- **Motivation** - Democratizing finance (relatable goal)
- **Complexity** - Multi-agent coordination, conditional routing, state management
- **Practicality** - Actually useful output (stock analysis reports)
- **Transparency** - Open source, documented architecture, tested code

Use this in a course on:

- LangGraph and stateful AI workflows
- Multi-agent systems and debate patterns
- RAG (retrieval-augmented generation) with ChromaDB
- Prompt engineering (versioned prompts with metadata)
- Production ML (error handling, testing, observability)

---

## When to Use Agentic AI (vs Traditional Code)

### ✅ Use Agents When:

1. **Task has multiple distinct sub-problems** (research, analysis, synthesis)
2. **Different perspectives improve decisions** (bull vs bear debate)
3. **Need systematic quality checks** (validation gates, thesis enforcement)
4. **Ambiguity requires reasoning** (interpreting news, assessing qualitative risks)
5. **Workflow might change** (easy to add new agents or modify routing)

### ❌ Don't Use Agents When:

1. **Task is simple** (single prompt works fine)
2. **Deterministic logic suffices** (traditional code is faster and cheaper)
3. **Latency is critical** (agents take multiple LLM calls)
4. **No benefit from specialization** (one generalist agent is enough)
5. **Budget is extremely tight** (LLM calls cost money, even on free tiers with rate limits)

### This Project's Choice:

Equity research is **ambiguous** (qualitative factors), **multi-faceted** (fundamentals + technicals + news + sentiment), and benefits from **adversarial debate** (bull vs bear). Traditional code can't handle "Does this CFO resignation signal governance issues or just normal turnover?" But it **can** handle "Is D/E ratio > 500%?" (deterministic check).

So we use **both**: deterministic validators for hard thresholds, LLM agents for reasoning.

---

## The Bigger Picture: Democratization

### Why This Matters

For decades, sophisticated analysis has been locked behind paywalls and credentials:

- **Finance** - Bloomberg terminals, analyst teams, proprietary models
- **Healthcare** - Specialist consultations, second opinions, research synthesis
- **Legal** - Multi-lawyer teams for complex cases
- **Enterprise** - Dedicated analyst teams for data-driven decisions

AI won't replace human experts. But it can **level the playing field**:

- A retail investor can run institutional-quality equity research
- A solo developer can build products that previously required a team
- A student can learn advanced patterns without access to expensive courses

This repository is one small example. The code is open source. The APIs are free or cheap. The documentation explains not just "how" but "why."

### The Catch

There are risks:

1. **Over-reliance** - Trusting AI output without verification (dangerous)
2. **Hallucination** - LLMs can fabricate data (always cross-check)
3. **Black-box syndrome** - Not understanding why the AI decided something (explainability matters)
4. **False confidence** - "The AI said buy, so I bought" (this is how people lose money)

The solution isn't to avoid AI—it's to use it **responsibly**:

- Treat AI output as a **starting point**, not a final answer
- Verify quantitative claims with primary sources (SEC filings, earnings reports)
- Understand the system's limitations (historical data, API gaps)
- Maintain human judgment in final decisions

---

## Getting Started with This Repository

### Quick Start (5 Minutes)

1. **Install Poetry** (dependency management)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. **Clone and install**
   ```bash
   git clone https://github.com/rgoerwit/investment-agent-public.git
   cd investment-agent-public
   poetry install
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env and add:
   #   GOOGLE_API_KEY (free tier: https://aistudio.google.com/)
   #   FINNHUB_API_KEY (free tier: https://finnhub.io/)
   #   TAVILY_API_KEY (free tier: https://tavily.com/)
   ```

4. **Run your first analysis**
   ```bash
   poetry run python -m src.main --ticker 0005.HK
   # Analyzes HSBC Holdings (Hong Kong)
   # Takes ~5-10 minutes
   # Results saved to results/0005_HK_2025-XX-XX.md
   ```

### Next Steps

- **Read the architecture** - See [CLAUDE.md](../CLAUDE.md) for file-by-file breakdown
- **Run tests** - `poetry run pytest tests/ -v`
- **Try batch analysis** - `./scripts/run_tickers.sh` (see [README.md](../README.md))
- **Modify the thesis** - Edit `prompts/portfolio_manager.json` to change criteria
- **Add a new agent** - Follow patterns in `src/agents.py`

---

## Further Reading

### Agentic AI Concepts

- [Multi-Agent Systems in AI: Concepts & Use Cases 2025](https://www.kubiya.ai/blog/what-are-multi-agent-systems-in-ai)
- [What is Agentic AI? Definition and Technical Overview in 2025](https://aisera.com/blog/agentic-ai/)
- [Multi-Agentic AI: Unlocking the Next Wave of Business Transformation](https://www.microsoft.com/en-us/microsoft-cloud/blog/2025/12/04/multi-agentic-ai-unlocking-the-next-wave-of-business-transformation/) (Microsoft)

### LangGraph and LangSmith

- [LangGraph Official Blog](https://blog.langchain.com/langgraph/)
- [Beginner's Guide to LangGraph: Understanding State, Nodes, and Edges](https://medium.com/@kbdhunga/beginners-guide-to-langgraph-understanding-state-nodes-and-edges-part-1-897e6114fa48)
- [LangSmith: Observability for LLM Applications](https://medium.com/@vinodkrane/langsmith-observability-for-llm-applications-ef5aaf6c2e5b)
- [LangSmith Tracing Quickstart](https://docs.langchain.com/langsmith/observability-quickstart)

### This Project

- [README.md](../README.md) - Project overview, quick start, architecture
- [CLAUDE.md](../CLAUDE.md) - Developer guide with file-by-file breakdown
- [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
- [examples/analyze_single_ticker.sh](../examples/analyze_single_ticker.sh) - Minimal example script

---

## Conclusion: An Invitation

This repository isn't just code. It's a **proof of concept** that sophisticated, multi-agent AI systems can be:

- **Accessible** - Free or cheap-tier APIs, runs on a laptop
- **Practical** - Solves a real problem (equity research)
- **Transparent** - Open source, documented, tested
- **Educational** - Demonstrates production patterns, not toy examples

Whether you're an AI engineer learning LangGraph, an investor exploring new tools, or a student studying multi-agent systems, I hope this repository shows what's possible when you combine modern AI frameworks with a desire to **democratize gatekept fields**.

The code is imperfect. The analysis isn't always right. But it's a start.

**Fork it. Improve it. Share it.** Let's build a world where sophisticated analysis isn't a luxury good.

---

**Questions? Feedback? Contributions?**

- **Issues:** [GitHub Issues](https://github.com/rgoerwit/investment-agent-public/issues)
- **Discussions:** [GitHub Discussions](https://github.com/rgoerwit/investment-agent-public/discussions)
- **Contributing:** See [CONTRIBUTING.md](../CONTRIBUTING.md)

---

*Last updated: December 2025*
