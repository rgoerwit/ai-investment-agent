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

- **Macro Context Analyst** - Pre-graph regional regime context for News Analyst
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
- **Conditional routing** - `Sync Check` (`src/graph/routing.py`) ends an unassessable run with a diagnostic and sends an issuer-level rejection to a deterministic fast-fail, skipping the debate
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

Here's how a single stock analysis flows through the system. The authoritative
diagram is in [README.md](../README.md); the edges live in `src/graph/builder.py`
and the routers in `src/graph/routing.py`.

```
User: "Analyze 0005.HK"
    ↓
[Pre-Graph Macro Context]
  └─ Macro Context Analyst summarizes cached regional regime background
     and feeds it only to News Analyst
    ↓
[Dispatcher: parallel evidence gathering]
  ├─ Market, Sentiment and News analysts
  ├─ Junior Fundamentals -> Foreign Language + Legal Counsel
  │     -> barrier -> Senior Fundamentals -> Financial Validator (code)
  ├─ Value Trap Detector
  └─ Forensic Auditor (optional; full mode only)
    ↓
[Sync Check: deterministic routing on the analysis outcome]
    ├─ Required evidence unassessable → diagnostic output (no verdict)
    ├─ Issuer rejected → PM Fast-Fail (code), skip debate
    └─ Otherwise → continue to debate
    ↓
[Adversarial Debate]
    ├─ Bull and Bear Researchers run in parallel
    └─ One round in quick mode; a second round in full mode
    ↓
[Research Manager]
    └─ Synthesize both sides into the investment plan
    ↓
[Post-research review, in parallel with Valuation Calculator]
    └─ Optional APAC specialist and External Consultant; quick mode
       may bypass the Consultant on a clear negative or clean consensus
    ↓
[Trader → Risk Assessment Team]
    ├─ Trader proposes a plan (it places no orders)
    └─ Risky, Safe and Neutral analysts weigh it in parallel
    ↓
[Portfolio Manager + deterministic verdict policy]
    └─ FINAL DECISION: BUY / HOLD / DO NOT INITIATE
```

This pre-graph macro brief is intentionally separate from portfolio-detected
macro events. `MacroEventsStore` holds sparse discrete shocks discovered by the
portfolio path, while `src.macro_context.py` holds a short cached regional
regime brief used as News Analyst background.

The analyzer is initiation-only and portfolio-blind: it never recommends selling
a holding. Sell and trim decisions belong to the separate broker-reconciliation
path in `src/ibkr/`, which has the holdings and account context a standalone
report lacks.

**Why this works better than a single prompt:**

1. **Specialization** - Each agent focuses on one domain (no "jack of all trades")
2. **Adversarial debate** - Bull/Bear researchers catch each other's blind spots
3. **Multi-perspective risk** - Conservative/neutral/aggressive analysts represent different risk tolerances
4. **Deterministic gates** - The Financial Validator is code, not a model: sector-aware leverage and coverage thresholds (`src/validators/financial_rules.py`) produce red flags the model cannot argue away
5. **Thesis enforcement** - The Portfolio Manager applies the quantitative criteria in `src/thesis_constants.py`, and the verdict policy (`src/agents/verdict_policy.py`) constrains what it may conclude

### Real-World Example: Red-Flag Detection

Consider a stock with these financials:

- **P/E ratio:** 12 (cheap!)
- **Revenue growth:** 15% YoY (strong!)
- **Debt-to-Equity ratio:** 650% (uh oh...)
- **Interest coverage:** 1.2x (barely paying interest)

**What a single-prompt LLM might say:**

> "This stock looks promising with strong revenue growth and an attractive P/E ratio. Consider buying."

**What this multi-agent system does:**

1. **Senior Fundamentals** extracts metrics into a structured DATA_BLOCK
2. **Financial Validator** (code) detects:
   - ❌ EXTREME_LEVERAGE (D/E above the sector-aware threshold)
   - ❌ REFINANCING_RISK (weak interest coverage together with high leverage)
3. **Conditional routing** skips the debate and goes to the deterministic PM Fast-Fail
4. **Final decision:** DO NOT INITIATE, with the validator's reasons recorded in the report

This saves the debate, research and review calls for stocks that cannot qualify, and prevents "hope bias" (where debate might rationalize fatal flaws).

### The GARP Thesis: Disciplined Investing

This system isn't a black box. It enforces a specific investment strategy: **GARP (Growth at a Reasonable Price)**. The alignment with this thesis is visualized via a **6-Axis Radar Chart**:

1.  **Health** - Financial health composite (incorporating D/E and ROA).
2.  **Growth** - Growth transition score alignment.
3.  **Value** - Valuation metrics (P/E, PEG).
4.  **Regulatory** - Compliance risks (PFIC, VIE, CMIC).
5.  **Undiscovered** - Institutional/Analyst coverage gaps.
6.  **Jurisdiction** - Geopolitical and exchange stability.

**Designing for Reliability**: A common failure mode in LLM applications is "fragile parsing"—relying on regex to hunt through long narrative text for a single number. This system mitigates this by using a versioned **Structured DATA_BLOCK**. The LLM is forced to output critical metrics in a strict schema, which the Python plotting layer then extracts deterministically. This turns the LLM into a reliable data-producer for the visualization engine.

**Hard requirements** (a failure means DO NOT INITIATE; exact thresholds live in `src/thesis_constants.py`, and `tests/prompts/test_threshold_parity.py` keeps the prompts in step):

- Financial Health Score above the floor (profitable, positive cash flow, manageable debt)
- Growth Score above the floor (revenue/EPS growth, margin expansion, turnaround trajectory)
- Liquidity above a hard daily-turnover floor, with a higher level for a full pass (tradeable without slippage)
- Analyst coverage below a ceiling (undiscovered by mainstream US research)

**Soft factors** (influence risk scoring):

- Valuation (P/E, PEG and P/B against the thesis ceilings)
- US revenue exposure (capped for diversification)
- Qualitative risks (geopolitical, industry headwinds, management issues)

**Philosophy:** Find mid-cap international stocks transitioning from value (undervalued) to growth (expansion phase) before US analysts discover them. This is one of the few ways retail investors can compete with institutional funds—by looking where others aren't.

### Performance: The Honest Truth

**Speed and cost:** both depend on the resolved model seats, provider queueing and
retries, so no figure here would stay true. Quick mode runs one debate round and
skips the optional review seats; see the Runtime and cost section of
[README.md](../README.md). Every saved run records its token usage and cost by
seat, vendor, model and service tier.

**Accuracy:**
- Quantitative metrics: ~80% match to Bloomberg/Yahoo Finance (when data available)
- Qualitative synthesis: Good for generating hypotheses, **not for final decisions**
- Weaknesses: Historical data only, free APIs have gaps, sentiment analysis limited

**Reality check:** This is a **research tool**, not an oracle. Use it to generate a shortlist for deeper due diligence. Always verify with SEC filings, earnings calls, and independent sources. Never blindly trust AI—or any analyst, human or machine.

---

## Learning from This Repository

### For AI Engineers: Production Patterns

This codebase demonstrates production-grade patterns often missing from tutorials:

1. **Error handling** - Multi-source data fallback (yfinance and optional EODHD/FMP, with Tavily search), so a gap is reported rather than filled with a guess
2. **Rate limiting** - Exponential backoff, configurable RPM limits, graceful degradation
3. **Memory isolation** - Ticker-specific ChromaDB collections prevent contamination
4. **Prompt versioning** - JSON files with metadata tracking (`prompts/fundamentals_analyst.json`)
5. **Testing** - an offline-by-default suite covering unit, contract and edge cases (rate limits, missing data, malformed responses); integration tests opt in with `--run-integration`
6. **Observability** - LangSmith tracing, structured logging, token usage tracking

### For Investors: A New Tool

This system provides:

- ✅ Systematic discipline (no emotional decisions)
- ✅ Multi-perspective analysis (bull/bear/risk synthesis)
- ✅ International coverage (HK, JP, TW, KR equities)
- ✅ Auditable research (saved evidence, bindings and decision trace; model output can still vary between runs)
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
   git clone https://github.com/rgoerwit/ai-investment-agent.git
   cd ai-investment-agent
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
   # Takes minutes; elapsed time depends on provider conditions
   # Pass --output results/0005.HK.md to save the report
   ```

### Next Steps

- **Read the architecture** - Start with [README.md](../README.md), then `docs/CODEBASE_MEMORY.md`, then the live tree
- **Run tests** - `make test-ci` for the deterministic offline suite (see the Testing section of [README.md](../README.md))
- **Try batch analysis** - `./scripts/run_pipeline.sh` (see [README.md](../README.md))
- **Modify the thesis** - Thresholds belong to `src/thesis_constants.py`; the prompts under `prompts/` restate them as prose, and `tests/prompts/test_threshold_parity.py` fails when the two drift. Edit prompt JSON through a JSON parser, then run `make test-prompts`
- **Add a new agent** - Declare its seat in `src/llm_runtime/seats.py` (provider binding, tier and timeout class), build its model through `src/llm_runtime/construction.py`, then wire the node and edges in `src/graph/builder.py` with routers in `src/graph/routing.py`

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
- [docs/CODEBASE_MEMORY.md](CODEBASE_MEMORY.md) - Short orientation note for the current code layout
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

*Last updated: October 2026*
