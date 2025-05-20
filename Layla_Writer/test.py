from crewai import Agent, Task, Crew, LLM
from crewai.tools import WebSearchTool, ScrapeTool
from dotenv import load_dotenv
import streamlit as st
import os, json

# ——— Load environment variables ———
load_dotenv()
serper_key   = os.getenv("SERPER_API_KEY")
deepseek_key = os.getenv("DEEPSEEK_API_KEY")

# ——— Initialize LLM ———
llm = LLM(
    provider="deepseek",
    model="deepseek/deepseek-reasoner",
    api_key=deepseek_key
)

# ——— Instantiate external tools ———
serper  = WebSearchTool(api_key=serper_key)   # Serper.ai wrapper for web search
scraper = ScrapeTool()                       # Firecrawl or custom scraper wrapper

# ——— Define Agents ———
# 1. Project Manager (orchestrator)
manager = Agent(
    role="Project Manager",
    goal="Decompose a consulting request into actionable subtasks and coordinate the workflow.",
    backstory=(
        "You are the orchestrator: given a topic and stakeholder, you produce a JSON array of subtasks "
        "and dispatch them to specialist agents."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# 2. Specialist Agents with real external research capabilities
market_agent = Agent(
    role="Market Research Specialist",
    goal=(
        "Gather current market-size figures, segment CAGRs, and industry trends for {topic}. "
        "When you need up-to-date data, call `web_search(query)` and cite the URLs."
    ),
    backstory=(
        "Use the `web_search(...)` tool to find credible market reports. "
        "If you locate a PDF or HTML table, pass its URL to `scrape(url)` to extract detailed data."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper, scraper]
)

competitor_agent = Agent(
    role="Competitive Intelligence Specialist",
    goal=(
        "Identify and profile the top 5 competitors in {topic}, covering their offerings, pricing, "
        "market share, and strategic positioning."
    ),
    backstory=(
        "Use `web_search(...)` to locate competitor websites and filings. "
        "When you find a data table, extract it via `scrape(url)` for accuracy."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper, scraper]
)

consumer_agent = Agent(
    role="Consumer Insights Specialist",
    goal=(
        "Compile consumer sentiment, pain-points, and emerging preferences for {topic}. "
        "Use `web_search('site:reddit.com {topic} reviews')` or similar, and scrape threads with `scrape(url)`."
    ),
    backstory=(
        "Aggregate consumer feedback from forums, review sites, and social channels. "
        "Quantify and cite your findings."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper, scraper]
)

regulatory_agent = Agent(
    role="Regulatory & Risk Specialist",
    goal=(
        "Investigate compliance hurdles, regulations, and cross-border considerations for {topic}. "
        "Use `web_search('regulatory {topic} site:.gov')` to find statutes or rulings."
    ),
    backstory=(
        "Summarize key legal risks and approvals needed, citing government or legal sources."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper]
)

# 3. Synthesis Writer (final deliverable)
writer_agent = Agent(
    role="Synthesis Writer",
    goal=(
        "Combine all specialist outputs into a cohesive executive summary and slide deck outline for {stakeholder}. "
        "Use clear headings, key metrics, and actionable recommendations."
    ),
    backstory=(
        "Craft a polished, stakeholder-ready document by weaving together market, competitor, "
        "consumer, and regulatory insights."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# ——— Streamlit UI ———
st.set_page_config(page_title="AI Business Consultant", layout="wide")
page_bg = """
<style>
.stApp { background: url('https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-042.jpg') no-repeat center center fixed; background-size: cover; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.title("🧠 AI-Powered Business Consultant")
with st.sidebar:
    st.image("https://img.evbuc.com/https%3A%2F%2Fcdn.evbuc.com%2Fimages%2F77256295%2F228474022754%2F2%2Foriginal.png?w=225", use_container_width=True)
    st.write("A hierarchical multi-agent architecture with live web search and scraping")

business    = st.text_input("Business topic (e.g., orthodontics)")
stakeholder = st.text_input("Stakeholder / audience")

if st.button("Run"):
    # STEP 1: Manager creates subtasks
    plan_task = Task(
        description=(
            f"As the Project Manager, break down the consulting request on '{business}' "
            f"for stakeholder '{stakeholder}' into 4 subtasks. Return a JSON array of {{name, desc}}."
        ),
        expected_output="JSON list of { name: string, desc: string }",
        agent=manager
    )
    plan_crew = Crew(agents=[manager], tasks=[plan_task], verbose=False)
    plan_resp = plan_crew.kickoff(inputs={})
    plan_raw  = plan_resp.raw
    try:
        subtasks = json.loads(plan_raw)
    except json.JSONDecodeError:
        st.error("Failed to parse manager's JSON plan:")
        st.code(plan_raw, language="json")
        st.stop()

    # STEP 2: Build specialist tasks
    tasks = []
    for stask in subtasks:
        name, desc = stask['name'], stask['desc']
        lname = name.lower()
        if 'market' in lname:
            agent = market_agent
        elif 'competitor' in lname:
            agent = competitor_agent
        elif 'consumer' in lname:
            agent = consumer_agent
        elif 'regulator' in lname or 'risk' in lname:
            agent = regulatory_agent
        else:
            agent = market_agent
        tasks.append(Task(description=desc, expected_output=f"Report on '{name}'", agent=agent))

    # STEP 3: Final synthesis task
    synth_desc = "You have these reports:\n"
    for stask in subtasks:
        synth_desc += f"- **{stask['name']}**: earlier report.\n"
    synth_desc += (
        f"\nNow synthesize everything into a 500-word executive summary and slide deck outline "
        f"for stakeholder '{stakeholder}'."
    )
    tasks.append(Task(description=synth_desc,
                      expected_output="Final summary & outline in Markdown",
                      agent=writer_agent))

    # STEP 4: Execute all tasks
    all_agents = [market_agent, competitor_agent, consumer_agent, regulatory_agent, writer_agent]
    final_crew = Crew(agents=all_agents, tasks=tasks, verbose=True)
    final_resp = final_crew.kickoff(inputs={})

    # STEP 5: Render the synthesized Markdown
    st.markdown(final_resp.raw, unsafe_allow_html=True)

