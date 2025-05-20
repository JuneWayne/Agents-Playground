import os
import json
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool

# —— Load environment variables ——
load_dotenv()
deepseek_key = os.getenv("DEEPSEEK_API_KEY")
serper_api_key  = os.getenv("SERPER_API_KEY")

# —— Initialize language model & tool ——
llm = LLM(
    provider="deepseek",
    model="deepseek/deepseek-reasoner",
    api_key=deepseek_key,
    chat_prefix_completion=True,
    prefix_mode="prefix"  
)
serper_tool = SerperDevTool(api_key=serper_api_key)

# —— Define Agents ——
manager = Agent(
    role="Project Manager",
    goal=(
        "Decompose a consulting request into actionable research tasks and delegate them to specialists."
    ),
    backstory=(
        "As manager, you receive a topic and stakeholder, create subtasks for market research, competitor analysis, "
        "consumer insights, regulatory review, and synthesis, then assign them to the right agent."
    ),
    allow_delegation=True,
    verbose=True,
    llm=llm
)

market_agent = Agent(
    role="Market Research Specialist",
    goal=(
        "Gather current market size, segment CAGRs, and top industry trends for {topic}. "
        "When you need fresh data, call `serper_tool.search(query)` and cite the source URLs."
    ),
    backstory=(
        "Use `serper_tool.search(...)` to fetch reports, statistics, and credible insights."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper_tool]
)

competitor_agent = Agent(
    role="Competitive Intelligence Specialist",
    goal=(
        "Identify and profile the top 5 competitors in {topic}, including offerings, pricing, "
        "market share, and strategic differentiators."
    ),
    backstory=(
        "Use `serper_tool.search(...)` to locate competitor websites, filings, or news articles, "
        "and summarize their key details."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper_tool]
)

consumer_agent = Agent(
    role="Consumer Insights Specialist",
    goal=(
        "Compile consumer sentiment, pain-points, and preferences for {topic}. "
        "Search reviews and forums via `serper_tool.search(query)` and include direct quotes."
    ),
    backstory=(
        "Aggregate and quantify consumer feedback, making sure to cite each source URL."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper_tool]
)

regulatory_agent = Agent(
    role="Regulatory & Risk Specialist",
    goal=(
        "Investigate compliance requirements and legal risks for {topic}. "
        "Call `serper_tool.search('site:.gov {topic} regulation')` to find statutes and rulings."
    ),
    backstory=(
        "Summarize any regulatory hurdles, approvals, or compliance steps needed."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm,
    tools=[serper_tool]
)

writer_agent = Agent(
    role="Synthesis Writer",
    goal=(
        "Integrate all specialist outputs into a coherent executive summary and slide deck outline for {stakeholder}."
    ),
    backstory=(
        "Weave together market, competitor, consumer, and regulatory insights into a polished deliverable."
    ),
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# —— Streamlit App UI ——
st.set_page_config(page_title="Delegated AI Business Consultant", layout="wide")
page_bg = """
<style>
.stApp { background:url('https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-042.jpg') no-repeat center center fixed; background-size:cover; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.title("🧠 Delegated AI Business Consultant")
with st.sidebar:
    st.image(
        "https://www.marefa.org/w/images/thumb/d/d5/University_of_Virginia_School_of_Data_Science_logo.svg/1200px-University_of_Virginia_School_of_Data_Science_logo.svg.png",
        use_container_width=True
    )
    st.write("One-step delegation with SerperDevTool for live research")

business    = st.text_input("Business topic (e.g., orthodontics)")
stakeholder = st.text_input("Stakeholder / audience")

if st.button("Run"):
    delegate_task = Task(
        description=(
            f"As Project Manager, for topic '{business}' and stakeholder '{stakeholder}', "
            "delegate research tasks for Market Research, Competitor Analysis, Consumer Insights, "
            "Regulatory Review, and Synthesis to the appropriate agents."    
        ),
        expected_output="Automated delegation and execution of specialist and synthesis tasks.",
        agent=manager
    )

    crew = Crew(
        agents=[manager, market_agent, competitor_agent, consumer_agent, regulatory_agent, writer_agent],
        tasks=[delegate_task],
        verbose=True
    )

    result = crew.kickoff(inputs={"topic": business, "stakeholder": stakeholder})
    st.markdown(result.raw, unsafe_allow_html=True)