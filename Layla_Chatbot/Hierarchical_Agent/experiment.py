import os
import gc
import time
import streamlit as st
from dotenv import load_dotenv

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool

import markdown
from weasyprint import HTML


load_dotenv(dotenv_path="../../.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = LLM(model='gpt-4o', api_key=openai_api_key)
serper_tool = SerperDevTool(api_key=serper_api_key)

def create_pdf_from_markdown(report_text: str) -> bytes:
    """
    Convert the markdown report to HTML and then generate a PDF
    preserving formatting and styling using WeasyPrint.
    """

    html_content = markdown.markdown(report_text, output_format="html5")
    html_full = f"""
    <html>
      <head>
        <meta charset="utf-8">
        <style>
          body {{
            font-family: Arial, sans-serif;
            margin: 1cm;
            font-size: 12pt;
            line-height: 1.4;
          }}
          h1, h2, h3, h4, h5, h6 {{
            color: #333333;
          }}
          p {{
            margin-bottom: 1em;
          }}
          ul, ol {{
            margin-left: 20px;
          }}
          a {{
            color: #1155cc;
            text-decoration: none;
          }}
        </style>
      </head>
      <body>
      {html_content}
      </body>
    </html>
    """
    # Generate PDF bytes from HTML
    pdf_bytes = HTML(string=html_full).write_pdf()
    return pdf_bytes

def create_market_research_team() -> Crew:
    """
    Build and return a Crew object for enterprise-grade market research
    using a multi-agent team structure.
    """

    # Manager (Team Lead) – provided separately as manager_agent
    team_lead = Agent(
        role="Team Lead - Market Research Director",
        goal=(
            "Oversee the entire research project and synthesize all analyst reports into a cohesive, "
            "enterprise-grade final report. Ensure that every data point is supported by a direct citation."
        ),
        backstory=(
            "A seasoned project director with extensive experience in strategic consulting and market research. "
            "Guides the team to deliver a high-level final report meeting top-tier standards."
        ),
        allow_delegation=True,
        verbose=True,
        llm=llm,
    )

    # Specialized Agents (manager agent is not included in the agents list)
    macro_market_analyst = Agent(
        role="Macro Market Analyst",
        goal=(
            "Analyze the overall market for '{project_title}', including market size, growth trends, segmentation, "
            "and key drivers. Provide detailed insights with direct citations."
        ),
        backstory="An expert in macroeconomic trends and market dynamics.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool],
    )

    consumer_insights_analyst = Agent(
        role="Consumer Insights Analyst",
        goal=(
            "Research and analyze consumer behavior, demographics, and preferences for '{project_title}'. "
            "Identify key segments and emerging trends with supporting data and direct citations."
        ),
        backstory="A specialist in consumer behavior and market segmentation.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool],
    )

    competitor_intelligence_analyst = Agent(
        role="Competitive Intelligence Analyst",
        goal=(
            "Examine the competitive landscape for '{project_title}', evaluating competitor strategies, market share, "
            "pricing, and positioning. Substantiate each insight with direct source citations."
        ),
        backstory="An expert in competitive intelligence and market benchmarking.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool],
    )

    financial_risk_analyst = Agent(
        role="Financial & Risk Analyst",
        goal=(
            "Provide a comprehensive financial analysis for '{project_title}', including ROI projections, cost estimations, "
            "and risk assessments. Every financial data point must include a direct citation."
        ),
        backstory="An expert in financial modeling and risk management.",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool],
    )

    # Define Tasks for each agent
    macro_market_task = Task(
        description=(
            "Conduct a detailed analysis of the overall market for '{project_title}', focusing on market size, growth trends, "
            "segmentation, and key drivers. Include direct citations for every claim."
        ),
        expected_output=(
            "A bullet-point summary and narrative of the macro market landscape for '{project_title}', with each point supported by a credible source."
        ),
        agent=macro_market_analyst,
    )

    consumer_insights_task = Task(
        description=(
            "Analyze the consumer landscape for '{project_title}'. Identify key consumer segments, behavior patterns, and preferences. "
            "Provide direct citations for each insight."
        ),
        expected_output=(
            "A comprehensive breakdown of consumer insights for '{project_title}', with data points and direct source links."
        ),
        agent=consumer_insights_analyst,
    )

    competitive_intelligence_task = Task(
        description=(
            "Evaluate the competitive environment for '{project_title}', including competitor strategies, market share, pricing, and positioning. "
            "Ensure every insight is accompanied by a direct citation."
        ),
        expected_output=(
            "A detailed competitive analysis for '{project_title}' presented in bullet points with supporting citations."
        ),
        agent=competitor_intelligence_analyst,
    )

    financial_risk_task = Task(
        description=(
            "Develop a financial analysis for '{project_title}', including ROI projections, cost estimations, and risk assessments. "
            "Substantiate each financial data point with direct source citations."
        ),
        expected_output=(
            "A financial and risk assessment report for '{project_title}' detailing key metrics and risk factors with credible source links."
        ),
        agent=financial_risk_analyst,
    )

    synthesis_task = Task(
        description=(
            "Synthesize the findings from the Macro Market, Consumer Insights, Competitive Intelligence, and Financial & Risk tasks into "
            "one cohesive, enterprise-grade market research report for '{project_title}'. Ensure a consistent tone and that every data point is cited."
        ),
        expected_output=(
            "A final, polished market research report for '{project_title}' that integrates all analyses with thorough citations."
        ),
        agent=team_lead,
    )

    # Create the Crew (manager agent is provided separately)
    market_research_crew = Crew(
        agents=[
            macro_market_analyst,
            consumer_insights_analyst,
            competitor_intelligence_analyst,
            financial_risk_analyst
        ],
        tasks=[
            macro_market_task,
            consumer_insights_task,
            competitive_intelligence_task,
            financial_risk_task,
            synthesis_task
        ],
        manager_agent=team_lead,
        process=Process.hierarchical,
        verbose=True,
    )

    return market_research_crew

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def main():
    st.title("Enterprise Market Research Team")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crew" not in st.session_state:
        st.session_state.crew = None

    user_prompt = st.chat_input(
        "Enter the market, product, or project to analyze with our enterprise-grade research team:"
    )

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        if st.session_state.crew is None:
            st.session_state.crew = create_market_research_team()

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Our research team is gathering data..."):
                # Display loading images
                loading_placeholder = st.empty()
                loading_placeholder2 = st.empty()
                loading_placeholder3 = st.empty()
                loading_placeholder.image("../../Data_file/cattyping2.gif")
                loading_placeholder2.image("../../Data_file/cattyping.gif")
                loading_placeholder3.image("../../Data_file/frogtyping.gif")

                inputs = {"project_title": user_prompt}
                result = st.session_state.crew.kickoff(inputs=inputs).raw

                st.success("Here is your comprehensive market research report:")
                loading_placeholder.empty()
                loading_placeholder2.empty()
                loading_placeholder3.empty()

            # Stream the text to the UI
            for i, line in enumerate(result.split('\n')):
                full_response += line + ("\n" if i < len(result.split('\n')) - 1 else "")
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.1)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": result})

        # Generate PDF from the final report text (using markdown conversion to preserve formatting)
        pdf_bytes = create_pdf_from_markdown(result)
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name="market_research_report.pdf",
            mime="application/pdf"
        )

    if st.button("Reset Chat"):
        reset_chat()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
