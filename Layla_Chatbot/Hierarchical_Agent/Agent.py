import os
import gc
import time
import streamlit as st

from dotenv import load_dotenv
from typing import Type, List, Any
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import SerperDevTool

load_dotenv(dotenv_path="../../.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = LLM(model='gpt-4o', api_key=openai_api_key)
serper_tool = SerperDevTool(api_key=serper_api_key)

def execute_task() -> Crew:
    """
    Build and return a Crew object with the desired agents and tasks.
    """

    manager_agent = Agent(
        role='Project Research Manager',
        goal='Oversee the project research',
        backstory="""You are an experienced project manager responsible
                     for ensuring project research is carried out successfully
                     and with good quality.""",
        allow_delegation=True,
        verbose=True,
        llm=llm,
    )

    market_demand_agent = Agent(
        role="Market Demand Analyst",
        goal="Analyze market demand for new projects.",
        backstory="""A skilled market analyst with expertise in
                     evaluating product-market fit.""",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool]
    )

    risk_analysis_agent = Agent(
        role='Risk Analysis Analyst',
        goal='Assess potential risks associated with the project.',
        backstory="""A financial and strategic expert
                     focused on identifying business risks.""",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool]
    )

    return_on_investment_agent = Agent(
        role='Return on Investment Analyst',
        goal='Estimate the financial return on investment.',
        backstory="""You are an expert in financial modeling and
                     investment analysis.""",
        allow_delegation=False,
        verbose=True,
        llm=llm,
        tools=[serper_tool]
    )

    manager_task = Task(
        description="""Oversee the project research on {project_title}
                       and ensure timely, high-quality responses""",
        expected_output="""A manager-approved response ready to be sent
                           as an article on {project_title}""",
        agent=manager_agent,
    )

    market_demand_task = Task(
        description="Analyze the demand for the project '{project_title}'",
        expected_output="A detailed bulletlist of market demand trends with web-source references cited each point",
        agent=market_demand_agent,
    )

    risk_analysis_task = Task(
        description="Analyze the risk of the project '{project_title}'",
        expected_output="A categorized risk assessment report, a bullet list following the structure of a SWOT analysis, with each point having relevant web-sources cited ",
        agent=risk_analysis_agent,
    )

    return_on_investment_task = Task(
        description="Analyze the return on investment of the project '{project_title}'",
        expected_output="A structured ROI estimate for the project, with relevant numerical data to support each argument and relevant web-sources cited",
        agent=return_on_investment_agent
    )

    final_report_task = Task(
        description="""Review the final response from the market demand,
                       risk analysis, and ROI agents and create a final report.""",
        expected_output="""A comprehensive report on the project '{project_title}'
                           containing market demand, risk analysis,
                           and return on investment.""",
        agent=manager_agent
    )

    project_research_crew = Crew(
        agents=[market_demand_agent, risk_analysis_agent, return_on_investment_agent],
        tasks=[
            market_demand_task,
            risk_analysis_task,
            return_on_investment_task,
            final_report_task
        ],
        manager_agent=manager_agent,
        process=Process.hierarchical,
        verbose=True,
    )

    return project_research_crew

def reset_chat():
    st.session_state.messages = []
    gc.collect()

def main():
    st.title("Goldman Stanley's Consulting Team")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crew" not in st.session_state:
        st.session_state.crew = None

    prompt = st.chat_input("What would you like to research today?")
    if prompt:

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if st.session_state.crew is None:
            st.session_state.crew = execute_task()

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Thinking..."):
                #st.success("摩根大通不辱使命，用更好的结果欢迎曹总的到来！")
                loading_placeholder = st.empty()
                loading_placeholder2 = st.empty()
                loading_placeholder3 = st.empty()
                loading_placeholder.image("../../Data_file/cattyping2.gif")
                loading_placeholder2.image("../../Data_file/cattyping.gif")
                loading_placeholder3.image("../../Data_file/frogtyping.gif")
                inputs = {"project_title": prompt}
                result = st.session_state.crew.kickoff(inputs=inputs).raw
                # crew_conversation = result.debug
                # with st.expander("Agent work in progress..."):
                #     st.write(crew_conversation)
                loading_placeholder.empty()
                loading_placeholder2.empty()
                loading_placeholder3.empty()
            lines = result.split('\n')
            for i, line in enumerate(lines):
                full_response += line
                if i < len(lines) - 1:
                    full_response += '\n'
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.1)
            message_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": result})

    if st.button("Reset Chat"):
        reset_chat()
        st.experimental_rerun()

if __name__ == "__main__":
    main()
