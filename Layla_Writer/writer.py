from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv
import openai
import os
import streamlit as st
import json

load_dotenv()
serper_api_key = os.getenv("SERPER_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

llm = LLM(
    provider="deepseek",             
    model="deepseek/deepseek-reasoner",           
    api_key=deepseek_api_key
)


# UI environment set up (streamlit)

page_background = """
<style>
.stApp {
    background-image: url("https://visme.co/blog/wp-content/uploads/2017/07/50-Beautiful-and-Minimalist-Presentation-Backgrounds-042.jpg");
    background-size: cover;
    }
</style>
"""
st.markdown(page_background, unsafe_allow_html=True)

st.title('Business Consulting')

logo_url = 'https://img.evbuc.com/https%3A%2F%2Fcdn.evbuc.com%2Fimages%2F77256295%2F228474022754%2F2%2Foriginal.png?w=225&auto=format%2Ccompress&q=75&sharp=10&s=b7ae06b9aee5b9d1e14e099a53141c21'
st.sidebar.image(logo_url, caption='',use_container_width=True)
st.sidebar.write('This is an AI business consultant built with a hierarchial multi-agent architecture')

business = st.text_input('What is the business you want to consult about?')
stakeholder = st.text_input('Who is the team you are presenting to? Or who do you represent')

# 1. Team Lead – Market Research Director

planner = Agent(
    role="Business Consultant",
    goal="Plan engaging and factually accurate content about the : {topic}",
    backstory="You're working on providing Insights about : {topic} "
              "to your stakeholder who is : {stakeholder}."
              "You collect information that help them take decisions "
              "Your work is the basis for "
              "the Business Writer to deliver good insights.",
    allow_delegation=False,
 verbose=True,
    llm = llm
)


writer = Agent(
    role="Business Writer",
    goal="Write insightful and factually accurate "
         "insights about the topic: {topic}",
    backstory="You're writing a Business Insights document "
              "about the topic: {topic}. "
              "You base your design on the work of "
              "the Business Consultant, who provides an outline "
              "and relevant context about the : {topic}. "
              "and also the data analyst who will provide you with necessary analysis about the : {topic} "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provided by the Business Consultant. "
              "You also provide objective and impartial insights "
              "and back them up with information "
              "provided by the Business Consultant."
              "design your document in a professional way to be presented to : {stakeholder}."
              ,
    allow_delegation=False,
    verbose=True,
    llm=llm
)


analyst = Agent(
    role="Data Analyst",
    goal="Perform Comprehensive Statistical Analysis on the topic: {topic} ",
    backstory="You're using your strong analytical skills to provide a comprehensive statistical analysis with numbers "
              "about the topic: {topic}. "
              "You base your design on the work of "
              "the Business Consultant, who provides an outline "
              "and relevant context about the : {topic}. "
              "You follow the main objectives and "
              "direction of the outline, "
              "as provided by the Business Consultant. "
              "You also provide comprehensive statistical analysis with numbers to the Business Writer "
              "and back them up with information "
              "provided by the Business Consultant.",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

plan = Task(
    description=(
        "1. Prioritize the latest trends, key players, "
            "and noteworthy news on the {topic}.\n"
        "2. Place your business insights.\n"
        "3. Also give some suggestions and things to consider when \n "
            "dealing with International operators.\n"
        "5. Limit the document to only 500 words"
    ),
    expected_output="A comprehensive Business Consultancy document "
        "with an outline, and detailed insights, analysis and suggestions",
    agent=planner,
    # tools = [tool]

)


write = Task(
    description=(
        "1. Use the business consultant's plan to craft a compelling "
            "document about {topic}.\n"
      "2. Sections/Subtitles are properly named "
            "in an engaging manner.\n"
        "3. Proofread for grammatical errors and "
            "alignment with the brand's voice.\n"
         "3. Limit the document to only 200 words "
         "4. Use impressive images and charts to reinforce your insights "
    ),
    expected_output="A well-written Document "
        "providing insights for {stakeholder} ",
    agent=writer
)


analyse = Task(
    description=(
        "1. Use the business consultant's plan to do "
            "the needed statistical analysis with numbers on {topic}.\n"
      "2. to be presented to {stakeholder} "
            "in a document which will be deisgned by the Business Writer.\n"
        "3. You'll collaborate with your team of Business Consultant and Business writer "
            "to align on the best analysis to be provided about {topic}.\n"
 ),
    expected_output="A clear comprehensive data analysis "
        "providing insights and statistics with numbers to the Business Writer ",
    agent=analyst
)

crew = Crew(
    agents=[planner, analyst, writer],
    tasks=[plan, analyse, write],
    verbose=True 
)


if st.button("Run"):
    with st.spinner('Loading...'):
        raw_resp = crew.kickoff(inputs={"topic": business, "stakeholder": stakeholder})
        if isinstance(raw_resp, str):
            try:
                payload = json.loads(raw_resp)
            except json.JSONDecodeError:
                payload = {"raw": raw_resp}
        elif isinstance(raw_resp, dict):
            payload = raw_resp
        else:
            payload = {"raw": str(raw_resp)}

        md = payload.get("raw", "")
        st.markdown(md, unsafe_allow_html=True)