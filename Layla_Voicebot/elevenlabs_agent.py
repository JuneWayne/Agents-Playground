import os
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ElevenLabs Chatbot", layout="wide")
st.title("🗣️ ElevenLabs Conversational AI Chatbot")

# 1) Let the user supply their AGENT_ID (or pull from env)
agent_id = st.text_input(
    "ElevenLabs Agent ID",
    value=os.getenv("AGENT_ID", ""),
    help="Make sure this agent is public (auth disabled) or use a signed URL."
)

# 2) (Optional) You could also allow dynamic-variables, overrides, etc.:
# dynamic_vars = st.text_area("Dynamic Variables (JSON)", value="{}")

if agent_id:
    # 3) Render the widget HTML + script via Streamlit Components
    widget_html = f"""
    <!-- ElevenLabs Conversational AI widget -->
    <elevenlabs-convai
      agent-id="{agent_id}"
      variant="expanded"
      action-text="Chat with Assistant"
    ></elevenlabs-convai>
    <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
    """
    # You can adjust height to suit your layout
    components.html(widget_html, height=600, scrolling=True)
else:
    st.info("Please enter your ElevenLabs Agent ID to load the chatbot.")
