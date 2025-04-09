import streamlit as st
from chatbot_initialization import Chatbot
from evaluator import Evaluator

def main():
    st.title("Layla Research Assistant")

    # Initialize session state for chat history, chatbot, and evaluator if not already present.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "chatbot" not in st.session_state:
        st.session_state["chatbot"] = Chatbot()
    if "evaluator" not in st.session_state:
        st.session_state["evaluator"] = Evaluator()

    # Use st.chat_input to capture user messages (this comes with its own submission behavior).
    user_query = st.chat_input("Ask Layla about anything in this research paper!")
    
    # Process the user's query as soon as it's provided.
    if user_query:
        # Log and display the user message.
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        
        # Generate the assistant's response.
        with st.spinner("Generating response..."):
            response, context = st.session_state.chatbot.generate_response(user_query)
        
        # Log and display the assistant's response.
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
        
        # Optionally display additional details in an expander.
        with st.expander("View Details"):
            st.subheader("Context Used for the Response")
            st.text_area("Context", context, height=200)
            
            with st.spinner("Evaluating response..."):
                try:
                    evaluation_df = st.session_state.evaluator.evaluate_responses(user_query, response, context)
                    
                    st.subheader("Evaluation Results DataFrame")
                    st.dataframe(evaluation_df)
                    
                    # Define your metric columns.
                    metric_columns = [
                        "answer_correctness", 
                        "answer_relevancy", 
                        "context_recall", 
                        "context_precision", 
                        "faithfulness"
                    ]
                    
                    if not evaluation_df.empty:
                        # Extract the metrics from the first row.
                        metrics_series = evaluation_df[metric_columns].iloc[0]
                        
                        # Build a Markdown string with bullet points for each metric.
                        markdown_str = "### Evaluation Metrics\n\n"
                        for metric, value in metrics_series.items():
                            # Multiply by 100 and round for percentage display.
                            markdown_str += f"- **{metric.replace('_', ' ').title()}**: {round(value, 4)*100:.2f}%\n"
                        st.markdown(markdown_str)
                        
                        # Create and display a bar chart using the metrics.
                        metrics_df = metrics_series.to_frame(name="Score")
                        st.subheader("Evaluation Metrics Bar Chart")
                        st.bar_chart(metrics_df)
                    else:
                        st.warning("No evaluation metrics to display.")
                except Exception as e:
                    st.error(f"An error occurred during evaluation: {e}")

if __name__ == "__main__":
    main()
