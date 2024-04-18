from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_community.llms import Ollama
import pandas as pd
from dotenv import load_dotenv 
import json
import streamlit as st
import re
load_dotenv()

ollm = Ollama(model="llama2")

cllm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0.0, max_tokens=1024)
def csv_tool(filename: str):
    df = pd.read_csv(filename)
    return create_pandas_dataframe_agent(cllm, df, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True})

def ask_agent(agent, query):

    # Prepare the prompt with query guidelines and formatting
    prompt = (
        """
        Let's decode the way to respond to the queries. The responses depend on the type of information requested in the query. 

        1. If the query requires a table, format your answer like this:
            {"table": {"columns": ["column1", "column2", ...], "data": [[value1, value2, ...], [value1, value2, ...], ...]}}

        2. For a bar chart, respond like this:
            {"bar": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}

        3. If a line chart is more appropriate, your reply should look like this:
            {"line": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}



        4. For a plain question that doesn't need a chart or table, your response should be:
            {"answer": "Your answer goes here"}

        For example:
            {"answer": "The Product with the highest Orders is '15143Exfo'"}

        5. If the answer is not known or available, respond with:
            {"answer": "I do not know."}

        6. For statistical analysis, respond with:
        {"statistics": {"mean": value, "median": value, "mode": value, "standard_deviation": value, "correlation_coefficient": value}}

        7. For plots, respond with:
        {"plots": {"histogram": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}, "scatter_plot": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}, "line_plot": {"columns": ["A", "B", "C", ...], "data": [25, 24, 10, ...]}}}

        8. For a basic CSV overview, provide column names and data types.

        Return all output as a string. Remember to encase all strings in the "columns" list and data list in double quotes. 
        For example: {"columns": ["Products", "Orders"], "data": [["51993Masc", 191], ["49631Foun", 152]]}

        IMPORTANT: Every <Thought:> must either come with an <Action: and Action Input:> or <Final Answer:>

        Now, let's tackle the query step by step. Here's the query for you to work on: 
        """
        + query
    )

    # Run the prompt through the agent and capture the response.
    response = agent.run(prompt)

    # Return the response converted to a string.
    return str(response)



def decode_response(response: str) -> dict:
    
    return json.loads(response)

def write_raw(response_dict: dict):
    print("Response Dictionary:", response_dict)

    st.write(response_dict)

def write_answer(response_dict: dict):
    print("Response Dictionary:", response_dict)  # Add this line to see the response dictionary
    
    # Check if the response is an answer.
    if "answer" in response_dict:
        st.write(response_dict["answer"])

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        try:
            df_data = {col: [x[i] for x in data['data']] for i, col in enumerate(data['columns'])}
            df = pd.DataFrame(df_data)
            st.line_chart(df)
        except ValueError:
            print(f"Couldn't create DataFrame from data: {data}")

    # Check if the response includes statistical analysis.
    if "statistics" in response_dict:
        statistics = response_dict["statistics"]
        st.subheader("Statistical Analysis")

        if "mean" in statistics:
            st.write(f"Mean: {statistics['mean']}")

        if "median" in statistics:
            st.write(f"Median: {statistics['median']}")

        if "mode" in statistics:
            st.write(f"Mode: {statistics['mode']}")

        if "standard_deviation" in statistics:
            st.write(f"Standard Deviation: {statistics['standard_deviation']}")

        if "correlation_coefficient" in statistics:
            st.write("Correlation Coefficient:")
            st.write(statistics['correlation_coefficient'])

    # Check if the response includes plots.
    if "plots" in response_dict:
        plots = response_dict["plots"]
        if "histogram" in plots:
            st.subheader("Histogram")
            data = plots["histogram"]
            try:
                # Ensure data['data'] is a list to avoid the TypeError
                if isinstance(data['data'], list):
                    df_data = {col: data['data'] for col in data['columns']}
                    df = pd.DataFrame(df_data)
                    st.bar_chart(df)
                else:
                    print("Data format error: Data should be a list.")
            except ValueError:
                print(f"Couldn't create DataFrame from data: {data}")

        if "scatter_plot" in plots:
            st.subheader("Scatter Plot")
            data = plots["scatter_plot"]
            try:
                # Ensure data['data'] is a list to avoid the TypeError
                if isinstance(data['data'], list):
                    df_data = {col: data['data'][i] for i, col in enumerate(data['columns'])}
                    df = pd.DataFrame(df_data)
                    st.write(df)
                    st.line_chart(df)
                else:
                    print("Data format error: Data should be a list.")
            except ValueError:
                print(f"Couldn't create DataFrame from data: {data}")

        if "line_plot" in plots:
            st.subheader("Line Plot")
            data = plots["line_plot"]
            try:
                # Ensure data['data'] is a list to avoid the TypeError
                if isinstance(data['data'], list):
                    df_data = {col: data['data'][i] for i, col in enumerate(data['columns'])}
                    df = pd.DataFrame(df_data)
                    st.line_chart(df)
                else:
                    print("Data format error: Data should be a list.")
            except ValueError:
                print(f"Couldn't create DataFrame from data: {data}")

    # Check if there was a parsing error.
    if "parsing_error" in response_dict:
        st.error(f"Parsing error: {response_dict['parsing_error']}")




#UI
st.set_page_config(page_title="Analyse Your CSV")
st.title("CSV chat")

st.write("Please upload your CSV file below.")

data = st.file_uploader("Upload a CSV" , type="csv")

query = st.text_area("Send a Message")

if st.button("Submit Query", type="primary"):
    # Create an agent from the CSV file.
    agent = csv_tool(data)

    # Query the agent.
    response = ask_agent(agent=agent, query=query)

    # Decode the response.
    decoded_response = decode_response(response)

    # Write the response to the Streamlit app.
    # Raw Object
    write_raw(decoded_response) 
    # Formatted Output
    write_answer(decoded_response)