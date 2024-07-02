import os
import uuid
import hashlib

import boto3
import pandas as pd

from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_experimental.tools import PythonAstREPLTool
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone

from langgraph.graph import START, END, StateGraph
from typing import List
from typing_extensions import TypedDict

import streamlit as st

st.set_page_config(page_title="LLM Workshop", layout="wide")
st.title('LLM Workshop')

index_name = "genese-llm-workshop"
namespace = "tech-tuesday"

pc = Pinecone(api_key="8b444f46-fca7-4b41-a142-463ed4167ba7")

session = boto3.Session(profile_name='genese-llm-acc')
bedrock_client = session.client(
    'bedrock-runtime' , 
    'us-east-1', 
    endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com'
)

bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_client
)

bedrock_llm = ChatBedrock(
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    client=bedrock_client,
    model_kwargs={
        "temperature": 0.0
    }
)

PINECONE_INDEX_HOST = pc.describe_index(index_name)["host"]
pc_index = pc.Index(host=PINECONE_INDEX_HOST)

pc_vectorstore = PineconeVectorStore(
    index=pc_index, 
    embedding=bedrock_embeddings,
    text_key='text',
    namespace=namespace
)

def calculate_hash(text):
    hash = hashlib.sha256()
    hash.update(text.encode('utf-8'))
    hexdigest = hash.hexdigest()
    return hexdigest

def pinecone_upsert(texts, metadatas, ids):
    pc_vectorstore.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids
    )

def pinecone_similarity_search(question, namespace=namespace, k=2, filter=None):
    docs = pc_vectorstore.similarity_search(
        question,
        k=k,
        filter=filter,
        namespace=namespace
    )
    return docs

code_executor = create_python_agent(
    llm=bedrock_llm,
    tool=PythonAstREPLTool(),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    agent_executor_kwargs={
        "handle_parsing_errors": True, 
        "max_iterations": 5, 
        "early_stopping_method": "generate"
    }
)

def prepare_csv_file_context(docs):
    csv_file_context = ""
    
    for doc in docs:
        csv_file_path = doc.metadata["csv_file_path"]
        context = f"For csv file path {csv_file_path} it's file description is as follow:\n" + doc.page_content
        df = pd.read_csv(csv_file_path).head(2)
        df_md = df.to_markdown()
        csv_file_context += context + "It's first 2 rows in markdown format is as follow :\n" + df_md + "\n\n"
    
    csv_file_context = csv_file_context.strip()
    return csv_file_context

csv_file_path_prompt_template = """\
You are provided with csv files with their file description and 2 sample rows in markdown format for additional context.
You are also provided with user question and your task is to return csv file paths that could answer the user question.

User question is as follow:
{question}

CSV file context is as follow:
{csv_file_context}

Please give your final response in the format specified below:
{format_instructions}

NOTE: NO PREAMBLE AND NO POSTAMBLE. JUST GIVE THE DESIRED RESPONSE ONLY IN THE JSON FORMAT SPECIFIED ABOVE.

Your response: 


"""

csv_filepath_schemas = [
    ResponseSchema(
        name="csv_file_paths", 
        description="CSV file paths that could solve the user question.",
        type="array"
    ),
    ResponseSchema(
        name="explanation",
        description="Explanation on why the above csv filepaths were choosen.",
        type="string"
    )
]

csv_filepath_parser = StructuredOutputParser.from_response_schemas(csv_filepath_schemas)

csv_file_path_prompt = PromptTemplate(
    template=csv_file_path_prompt_template,
    input_variables=["question", "csv_file_context"],
    partial_variables={"format_instructions": csv_filepath_parser.get_format_instructions()},
)

csv_file_path_chain = csv_file_path_prompt | bedrock_llm | csv_filepath_parser

def get_pandas_code_from_text(response):
    if '```python' not in response:
        return response.strip()
    elif '```python' in response:
        response = response.split('```python')[-1]
        response = response.split('```')[0]
        response = response.strip()
        return response
    else:
        return None
    
def get_pandas_agent_prompt(question, resulting_fig_filepath, resulting_csv_filepath, csv_file_paths):
    sample_rows_prompt = "Dataframe first 2 rows for csv filepath is given below: "
    
    for csv_file_path in csv_file_paths:
        df = pd.read_csv(csv_file_path)
        df_md = df.head(2).to_markdown()
        _template = f"""
        
        For csv file path '{csv_file_path}' it's first 2 rows as pandas dataframe in markdown format as follow:
        {df_md}
        
        """
        sample_rows_prompt = sample_rows_prompt + _template

    sample_rows_prompt = sample_rows_prompt.strip()
    
    template = f"""\
    Given the user question, create syntatically correct python code to answer the user question.
    
    User question is as follow:
    {question}

    CSV file paths and their sample rows that could help answer the user question are as follow:
    {sample_rows_prompt}
    
    Strictly follow the following rules while generating python code to answer the user question.
    
    1. Always convert date or datetime or timestamp field if present any to datetime field for consistency using pd.to_datetime()
    2. Always do the following steps for NaN values as preprocessing step.
        2.1 Fill NaNs value using df.ffill(inplace=True) and then
        2.2 Use df.dropna(inplace=True) to drop if any NaNs present after ffill
        statements 2.1 and 2.2 can also be applied to the final dataframe obtained after certain analysis to remove any NaNs from the resulting dataframe.
    3. Always double check if necessary libraries are imported for example code might have used np but import numpy as np might missing from code.
    4. Please use latest version pandas code so that there will be no issue like depreciated warnings.
    5. Always double check if the dataframe column names as well as the variable names in python code are properly defined so that python code generated will not raise error during code execution.
    6. If question is asking for plotting graphs then only include plotting code otherwise do not include plotting code in the generated code, follow below rules for plotting
        6.1 Always use matplotlib or seaborn for plotting
        6.2 Always save the plotting to provided figure filepath which is '{resulting_fig_filepath}' in png format.
    7. At last always save the final resulting dataframe obtained after some data analysis to new csv file in filepath: '{resulting_csv_filepath}' and also limit floating point number upto 2 precisions.
    8. Always include below code snippet in the generated pandas code at very top to supress warning in code execution
        ```python
        import warnings
        warnings.filterwarnings("ignore")
    
    Do not try generate csv filepaths on your own; you are only limited to use csv filepath provided in context above as filepath and sample dataframe rows.
    
    Always give full executable python code as your Final Answer;
    
    NO POSTAMBLE i.e. do no add any text content after python code blocks.
    
    Always give a print statement at the last line of generated python code saying "Python code provided above will give answer to the user question" and \
    if the agent observation is the above print statement then code was executed successfully so you can give your Final Answer which is the python code as it is.
    
    Follow the format below for python agent:
    
    Observation: Result/Error of python code execution
    Thought: Here is the python code to answer the user question
    Action: python_repl_ast
    Action Input:
    ```python
    # python code to answer user question
    ```
        
    Final Answer: ```python

    
    """
    return template

def execute_pandas_agent_chain(question, uuid_str, resulting_csv_filepath, resulting_fig_filepath, csv_file_paths):
    pandas_agent_prompt = get_pandas_agent_prompt(
        question,
        resulting_fig_filepath,
        resulting_csv_filepath,
        csv_file_paths
    )
    response = code_executor.invoke(pandas_agent_prompt)
    python_code = get_pandas_code_from_text(response['output'])
    print("\n\nPython code is ::: \n\n", python_code)
    return python_code


answer_prompt_template = """\
You are provided with the user data question and resulting csv dataframe in markdown string for that user question.
Please give answer for that question by analysing the result provided.

User question is as follow:
{question}

Resulting csv file dataframe in markdown string is as follow:
{df_md}

Please use format specified below to give your final answer.
{format_instructions}

NO PREMABLE and NO POSTAMBLE

Your Answer: 


"""

answer_schemas = [
    ResponseSchema(
        name="answer", 
        description="Answer for the user question.",
        type="string"
    )
]

answer_parser = StructuredOutputParser.from_response_schemas(answer_schemas)

answer_prompt = PromptTemplate(
    template=answer_prompt_template,
    input_variables=["question", "df_md"],
    partial_variables={"format_instructions": answer_parser.get_format_instructions()},
)

answer_chain = answer_prompt | bedrock_llm | answer_parser

def prepare_df_md(resulting_csv_filepath):
    df = pd.read_csv(resulting_csv_filepath)
    df = df.head(10)
    df_md = df.to_markdown(index=False)
    return df_md

def prepare_answer(question, resulting_csv_filepath):
    df_md = prepare_df_md(resulting_csv_filepath)
    
    answer_chain_response = answer_chain.invoke({
        "question": question,
        "df_md": df_md
    })
    return answer_chain_response['answer']


class GraphState(TypedDict):
    """
    question : user question
    uuid_str : uuid string for making file names
    namespace : pinecone index namespace
    top_k : number of relevant docs to retrieve
    csv_file_paths: csv file paths that could solve user question
    csv_file_paths_context : csv file descriptions from vector store
    pandas_code : pandas code obtained from LLM
    pandas_code_filepath : pandas code saved file path
    resulting_csv_filepath : csv file path where resulting dataframe is saved after code execution
    resulting_fig_filepath : fig file path where resulting figure is saved after code execution
    pandas_code_executed : wether pandas code executed or not
    answer : answer of the question or fallback message
    """
    question: str
    uuid_str: str
    namespace: str
    top_k: int
    csv_file_paths_context: str
    csv_file_paths: List[str]
    pandas_code: str
    pandas_code_filepath: str
    resulting_csv_filepath: str
    resulting_fig_filepath: str
    pandas_code_executed: bool
    answer: str

def retrieve_csv_file_paths_context(state):
    print("----- RETRIEVE RELEVANT CSV FILE CONTEXT -----")

    question = state["question"]
    namespace = state["namespace"]
    top_k = state["top_k"]
    
    docs = pinecone_similarity_search(
        question,
        k=top_k,
        namespace=namespace
    )
    csv_file_paths_context = prepare_csv_file_context(docs)
    
    return {
        "csv_file_paths_context": csv_file_paths_context
    }


def extract_csv_file_paths(state):
    print("----- EXTRACT CSV PATHS -----")
    
    question = state["question"]
    csv_file_paths_context = state["csv_file_paths_context"]

    csv_file_path_chain_response = csv_file_path_chain.invoke({
        "question": question,
        "csv_file_context": csv_file_paths_context
    })
    csv_file_paths = csv_file_path_chain_response["csv_file_paths"]
    print(csv_file_paths)
    
    return {
        "csv_file_paths": csv_file_paths
    }


def generate_pandas_code(state):
    print("----- GENERATE PANDAS CODE -----")

    question = state["question"]
    uuid_str = state["uuid_str"]
    csv_file_paths = state["csv_file_paths"]

    resulting_csv_filepath = f"/tmp/{uuid_str}.csv"
    resulting_fig_filepath = f"/tmp/{uuid_str}.png"
    
    try:
        pandas_code = execute_pandas_agent_chain(
            question,
            uuid_str,
            resulting_csv_filepath,
            resulting_fig_filepath,
            csv_file_paths
        )
    except Exception as ex:
        print(ex)
        python_code = None
    
    if pandas_code:
        return {
            "pandas_code": pandas_code,
            "resulting_csv_filepath": resulting_csv_filepath,
            "resulting_fig_filepath": resulting_fig_filepath,
            "pandas_code_executed": True
        }
    else:
        return {
            "pandas_code_executed": False
        }


def check_if_pandas_code_executed_successfully(state):
    print("----- CHECK IF PANDAS CODE EXECUTED SUCCESSFULLY -----")

    pandas_code_executed = state["pandas_code_executed"]
    if pandas_code_executed:
        return "generate_answer_for_question"
    else:
        return "pandas_code_execution_fallback"


def pandas_code_execution_fallback(state):
    print("----- PANDAS CODE EXECUTION FALLBACK -----")

    return {
        "answer": "Failed to execute pandas code; may be code generate was wrong with multiple retries. Please try again later!"
    }


def generate_answer_for_question(state):
    print("----- GENERATE ANSWER FOR QUESTION -----")

    question = state["question"]
    resulting_csv_filepath = state["resulting_csv_filepath"]
    
    df_md = prepare_df_md(resulting_csv_filepath)
    
    answer_chain_response = answer_chain.invoke({
        "question": question,
        "df_md": df_md
    })
    answer = answer_chain_response['answer']

    return {
        "answer": answer
    }
    

def final_response(state):
    print("----- FINAL RESPONSE -----")
    
    answer = state["answer"]
    resulting_csv_filepath = state["resulting_csv_filepath"]
    resulting_fig_filepath = state["resulting_fig_filepath"]

    return {
        "answer": answer,
        "resulting_csv_filepath": resulting_csv_filepath,
        "resulting_fig_filepath": resulting_fig_filepath
    }

workflow = StateGraph(GraphState)

workflow.add_node("retrieve_csv_file_paths_context", retrieve_csv_file_paths_context)
workflow.add_node("extract_csv_file_paths", extract_csv_file_paths)
workflow.add_node("generate_pandas_code", generate_pandas_code)
workflow.add_node("pandas_code_execution_fallback", pandas_code_execution_fallback)
workflow.add_node("generate_answer_for_question", generate_answer_for_question)
workflow.add_node("final_response", final_response)

workflow.add_edge("retrieve_csv_file_paths_context", "extract_csv_file_paths")
workflow.add_edge("extract_csv_file_paths", "generate_pandas_code")

workflow.add_conditional_edges(
    "generate_pandas_code",
    check_if_pandas_code_executed_successfully,
    {
        "pandas_code_execution_fallback": "pandas_code_execution_fallback",
        "generate_answer_for_question": "generate_answer_for_question"
    }
)

workflow.add_edge("pandas_code_execution_fallback", "final_response")
workflow.add_edge("generate_answer_for_question", "final_response")

workflow.add_edge(START, "retrieve_csv_file_paths_context")
workflow.add_edge("final_response", END)

app = workflow.compile()

with st.form(key="question_form", clear_on_submit=True):
    question = st.text_input(
        label='Your Question', 
        placeholder='Is there a correlation between border crossing entries and the employment rate in border states?'
    )
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        output = app.invoke(
            {
                "question": question,
                "uuid_str": str(uuid.uuid4()),
                "namespace": namespace,
                "top_k": 2
            }
        )

        output_answer = output["answer"]
        output_csv_filepath = output["resulting_csv_filepath"]
        output_fig_filepath = output["resulting_fig_filepath"]

        if question:
            st.write(f'<span style="color:blue; font-weight: bold;">Question: {question}</span>', unsafe_allow_html=True)

        if output_csv_filepath:
            if os.path.exists(output_csv_filepath):
                df = pd.read_csv(output_csv_filepath)
                st.dataframe(df)
        
        if output_fig_filepath:
            if os.path.exists(output_fig_filepath):
                st.image(output_fig_filepath, channels="RGB", output_format="auto")

        if output_answer:
            st.write(f'<span style="color:green; font-weight: bold;">Answer: {output_answer}</span>', unsafe_allow_html=True)