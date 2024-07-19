import time
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st
import urllib.parse

def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    try:
        db_uri = f"mysql+mysqlconnector://{user}:{urllib.parse.quote(password)}@{host}:{port}/{database}"
        return SQLDatabase.from_uri(db_uri)
    except Exception as e:
        st.error(f"Error initializing database: {e}")
        return None

def get_sql_chain(db):
    template = """
        Database Name: employee_db
        Tables:
        employee_data:
        EmpID
        First Name
        Last Name
        Start Date
        Exit Date
        Title
        Supervisor
        Email
        Business Unit
        Employee Status
        Employee Type
        Pay Zone
        Employee Classification Type
        Termination Type
        Termination Description
        Department Type
        Division Description
        DOB (Date of Birth)
        State
        Job Function
        Gender
        Location
        Race (or) Ethnicity
        Marital Status
        Performance Score
        Current Employee Rating

        employee_engagement_survey_data:
        Employee ID
        Survey Date
        Engagement Score
        Satisfaction Score
        Work-Life Balance Score

        recruitment_data:
        Applicant ID
        Application Date
        First Name
        Last Name
        Gender
        Date of Birth
        Phone Number
        Email
        Address
        City
        State
        Zip Code
        Country
        Education Level
        Years of Experience
        Desired Salary
        Job Title
        Status

        training_and_development_data:
        Employee ID
        Training Date
        Training Program Name
        Training Type
        Training Outcome
        Location
        Trainer
        Training Duration (Days)
        Training Cost

        User's Question: {question}

        Instructions:
        Write a SQL query based on the user's question.
        Use the tables and fields mentioned above.
        Ensure that column names with spaces are enclosed in backticks (`).
        Provide only the SQL query and nothing else.
        Do not include additional text or formatting.

        Examples:
        Question: What is the average engagement score?
        SQL Query: SELECT AVG(`Engagement Score`) AS avg_engagement_score FROM employee_engagement_survey_data;

        Question: How has the average engagement score changed over time?
        SQL Query: SELECT `Survey Date`, AVG(`Engagement Score`) AS avg_engagement_score FROM employee_engagement_survey_data GROUP BY `Survey Date` ORDER BY `Survey Date`;

        Question: Show the engagement scores for all employees surveyed in the last quarter.
        SQL Query: SELECT `Employee ID`, `Engagement Score` FROM employee_engagement_survey_data WHERE `Survey Date` >= DATE('now', '-3 month');

        Question: What is the range of work-life balance scores?
        SQL Query: SELECT MIN(`Work-Life Balance Score`) AS min_score, MAX(`Work-Life Balance Score`) AS max_score FROM employee_engagement_survey_data;

        Your Turn:
        Question: {question}
        SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>
    
        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    raw_response = chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

    # Clean up the response to remove explanatory text
    cleaned_response = raw_response.split(".")[0] + "."

    return cleaned_response

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

# Load environment variables if necessary
load_dotenv()

# Set page title and icon
st.set_page_config(page_title="Garnishment AI Engine", page_icon=":speech_balloon:")

# Define function to toggle theme
def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
    st.experimental_set_query_params(theme=st.session_state.theme)
    st.experimental_rerun()

# Initialize theme
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Apply theme
dark_theme = st.session_state.theme == "dark"
theme_styles = """
    <style>
    body {{
        color: {text_color};
        background-color: {bg_color};
    }}
    .stButton>button {{
        background-color: {button_bg};
        color: {button_text};
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s, color 0.3s;
    }}
    .stButton>button:hover {{
        background-color: {button_hover_bg};
        color: {button_hover_text};
    }}
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background-color: {input_bg};
        color: {input_text};
        border-radius: 10px;
        border: 1px solid {border_color};
        padding: 10px;
        font-size: 16px;
        width: 100%;
    }}
    .chat-message {{
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
        max-width: 80%;
        display: inline-block;
    }}
    .ai-message {{
        background-color: {ai_bg};
        color: {ai_text};
        text-align: right;
        float: right;
    }}
    .human-message {{
        background-color: {human_bg};
        color: {human_text};
        text-align: left;
        float: left;
    }}
    </style>
""".format(
    text_color="white" if dark_theme else "black",
    bg_color="#1E1E1E" if dark_theme else "white",
    button_bg="#333" if dark_theme else "#e0e0e0",
    button_text="white" if dark_theme else "black",
    button_hover_bg="#444" if dark_theme else "#c0c0c0",
    button_hover_text="white" if dark_theme else "black",
    input_bg="#333" if dark_theme else "#f0f0f0",
    input_text="white" if dark_theme else "black",
    border_color="#444" if dark_theme else "#ddd",
    ai_bg="#282828" if dark_theme else "#e0e0e0",
    ai_text="white" if dark_theme else "black",
    human_bg="#1E1E1E" if dark_theme else "#f0f0f0",
    human_text="white" if dark_theme else "black",
)
st.markdown(theme_styles, unsafe_allow_html=True)

# Page title
st.title("Garnishment AI Engine")


# Sidebar for database connection settings
with st.sidebar:
    st.subheader("Database Connection")
    host = st.text_input("Host", value="localhost")
    port = st.text_input("Port", value="3306")
    user = st.text_input("User", value="root")
    password = st.text_input("Password", type="password", value="")
    database = st.text_input("Database", value="employee_db")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(user, password, host, port, database)
            if db:
                st.session_state.db = db
                st.session_state.chat_history = []
                st.success("Connected to database!")
            else:
                st.error("Failed to connect to database. Check your credentials.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    message_type = "ai-message" if isinstance(message, AIMessage) else "human-message"
    st.markdown(f"""
        <div class="chat-message {message_type}">
            {message.content}
        </div>
    """, unsafe_allow_html=True)

# User input for chat
user_query = st.text_area("Your question:", key="user_query")

# Send button for chat
if st.button("Send"):
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.spinner("Thinking..."):
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
        st.session_state.chat_history.append(AIMessage(content=response))
    st.experimental_rerun()

# Theme toggle button
st.sidebar.button("Toggle Theme", on_click=toggle_theme)
