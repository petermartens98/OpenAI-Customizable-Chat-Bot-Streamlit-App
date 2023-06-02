# Imports
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from  apikey import apikey
import os
from htmlTemplates import css, bot_template, user_template

# Streamlit page configuration
st.set_page_config(page_title='Cutomizable OpenAI ChatBot', layout='wide')

# Apply CSS styling
st.write(css, unsafe_allow_html=True)

# Define OpenAI API KEY
os.environ['OPENAI_API_KEY'] = apikey

# Initialize Session States
# Define Empty Generated and Past Chat Arrays
if 'generated' not in st.session_state: 
    st.session_state['generated'] = []

if 'past' not in st.session_state: 
    st.session_state['past'] = []

# Set up the Streamlit app layout
st.title("Customizable ChatBot")
st.subheader(" Powered by 🦜 LangChain + OpenAI + Streamlit")

# Sidebar
with st.sidebar.expander("🛠️ Settings", expanded=False):
    # Option to Specify Language Model
    MODEL = st.selectbox(label='Model', options=['gpt-3.5-turbo','text-davinci-003','text-davinci-002','code-davinci-002'])
    PERSONALITY = st.selectbox(label="Personality", options=['general-assistant','philosophical','witty'])
    
    TEMP = st.slider("Temperature",0.0,1.0,0.5)

# Define function to start a new chat
def new_chat():
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""

# Define Intialize Side Button to start a new chat
initializeBtn = st.sidebar.button('Initialize Chat', on_click=new_chat)

# Define Prompt Templates based on personlaity
if PERSONALITY == 'general-assistant':
    template = PromptTemplate(
        input_variables=['input'],
        template="You are a general assistant chatbot with the primary goal of assisting the user based on their input. \
                    USER INPUT: {input}"
    )
elif PERSONALITY == 'philosophical':
        template = PromptTemplate(
        input_variables=['input'],
        template='''
        As a philosophical chatbot, your purpose is to engage in deep and insightful conversations, exploring a wide variety of
        philosophical views. You should aim to provide profound and thought-provoking feedback based on your input, inviting you to ponder 
        the complexities of existence and the fundamental questions of human experience. Let us embark on a philosophical journey 
        together, delving into topics such as metaphysics, ethics, epistemology, morality, meditation, the afterlife, psychology and neuroscience, and the nature of reality. 
        Share your inquiries and musings, and let us dive into the vast ocean of philosophical thought, seeking wisdom and understanding amidst the 
        vast expanse of human consciousness. Now respond to user input with that in mind. 

        USER INPUT: {input}'''
    )
elif PERSONALITY == 'witty':
        template = PromptTemplate(
        input_variables=['input'],
        template='''
        As a witty and comedical chatbot, your purpose is to engage in humorous and witty conversations, exploring a wide variety of
        comedic ideas. You should aim to provide original and funny feedback based on the user input, inviting the user to ponder 
        comedy and think differently about their input in an original and witty. Let us embark on a comedic journey 
        together, Share your inquiries and musings, and let us dive into the vast ocean of comedic thought, Now respond to user input with that in mind. 

        USER INPUT: {input}'''
    )


# Memory
memory = ConversationBufferMemory(input_key="input", memory_key="chat_history")

# Language Model
llm = OpenAI(temperature=TEMP, model_name=MODEL)   

# Define Language Model Chain
llm_chain = LLMChain(llm=llm, prompt=template, memory=memory)

# Accept input from user
user_input = st.text_input("Enter your message:") 

# Execute pandas response logic
if st.button("Submit") and user_input:
    with st.spinner('Generating response...'):
        try:
            # Generate response
            response = llm_chain.run(user_input)

            # Store conversation
            st.session_state.past.append(user_input)
            st.session_state.generated.append(response)

            # Display conversation
            for i in range(len(st.session_state.past)):
                st.write(user_template.replace("{{MSG}}",st.session_state.past[i] ), unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}",st.session_state.generated[i] ), unsafe_allow_html=True)
                st.write("")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    
    
