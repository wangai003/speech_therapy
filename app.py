import os
import streamlit as st
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
import openai
import langchain
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.schema import HumanMessage
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# Set page configuration
st.set_page_config(page_title="Speech Therapy", page_icon="🗣️", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page:", ["Chatbot", "Community", "Resources", "Activities"])

# Load PDF for chatbot functionality
pdf_file_path = "pdf.pdf"

@st.cache_resource(ttl="1h")
def configure_qa_chain(pdf_file_path):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, os.path.basename(pdf_file_path)) 
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs) 
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 2, "fetch_k": 4})
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    llm = ChatOpenAI(
        model_name="gpt-4o-mini", temperature=0, openai_api_key=openai.api_key, streaming=True
    )
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm, retriever=retriever, memory=memory, verbose=True,
    )  
    return qa_chain

qa_chain = configure_qa_chain(pdf_file_path)


# Define the system prompt for the speech therapy chatbot
system_prompt = """
Role and Objective: You are a professional speech therapist chatbot designed to assist individuals with speech disabilities. Your primary objective is to provide compassionate support, reliable information, and useful resources related to speech therapy and speech disabilities. You are tasked with creating a safe and welcoming environment for users, where they feel comfortable discussing their challenges and seeking help.

Response Style:

1. Empathetic Communication: Always respond with empathy and understanding. Acknowledge the feelings and experiences of the users. Use supportive language that makes them feel heard and valued.

2. Clarity and Simplicity: Use clear and straightforward language. Avoid jargon and complex terminologies unless they are explained in simple terms. Ensure that your explanations are easy to understand.

3. Informative Guidance: Provide accurate and relevant information related to speech therapy. This includes techniques for improving speech, resources for finding professional help, and tips for communication strategies.

4. Encouragement: Always encourage users to express themselves. Remind them that their thoughts and feelings are valid. Offer motivational support and reassure them that progress is possible.

5. Respect Privacy: Do not ask for personal information unless it is necessary for providing relevant advice. Ensure that users feel their privacy is respected at all times.

Types of Questions You May Encounter:

- Users may ask about specific speech disabilities (e.g., stuttering, articulation disorders).
- Users may seek advice on exercises to improve their speech.
- Users may inquire about finding a speech therapist or resources for therapy.
- Users may want to share their experiences or seek emotional support.

Response Examples:

- If a user asks about stuttering: "Stuttering is a speech disorder that affects the flow of speech. It's important to know that many people experience stuttering at different levels. Techniques such as practicing slow speech, using pauses, and speaking in front of a mirror can help. Have you tried any specific strategies?"

- If a user shares their feelings: "I appreciate you sharing that with me. It's completely normal to feel overwhelmed at times, and it's great that you're reaching out for support. Remember, you're not alone in this journey."

- If a user asks about finding a therapist: "Finding the right speech therapist can make a significant difference. I recommend looking for professionals who specialize in your specific needs. You can check online directories or local health services for qualified therapists in your area."
"""
 # (Keep your original system prompt here)

# Chatbot Page
if page == "Chatbot":
    st.title("🗣️ Speech Therapy")
    st.markdown("""Welcome to the Speech Therapy Chatbot! I’m here to assist you with resources and support related to speech therapy.""")
    
    # Initialize session state for messages
    if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
        st.session_state["messages"] = [{"role": system_prompt, "content": "Hello! How can I assist you today?"}]
    
    # Display chat history
    for msg in st.session_state["messages"]:
        st.chat_message(msg["role"]).write(msg["content"])

    # User input
    user_query = st.chat_input(placeholder="Ask anything about speech therapy...")

    if user_query:
        # Store user query in session state
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # Show response from the chatbot
        with st.chat_message("assistant"):
            response = qa_chain.run(user_query)
            st.session_state["messages"].append({"role": "assistant", "content": response})

            # Ensure the response is displayed before accepting the next input
            st.write(response)

    # About section
    with st.expander("About", expanded=False):
        st.markdown("The Speech Therapy Chatbot aims to provide users with access to reliable speech therapy information, resources, and support.")

# Community Page
elif page == "Community":
    st.title("🗣️ Community Thoughts")
    st.markdown("Share your experiences or thoughts related to speech therapy below:")
    
    # Initialize session state for community thoughts
    if "community_thoughts" not in st.session_state:
        st.session_state["community_thoughts"] = []
    
    community_input = st.text_area("Your Thoughts:", placeholder="Type your thoughts here...")

    if st.button("Submit"):
        if community_input:
            st.session_state["community_thoughts"].append(community_input)
            st.success("Thank you for sharing your thoughts!")
        else:
            st.warning("Please enter a message before submitting.")

    # Display shared community thoughts
    st.markdown("### Shared Thoughts:")
    for thought in st.session_state["community_thoughts"]:
        st.markdown(f"- {thought}")

    # Contact section
    with st.expander("Contact", expanded=False):
        st.markdown("""For more information, please contact us at: - Email: support@speechtherapychatbot.org - Phone: +1 (800) 123-4567""")

elif page == "Resources":
    st.title("🗣️ Resources and Tips for Speech Therapy")
    st.markdown("""
    Here are some valuable resources and tips that can help you with speech therapy:
    
    ### Useful Links:
    - [American Speech-Language-Hearing Association (ASHA)](https://www.asha.org)
    - [Speech Therapy Apps](https://www.speechpathology.com)
    - [National Stuttering Association](https://westutter.org)
    
    ### Tips for Improving Speech:
    1. **Practice Regularly**: Consistency is key in speech therapy. Set aside time each day for practice.
    2. **Record Yourself**: Listening to your own speech can help you identify areas for improvement.
    3. **Use Visual Aids**: Incorporate pictures and videos to support your understanding and learning.
    4. **Stay Patient**: Progress may be slow, but every small step counts. Celebrate your achievements!
    
    ### Books and Materials:
    - *The Complete Handbook of Speech and Language Therapy* by Sheila D. Johnson
    - *Talkability: 8 Steps to Teaching Your Child to Communicate* by Fern Sussman
    
    ### Support Groups:
    - Consider joining local or online support groups to connect with others who are on similar journeys.
    """)

elif page == "Activities":
    st.title("🗣️ Speech Therapy Activities and Games")
    st.markdown("""
    Engage in fun activities designed to enhance your speech skills!
    
    ### Word Puzzle:
    Try to guess the word based on the definition provided below!
    """)

    # Word puzzle data (you can expand this list with more words)
    word_definitions = {
        "Articulation": "The clear and precise pronunciation of words.",
        "Fluency": "The smoothness or flow with which sounds, syllables, words, and phrases are joined together.",
        "Stuttering": "A speech disorder that involves frequent and significant disruptions in the normal flow of speech.",
        "Phoneme": "The smallest unit of sound in speech.",
        "Language": "A system of communication used by a particular community or country."
    }

    # Select a random word definition
    import random
    word, definition = random.choice(list(word_definitions.items()))

    # Display the definition
    st.markdown(f"**Definition:** {definition}")

    # User input for guessing the word
    user_guess = st.text_input("Your Guess:", placeholder="Type your answer here...")

    # Check the user's guess
    if st.button("Submit Guess"):
        if user_guess.lower() == word.lower():
            st.success("Correct! 🎉 The word is: " + word)
        else:
            st.error("Incorrect! 😢 Try again.")
    
    # Optionally, you can give the option to reveal the answer
    if st.button("Reveal Answer"):
        st.info(f"The word was: {word}")


# Add any other useful features or links in the sidebar
st.sidebar.markdown("### Follow us on Social Media")
st.sidebar.markdown("[Facebook](https://www.facebook.com/speechtherapychatbot)")
st.sidebar.markdown("[Twitter](https://twitter.com/speechtherapybot)")
st.sidebar.markdown("[Instagram](https://www.instagram.com/speechtherapychatbot)")
