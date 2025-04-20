import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL

# Page configuration
st.set_page_config(
    page_title="Youtube Summarizer",
    page_icon=":movie_camera:",
    layout="wide",
)
st.title("LangChain: Summarize any URL. Youtube/Web URL Summarizer")
st.subheader("Summarize any Youtube or Web URL with LangChain")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Enter your Groq API Key", value="", type="password")
    
url = st.text_input("Enter a Youtube or Web URL", value="", placeholder="https://www.youtube.com/watch?v=example")

# Define prompt template
prompt_temp = """
Provide a summary of the following content in 300 words:
Context = {text}
"""

prompt = PromptTemplate(
    input_variables=["text"],
    template=prompt_temp,
)

# Main functionality
if st.button("Summarize"):
    # Input validation
    if not groq_api_key.strip() or not url.strip():
        st.error("Please enter a Groq API key and a URL.")
    elif not validators.url(url):
        st.error("Please enter a valid URL.")
    else:
        try:
            # Initialize LLM with API key
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model="llama-3.3-70b-versatile"
            )
            
            # Process based on URL type
            with st.spinner("Loading content and generating summary..."):
                if "youtube.com" in url:
                   loader = YoutubeLoader.from_youtube_url(
                        url,
                        add_video_info=True,
                        continue_on_failure=True,   # ← don’t bail out if captions missing
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
                    )
                
                # Load data and create summary
                data = loader.load()
                
                if not data:
                    st.error("Could not extract content from the provided URL.")
                else:
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="stuff",
                        verbose=True,
                        prompt=prompt,
                    )
                    
                    output = chain.invoke(data)
                    
                    st.success(output)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
