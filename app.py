import validators
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import os
import subprocess
import sys
import pytube
import requests
from youtube_transcript_api import YouTubeTranscriptApi

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
                if "youtube.com" in url or "youtu.be" in url:
                    try:
                        # Extract video ID
                        if "youtube.com" in url:
                            video_id = url.split("v=")[1].split("&")[0]
                        else:  # youtu.be format
                            video_id = url.split("/")[-1].split("?")[0]
                            
                        # First try with YouTubeTranscriptApi
                        try:
                            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                            transcript_text = " ".join([t["text"] for t in transcript_list])
                            
                            # Create a document-like object for the summarization chain
                            class TranscriptDocument:
                                def __init__(self, page_content):
                                    self.page_content = page_content
                                    self.metadata = {"source": url}
                            
                            data = [TranscriptDocument(transcript_text)]
                            st.info("Using video transcript for summarization")
                        except Exception as transcript_error:
                            # Fall back to YoutubeLoader
                            st.info(f"Transcript error: {str(transcript_error)}. Attempting to download video content...")
                            loader = YoutubeLoader.from_youtube_url(
                                url, 
                                add_video_info=True,
                                language=["en"]
                            )
                            data = loader.load()
                    except pytube.exceptions.RegexMatchError:
                        st.error(f"Could not extract video ID from URL: {url}")
                        data = []
                    except requests.exceptions.HTTPError as e:
                        st.error(f"HTTP Error when accessing YouTube: {str(e)}")
                        st.info("This may be due to region restrictions, age restrictions, or the video might be private.")
                        data = []
                    except Exception as e:
                        st.error(f"Error accessing YouTube content: {str(e)}")
                        st.info("Try using a different video or checking if the video is publicly available.")
                        data = []
                else:
                    try:
                        loader = UnstructuredURLLoader(
                            urls=[url],
                            ssl_verify=False,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
                        )
                        data = loader.load()
                    except Exception as e:
                        st.error(f"Error loading content from URL: {str(e)}")
                        data = []
                
                if not data:
                    st.error("Could not extract content from the provided URL.")
                else:
                    chain = load_summarize_chain(
                        llm=llm,
                        chain_type="stuff",
                        verbose=True,
                        prompt=prompt,
                    )
                    
                    output = chain.run(data)
                    
                    st.success(output)
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.info("Try checking your API key or using a different URL.")
