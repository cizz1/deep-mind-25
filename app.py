import streamlit as st
import os
from crewai import Agent, Task, Crew
from lite import LLMWithMultimodalSupport
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Gemini API key
gemini_api_key = os.getenv("GEMINI_API_KEY")

# Streamlit UI
st.title("🔍 Image Analysis with CrewAI & Gemini Vision")
st.write("Enter an image URL below and click 'Analyze Image' to get insights.")

# Input field for image URL
image_url = st.text_input("Enter Image URL:", "")

# Button to run analysis
if st.button("Analyze Image"):
    if not image_url:
        st.error("Please enter a valid image URL.")
    elif not gemini_api_key:
        st.error("GEMINI_API_KEY is missing. Please set it in your environment.")
    else:
        with st.spinner("Analyzing the image..."):
            # Run CrewAI task
            def run_crew(image_url):
                """Runs the image analysis using CrewAI."""
                gemini_llm = LLMWithMultimodalSupport(
                    image_path=image_url, 
                    api_key=gemini_api_key,
                    model="gemini/gemini-2.0-flash-exp"
                )

                vision_agent = Agent(
                    role="Image Analysis Expert",
                    goal="Provide detailed analysis of images",
                    backstory="I am an AI with expertise in visual analysis and image recognition.",
                    verbose=False,
                    llm=gemini_llm
                )

                analysis_task = Task(
                description=f"Analyze the image at {image_url} and describe all visible objects, landmarks, people, and the overall context of the image.",
                expected_output="what is the prominent object in the image or what the image is about alongth with a few bullet points about the most striking features of the image.",
                agent=vision_agent,
                verbose=False
            )

                crew = Crew(
                    agents=[vision_agent],
                    tasks=[analysis_task],
                    verbose=False
                )

                return crew.kickoff()

            # Get the result
            result = run_crew(image_url)
            st.success("Analysis Complete!")
            st.write("### Image Insights:")
            st.write(result)
            st.image(image_url, caption="Analyzed Image", use_container_width=True)
