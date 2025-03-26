
# client.py
import os
from crewai import Agent, Task, Crew, LLM
from lite import LLMWithMultimodalSupport
from dotenv import load_dotenv

def run_crew():
    """Run a CrewAI crew with Gemini Vision via LiteLLM."""
    # Load environment variables
    load_dotenv()
    
    # Image to analyze
    image_url = 'https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg'
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")
    
    # Create the LiteLLM Gemini Vision instance
    gemini_llm = LLMWithMultimodalSupport(
        image_path=image_url, 
        api_key=gemini_api_key,
        model="gemini/gemini-2.0-flash-exp")
    
    # Define a Crew AI Agent that uses Gemini Vision
    vision_agent = Agent(
        role="Image Analysis Expert",
        goal="Provide detailed analysis of images",
        backstory="I am an AI with expertise in visual analysis and image recognition.",
        verbose=True,
        # Use our custom LLM
        llm=gemini_llm
    )
    
    # Define a vision-based task
    analysis_task = Task(
        description=f"Analyze the image at {image_url} and describe all visible objects, landmarks, people, and the overall context of the image.",
        expected_output="what is the prominent object in the image or what the image is about alongth with a few bullet points about the most striking features of the image.",
        agent=vision_agent,
    )
    
    # Create a crew with the vision agent
    crew = Crew(
        agents=[vision_agent],
        tasks=[analysis_task],
        verbose=False
    )
    
    # Run the crew
    result = crew.kickoff()
    return result

if __name__ == "__main__":
    print("Starting CrewAI with LiteLLM Gemini Vision...")
    result = run_crew()
    print("\nFinal Result:")
    print(result)






