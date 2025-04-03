
### **README.md**  
```markdown ```
# Multimodal Image Analysis with CrewAI and Gemini Vision  

## Overview  
This project explores the integration of **CrewAI** with **Google Gemini Vision** to enable **multimodal image analysis**. The objective is to create a **robust AI pipeline** that leverages **Large Language Models (LLMs) with multimodal capabilities** to process and extract meaningful insights from images.  

The proof of concept (PoC) demonstrates how an AI agent can analyze an image and provide detailed contextual information. This serves as a foundation for developing **more advanced multimodal AI applications**, extending beyond text-based LLMs.  

---

## Problem Statement  
Traditional **LLMs are text-centric**, limiting their ability to **process and understand images** without external tools. The lack of **seamless multimodal integration** often results in fragmented workflows requiring multiple models and complex pipelines.  

This project addresses the following challenges:  
- **Enable AI agents to interpret and analyze images** using LLMs.  
- **Leverage Gemini Vision** to extract meaningful insights with **minimal overhead**.  
- **Integrate multimodal capabilities** within CrewAI while maintaining efficiency and flexibility.  

---

## Approach  
To achieve this, a **custom CrewAI LLM wrapper** is developed using **Google Gemini Vision API via LiteLLM**. The framework is designed to:  
1. **Accept an image URL as input**.  
2. **Process the image using Gemini Vision**.  
3. **Generate detailed image descriptions** with context-aware insights.  
4. **Utilize CrewAI’s agent-task architecture** for structured execution.  

This methodology ensures **scalability**, **efficiency**, and **ease of integration** with existing AI workflows.  

---

## Implementation  

### 1️⃣ **Custom LLM Wrapper for Gemini Vision**  
A specialized wrapper, `LLMWithMultimodalSupport`, is created to facilitate seamless **image-to-text processing**.  

```python
from lite import LLMWithMultimodalSupport

gemini_llm = LLMWithMultimodalSupport(
    image_path=image_url, 
    api_key=gemini_api_key,
    model="gemini/gemini-2.0-flash-exp"
)
```
- Dynamically **processes images** from a given URL.  
- Utilizes **Google Gemini Vision** for analysis.  
- Structured for **scalable multimodal processing**.  

---

### 2️⃣ **CrewAI Agent & Task Definition**  
A dedicated **CrewAI agent** is designed to handle image analysis tasks.  

```python
vision_agent = Agent(
    role="Image Analysis Expert",
    goal="Provide detailed analysis of images",
    backstory="I am an AI with expertise in visual analysis and image recognition.",
    llm=gemini_llm
)

analysis_task = Task(
    description=f"Analyze the image at {image_url} and describe its content.",
    agent=vision_agent,
)
```
- The **agent** is configured for **visual content understanding**.  
- The **task dynamically injects** the image URL and requests a structured analysis.  

---

### 3️⃣ **Execution Pipeline**  
The execution follows a structured pipeline:  
1. **Input an image URL**.  
2. **CrewAI processes the request** and sends it to Gemini Vision.  
3. **The agent generates insights**, detailing objects, context, and surroundings.  

```python
def run_crew(image_url):
    analysis_task.update_description(f"Analyze the image at {image_url}.")
    crew = Crew(agents=[vision_agent], tasks=[analysis_task])
    return crew.kickoff()
```
- **The task is dynamically updated** with the provided image.  
- **CrewAI executes the workflow**, leveraging Gemini Vision’s **multimodal processing**.  

---

## Expected Output  
### **Input:**  
> `https://storage.googleapis.com/github-repo/img/gemini/intro/landmark3.jpg`  

### **Output:**  
```
The image shows the Colosseum in Rome, Italy, at dusk or dawn. 
It is a large ancient amphitheater with illuminated arches. The sky has a gradient of blues, 
and some greenery is visible. This is a historical landmark and popular tourist destination.
```
- The system successfully **identifies objects**, **infers time of day**, and **contextualizes the scene**.  

---

## Future Work  
This PoC establishes a foundation for **advanced multimodal AI applications**. Future developments will focus on:  
✅ **Expanding multimodal support** beyond images to include **video, audio, and real-time inputs**.  
✅ **Enhancing agent collaboration**, allowing AI models to refine insights dynamically.  
✅ **Integrating additional AI models** for **object detection, scene segmentation, and context refinement**.  

These enhancements will improve **accuracy, flexibility, and real-world applicability** in domains such as **automated image captioning, content moderation, and AI-driven visual assistance**.  

---

## Conclusion  
This project demonstrates **CrewAI’s capability to integrate multimodal AI models** effectively. By incorporating **Google Gemini Vision**, it provides a **structured approach** to image analysis, paving the way for more sophisticated applications in **computer vision, AI-driven insights, and automated content understanding**.  

This PoC serves as a **baseline for further research and development**, extending multimodal AI’s impact across various industries.  


---
