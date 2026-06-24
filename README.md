# 🔭 AstroAI Multi-Model Orchestrator

An enterprise-grade, multi-agent benchmarking framework designed to autonomously orchestrate astrophotography session planning. 

Built with **Streamlit** and **LangGraph**, this application demonstrates complex LLM tool-use, stateful agentic routing, and dynamic hardware-aware reasoning across multiple frontier models (Google Gemini, Anthropic Claude, OpenAI, and Groq). 

By synthesizing real-time meteorological data with complex celestial ephemeris, the orchestrator generates optimized, location-specific imaging itineraries—factoring in local observatory coordinates, mount capabilities (e.g., harmonic drive meridian tracking), optical focal lengths, and camera sensor characteristics.

## ✨ Features
* **Agnostic Model Routing:** Instantly switch between frontier AI providers to benchmark tool-use and reasoning capabilities on identical tasks.
* **📍 Location-Aware Routing:** Input exact geographic coordinates (Latitude/Longitude) to generate highly localized weather forecasts and precise celestial transit times, completely decoupled from static IP locations.
* **Dynamic Tool Use:**
    * 🌤️ `get_weather_conditions(lat, lon)`: Queries the Open-Meteo API for exact hour-by-hour cloud cover and atmospheric seeing conditions at the specified coordinates.
    * 🪐 `get_target_transit(target, lat, lon)`: Utilizes `astropy` to calculate localized target altitudes, transit times, and visibility windows relative to the custom observatory location.
* **🔭 Hardware-Aware Reasoning:** Injects custom hardware context (e.g., ZWO AM5 mount, Apertura optics, ASI2600MC Pro sensors) into the agent state, allowing models to deduce specific meridian flip timings, sub-exposure lengths, and sensor gain settings.

## 🚀 Quick Start

### Prerequisites
* Python 3.9+
* API Keys for the models you wish to test (Anthropic, Google GenAI, OpenAI, Groq).

### Installation
1. Clone this repository:
```bash
git clone [https://github.com/yourusername/astro-ai-orchestrator.git](https://github.com/yourusername/astro-ai-orchestrator.git)
cd astro-ai-orchestrator
```

2. Install the required dependencies:
```bash
pip install streamlit langgraph langchain langchain-anthropic langchain-google-genai langchain-openai langchain-groq astropy requests
```

3. Launch the application:
```bash
streamlit run app.py
```
*(Note for Windows users: If the Streamlit server hangs on startup, you can run it in headless mode via `python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false`)*

## 🛠️ Adding Additional Models
This architecture is designed to be easily extensible. To add a new LLM provider (e.g., Cohere, Mistral):

1. **Install the Integration Package:** Add the LangChain wrapper for the provider (e.g., `pip install langchain-cohere`).
2. **Update the Imports:** Add the chat model to the top of `app.py`.
3. **Expand the UI Dropdown:** Add the provider name to the `provider_choice` list inside the Streamlit sidebar configuration.
4. **Update the Orchestrator Factory:** Add an `elif` block in the `build_orchestrator` function to map your new provider string to its respective LangChain instantiation logic.

## 🌟 Example Use Cases

* **The Narrowband Deep Dive:** "Plan a session for the Rosette Nebula using my ASI2600MC Pro. Give me the optimal gain setting and sub-exposure length to maximize faint detail while avoiding lunar interference."
* **The Multi-Target Marathon:** "I have a clear window from 10 PM to 4 AM. Plan an itinerary that starts with the Orion Nebula and switches to the Horsehead Nebula when it reaches peak altitude."
* **The Harmonic Drive Advantage:** "I'm using a ZWO AM5. When exactly should I execute the meridian flip for the Whirlpool Galaxy tonight to maximize continuous tracking time?"

## ⚠️ A Note on Frontier AI Evolution
The libraries and models powering this application are in a state of rapid, constant evolution. 

* **LangGraph/LangChain Updates:** Package structures change frequently. For instance, the core agent generation function recently migrated from `langgraph.prebuilt.create_react_agent` to `langchain.agents.create_agent`. If you encounter deprecation warnings or import errors, check the latest LangChain documentation.
* **Model Deprecation:** AI providers aggressively deprecate older model versions. If you encounter a `404 NOT_FOUND` or `RESOURCE_EXHAUSTED` error when routing to a specific model (such as `gemini-1.5-pro`), update the model instantiation string in the `build_orchestrator` function to point to the provider's newest active release (e.g., `gemini-2.5-pro` or `gemini-2.5-flash`).

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to open an issue or submit a pull request.
