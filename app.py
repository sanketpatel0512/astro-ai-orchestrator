from datetime import date

import streamlit as st
from langchain.agents import create_agent # Updated import for LangGraph/LangChain V1.0+

from astro_tools import ALL_TOOLS

# --- Multi-Model Integrations ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="AstroAI Multi-Model Orchestrator", page_icon="🔭", layout="wide")

# ==========================================
# ORCHESTRATOR FACTORY (Dynamic LLM Selection)
# ==========================================
def build_orchestrator(provider: str, api_key: str, mount: str, optics: str, focal_length_mm: float, camera: str, lat: float, lon: float, session_date: str):
    # Route to the chosen provider and instantiate the correct LangChain wrapper
    if provider == "Google (Gemini 2.5 Flash)":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    elif provider == "Google (Gemini 2.5 Pro)":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2, google_api_key=api_key)
    elif provider == "Anthropic (Claude 3.5 Sonnet)":
        llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.2, api_key=api_key)
    elif provider == "OpenAI (ChatGPT / GPT-4o)":
        llm = ChatOpenAI(model="gpt-4o", temperature=0.2, api_key=api_key)
    elif provider == "Groq (Llama 3 70B)":
        llm = ChatGroq(model="llama3-70b-8192", temperature=0.2, groq_api_key=api_key)
    else:
        raise ValueError("Unknown AI provider selected.")
        
    system_prompt = f"""
    You are an expert astrophotography orchestrator. Design the perfect imaging session plan
    by synthesizing meteorological data and celestial ephemeris targets.

    Context Parameters:
    - Mount: {mount}
    - Optics: {optics} ({focal_length_mm:g}mm focal length)
    - Camera: {camera}
    - Observatory Latitude: {lat}
    - Observatory Longitude: {lon}
    - Session Date: {session_date}

    Workflow:
    1. Use the Observatory Latitude and Longitude above for every location-based tool. Only call `geocode_location` if the user names a different place.
    2. Check cloud cover with `get_weather_conditions`, the darkness window with `get_darkness_window`, and moon interference with `get_moon_phase`, all for the Session Date.
    3. Check the target's altitude and transit using `get_target_transit` for the Session Date, and framing with `get_field_of_view` using the focal length and camera above.
    4. Synthesize this data into a structured markdown itinerary, and record it with `save_session_plan`.
    5. Provide specific advice on when to execute the meridian flip, factoring in the specific mount's capabilities (e.g., harmonic drives can track past the meridian safely).
    6. Recommend sub-exposure times and gain/ISO settings appropriate for the target, optics, and specific camera sensor characteristics.
    """

    return create_agent(llm, ALL_TOOLS, system_prompt=system_prompt)

# ==========================================
# UI LAYOUT
# ==========================================
st.title("🔭 AstroAI Multi-Model Observation Orchestrator")
st.markdown("An agnostic agentic ecosystem to autonomously plan astrophotography targets across global frontier models.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Expanded Model Dropdown
    provider_choice = st.selectbox(
        "Select AI Provider", 
        [
            "Google (Gemini 2.5 Flash)",
            "Google (Gemini 2.5 Pro)", 
            "Anthropic (Claude 3.5 Sonnet)", 
            "OpenAI (ChatGPT / GPT-4o)", 
            "Groq (Llama 3 70B)"
        ]
    )
    
    # Extract clean provider name for the input label
    provider_name = provider_choice.split(" ")[0]
    api_key = st.text_input(f"{provider_name} API Key", type="password", help=f"Authentication for {provider_name}.")
    
    st.markdown("---")
    st.subheader("🌍 Observatory Location")
    # Defaulting to a central US coordinate if user doesn't change it
    user_lat = st.number_input("Latitude", value=30.4548, format="%.4f", help="Use positive for North, negative for South.")
    user_lon = st.number_input("Longitude", value=-97.6223, format="%.4f", help="Use positive for East, negative for West.")
    session_date = st.date_input("Session Date", value=date.today())

    st.markdown("---")
    st.subheader("🔭 Hardware Setup")
    mount_input = st.text_input("Mount", value="ZWO AM5")
    optics_input = st.text_input("Optics", value="Apertura")
    focal_length_input = st.number_input("Focal Length (mm)", value=600.0, min_value=10.0, step=10.0)
    camera_input = st.text_input("Camera", value="ZWO ASI2600MC Pro")

    st.markdown("---")
    st.markdown("### Active Agent Tools")
    st.markdown("- 📍 `geocode_location(location)`")
    st.markdown("- 🌤️ `get_weather_conditions(lat, lon, date)`")
    st.markdown("- 🪐 `get_target_transit(target, lat, lon, date)`")
    st.markdown("- 🌙 `get_moon_phase(date)`")
    st.markdown("- 🌌 `get_darkness_window(lat, lon, date)`")
    st.markdown("- 🖼️ `get_field_of_view(focal_length_mm, sensor)`")
    st.markdown("- 📝 `save_session_plan(target, start, end)`")

# Main Dashboard Interface
target_object = st.text_input("Target Object (e.g., Rosette Nebula, M51, Whirlpool Nebula)", value="Rosette Nebula")
user_prompt = st.text_area("Custom Strategy Context", value="What time should I start, and when is the best continuous imaging window?")

if st.button("Execute Multi-Agent Synthesis", type="primary"):
    if not api_key:
        st.error(f"Please provide your secret {provider_name} API Key in the sidebar configuration.")
    elif not target_object:
        st.error("Target catalog or object designation missing.")
    else:
        with st.spinner(f"Active framework: {provider_name} core router executing node graph tasks..."):
            try:
                # Compile runtime graph
                app_agent = build_orchestrator(
                    provider=provider_choice, 
                    api_key=api_key, 
                    mount=mount_input, 
                    optics=optics_input,
                    focal_length_mm=focal_length_input,
                    camera=camera_input,
                    lat=user_lat,
                    lon=user_lon,
                    session_date=session_date.isoformat()
                )
                
                # Execute graph logic
                full_prompt = f"Target: {target_object}. {user_prompt}"
                inputs = {"messages": [("user", full_prompt)]}
                
                for chunk in app_agent.stream(inputs, stream_mode="values"):
                    final_message = chunk["messages"][-1]
                
                # Safely extract human-readable text, stripping away metadata
                raw_content = final_message.content
                if isinstance(raw_content, list):
                    clean_text = "\n".join([block["text"] for block in raw_content if isinstance(block, dict) and block.get("type") == "text"])
                else:
                    clean_text = raw_content

                # Render runtime response
                st.success(f"Execution complete via {provider_name} network node.")
                st.markdown(clean_text)
                
            except Exception as e:
                st.error(f"An exception occurred during graph execution loop: {e}")
