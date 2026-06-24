import os
import requests
import streamlit as st
from datetime import datetime
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astropy import units as u
from langchain_core.tools import tool
from langchain.agents import create_agent # Updated import for LangGraph/LangChain V1.0+

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
# TOOLS (Weather & Astronomical Ephemeris)
# ==========================================
@tool
def get_weather_conditions(lat: float, lon: float) -> str:
    """Fetches real-time hour-by-hour cloud cover and seeing conditions for a specific latitude and longitude."""
    # Using timezone=auto so Open-Meteo automatically adjusts to the user's local time based on coordinates
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=cloudcover,visibility&timezone=auto&forecast_days=1"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        times = data['hourly']['time']
        cloud_cover = data['hourly']['cloudcover']
        
        current_hour_idx = datetime.now().hour
        report = f"Tonight's Hourly Cloud Cover (%) for Lat: {lat}, Lon: {lon}:\n"
        for i in range(current_hour_idx, min(current_hour_idx + 12, len(times))):
            report += f"{times[i]}: {cloud_cover[i]}% clouds\n"
        return report
    return "Error: Could not fetch weather data."

@tool
def get_target_transit(target_name: str, lat: float, lon: float) -> str:
    """Calculates the transit time, altitude, and visibility of a celestial target for a specific latitude and longitude."""
    try:
        observatory = EarthLocation(lat=lat*u.deg, lon=lon*u.deg, height=200*u.m)
        time_now = Time.now()
        target = SkyCoord.from_name(target_name)
        
        delta_hours = range(0, 12)
        times = time_now + delta_hours * u.hour
        frame = AltAz(obstime=times, location=observatory)
        target_altaz = target.transform_to(frame)
        
        max_alt_idx = target_altaz.alt.argmax()
        max_alt = target_altaz.alt[max_alt_idx].degree
        transit_time = times[max_alt_idx].iso
        
        return (f"Target: {target_name}\n"
                f"Observatory Location: Lat {lat}, Lon {lon}\n"
                f"Peak Altitude: {max_alt:.2f} degrees\n"
                f"Transit Time (Local): {transit_time}\n"
                f"Target is above 30 degrees for: {sum(target_altaz.alt.degree > 30)} hours tonight.")
    except Exception as e:
        return f"Error resolving target '{target_name}'. Ensure it is a valid catalog name. Details: {e}"

# ==========================================
# ORCHESTRATOR FACTORY (Dynamic LLM Selection)
# ==========================================
def build_orchestrator(provider: str, api_key: str, mount: str, optics: str, camera: str, lat: float, lon: float):
    # Route to the chosen provider and instantiate the correct LangChain wrapper
    if provider == "Google (Gemini 1.5 Flash)":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, google_api_key=api_key)
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
        
    tools = [get_weather_conditions, get_target_transit]
    
    system_prompt = f"""
    You are an expert astrophotography orchestrator. Design the perfect imaging session plan 
    by synthesizing meteorological data and celestial ephemeris targets.
    
    Context Parameters:
    - Mount: {mount}
    - Optics: {optics}
    - Camera: {camera}
    - Observatory Latitude: {lat}
    - Observatory Longitude: {lon}
    
    Workflow:
    1. Check the weather using the `get_weather_conditions` tool. You MUST pass the exact Observatory Latitude and Longitude provided above.
    2. Check the target's altitude and transit using `get_target_transit`. You MUST pass the exact Observatory Latitude and Longitude provided above.
    3. Synthesize this data into a structured markdown itinerary.
    4. Provide specific advice on when to execute the meridian flip, factoring in the specific mount's capabilities (e.g., harmonic drives can track past the meridian safely).
    5. Recommend sub-exposure times and gain/ISO settings appropriate for the target, optics, and specific camera sensor characteristics.
    """
    
    return create_agent(llm, tools, prompt=system_prompt)

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
            "Google (Gemini 1.5 Flash)",
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

    st.markdown("---")
    st.subheader("🔭 Hardware Setup")
    mount_input = st.text_input("Mount", value="ZWO AM5")
    optics_input = st.text_input("Optics", value="Apertura")
    camera_input = st.text_input("Camera", value="ZWO ASI2600MC Pro")
    
    st.markdown("---")
    st.markdown("### Active Agent Tools")
    st.markdown("- 🌤️ `get_weather_conditions(lat, lon)`")
    st.markdown("- 🪐 `get_target_transit(target, lat, lon)`")

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
                    camera=camera_input,
                    lat=user_lat,
                    lon=user_lon
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