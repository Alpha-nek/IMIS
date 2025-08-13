# Add this at the top of main.py after the imports:

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4 0%, #ff7f0e 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
    }
    .warning-card {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
    }
    .error-card {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Update the header in the main() function:
def main():
    """Main application function."""
    initialize_session_state()
    
    # Professional header
    st.markdown("""
    <div class="main-header">
        <h1>�� IMIS Scheduler</h1>
        <p>Professional Hospitalist Scheduling System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Provider status indicator with better styling
    if st.session_state.get("providers_loaded", False) and not st.session_state.providers_df.empty:
        provider_count = len(st.session_state.providers_df)
        st.markdown(f"""
        <div class="metric-card success-card">
            <h4>✅ System Ready</h4>
            <p>{provider_count} providers loaded and ready for scheduling</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="metric-card error-card">
            <h4>❌ System Not Ready</h4>
            <p>No providers loaded. Please go to the Providers tab to load providers.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Rest of the main function remains the same...
