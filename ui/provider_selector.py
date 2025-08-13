"""
Provider Selector Component for IMIS Scheduler
Shows available providers and allows easy addition of new ones
"""
import streamlit as st
import pandas as pd
from core.data_manager import load_provider_names

def render_provider_selector():
    """Render provider selector with available providers from text file."""
    st.subheader("👥 Available Providers")
    
    # Load all provider names from text file
    all_provider_names = load_provider_names()
    
    if not all_provider_names:
        st.warning("No provider names found in 'provider full name.txt' file.")
        return
    
    # Get current active providers
    current_providers = set()
    if not st.session_state.providers_df.empty:
        current_providers = set(st.session_state.providers_df["initials"].astype(str).str.upper().tolist())
    
    # Separate providers by type
    physicians = {}
    apps = {}
    
    for initials, full_name in all_provider_names.items():
        if 'NP' in full_name or 'PA' in full_name:
            apps[initials] = full_name
        else:
            physicians[initials] = full_name
    
    # Show current providers
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🏥 Current Physicians:**")
        if not st.session_state.providers_df.empty:
            current_physicians = st.session_state.providers_df[
                st.session_state.providers_df["type"] == "Physician"
            ]
            for _, provider in current_physicians.iterrows():
                initials = provider["initials"]
                name = provider["name"]
                st.write(f"✅ {initials}: {name}")
        else:
            st.info("No physicians currently loaded")
    
    with col2:
        st.markdown("**👩‍⚕️ Current APPs:**")
        if not st.session_state.providers_df.empty:
            current_apps = st.session_state.providers_df[
                st.session_state.providers_df["type"] == "APP"
            ]
            for _, provider in current_apps.iterrows():
                initials = provider["initials"]
                name = provider["name"]
                st.write(f"✅ {initials}: {name}")
        else:
            st.info("No APPs currently loaded")
    
    # Show available providers to add
    st.markdown("---")
    st.subheader("➕ Add New Providers")
    
    # Physicians section
    if physicians:
        st.markdown("**🏥 Available Physicians:**")
        available_physicians = {k: v for k, v in physicians.items() if k not in current_providers}
        
        if available_physicians:
            for initials, full_name in list(available_physicians.items())[:10]:  # Show first 10
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    st.write(f"**{initials}**")
                with col2:
                    st.write(full_name)
                with col3:
                    if st.button("Add", key=f"add_physician_{initials}"):
                        # Add to session state
                        new_provider = pd.DataFrame([{
                            "initials": initials,
                            "name": full_name,
                            "type": "Physician"
                        }])
                        
                        if st.session_state.providers_df.empty:
                            st.session_state.providers_df = new_provider
                        else:
                            st.session_state.providers_df = pd.concat([
                                st.session_state.providers_df, new_provider
                            ], ignore_index=True)
                        
                        st.session_state.providers_loaded = True
                        
                        # Auto-save
                        from core.data_manager import save_providers
                        save_providers(st.session_state.providers_df, st.session_state.get('provider_rules', {}))
                        
                        st.success(f"Added {initials}: {full_name}")
                        st.rerun()
        else:
            st.info("All available physicians are already added")
    
    # APPs section
    if apps:
        st.markdown("**👩‍⚕️ Available APPs:**")
        available_apps = {k: v for k, v in apps.items() if k not in current_providers}
        
        if available_apps:
            for initials, full_name in list(available_apps.items())[:10]:  # Show first 10
                col1, col2, col3 = st.columns([2, 3, 1])
                with col1:
                    st.write(f"**{initials}**")
                with col2:
                    st.write(full_name)
                with col3:
                    if st.button("Add", key=f"add_app_{initials}"):
                        # Add to session state
                        new_provider = pd.DataFrame([{
                            "initials": initials,
                            "name": full_name,
                            "type": "APP"
                        }])
                        
                        if st.session_state.providers_df.empty:
                            st.session_state.providers_df = new_provider
                        else:
                            st.session_state.providers_df = pd.concat([
                                st.session_state.providers_df, new_provider
                            ], ignore_index=True)
                        
                        st.session_state.providers_loaded = True
                        
                        # Auto-save
                        from core.data_manager import save_providers
                        save_providers(st.session_state.providers_df, st.session_state.get('provider_rules', {}))
                        
                        st.success(f"Added {initials}: {full_name}")
                        st.rerun()
        else:
            st.info("All available APPs are already added")
    
    # Show statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Available", len(all_provider_names))
    
    with col2:
        st.metric("Physicians Available", len(physicians))
    
    with col3:
        st.metric("APPs Available", len(apps))
