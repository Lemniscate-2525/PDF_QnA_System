import streamlit as st
from supabase import Client, create_client

from app.config import SUPABASE_URL, SUPABASE_ANON_KEY


def _client() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Configure SUPABASE_URL and SUPABASE_ANON_KEY.")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def current_user():
    return st.session_state.get("auth_user")


def require_authentication() -> bool:
    """Render login/signup and return True only for an authenticated session."""
    if current_user():
        with st.sidebar:
            st.caption(f"Signed in as {current_user().get('email', 'user')}")
            if st.button("Sign out", use_container_width=True):
                try:
                    st.session_state.auth_client.auth.sign_out()
                finally:
                    for key in ("auth_user", "auth_client"):
                        st.session_state.pop(key, None)
                    st.rerun()
        return True

    st.title("PDF Q&A")
    st.subheader("Sign in to your private workspace")
    mode = st.radio("Account", ["Sign in", "Create account"], horizontal=True)
    with st.form("auth_form"):
        email = st.text_input("Email", max_chars=254).strip().lower()
        password = st.text_input("Password", type="password", max_chars=128)
        submitted = st.form_submit_button(mode, type="primary", use_container_width=True)

    if submitted:
        if not email or len(email) > 254 or not password or len(password) < 8:
            st.error("Enter a valid email and a password of at least 8 characters.")
            return False
        try:
            client = _client()
            if mode == "Create account":
                result = client.auth.sign_up({"email": email, "password": password})
                if not result.session:
                    st.success("Account created. Verify your email, then sign in.")
                    return False
            else:
                result = client.auth.sign_in_with_password({"email": email, "password": password})
            if not result.user or not result.session:
                st.error("Authentication did not return an active session.")
                return False
            st.session_state.auth_client = client
            st.session_state.auth_user = {
                "id": str(result.user.id),
                "email": result.user.email,
                "access_token": result.session.access_token,
                "refresh_token": result.session.refresh_token,
            }
            st.rerun()
        except Exception:
            st.error("Authentication failed. Check your credentials or account verification.")
    st.caption("Each account can access only documents it owns.")
    return False
