import re

import streamlit as st
from supabase import Client, create_client

from app.config import SUPABASE_URL, SUPABASE_ANON_KEY


def _client() -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Configure SUPABASE_URL and SUPABASE_ANON_KEY.")
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)


def _safe_auth_error(exc: Exception) -> str:
    """Return useful provider diagnostics without leaking configured secrets."""
    message = str(exc).strip() or "No additional error message was returned."
    for secret in (SUPABASE_ANON_KEY, SUPABASE_URL):
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"Bearer\\s+\\S+", "Bearer [redacted]", message, flags=re.IGNORECASE)
    return f"{type(exc).__name__}: {message[:400]}"


def current_user():
    return st.session_state.get("auth_user")


def require_authentication() -> bool:
    """Render login/signup and return True only for an authenticated session."""
    if current_user():
        try:
            verified = st.session_state.auth_client.auth.get_user(
                current_user()["access_token"]
            )
            if not verified.user or str(verified.user.id) != current_user()["id"]:
                raise RuntimeError("Session identity mismatch.")
        except Exception:
            try:
                refreshed = st.session_state.auth_client.auth.refresh_session(
                    current_user()["refresh_token"]
                )
                if not refreshed.user or not refreshed.session:
                    raise RuntimeError("Session refresh failed.")
                st.session_state.auth_user = {
                    "id": str(refreshed.user.id),
                    "email": refreshed.user.email,
                    "access_token": refreshed.session.access_token,
                    "refresh_token": refreshed.session.refresh_token,
                }
            except Exception:
                st.session_state.pop("auth_user", None)
                st.session_state.pop("auth_client", None)
                return False
        with st.sidebar:
            st.caption(f"Signed in as {current_user().get('email', 'user')}")
            if st.button("Sign out", use_container_width=True):
                try:
                    st.session_state.auth_client.auth.sign_out()
                finally:
                    for key in ("auth_user", "auth_client", "selected_doc_id", "last_ingestion_job"):
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
        except Exception as exc:
            st.error(f"Authentication request failed — {_safe_auth_error(exc)}")
    st.caption("Each account can access only documents it owns.")
    return False
