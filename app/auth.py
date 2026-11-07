import html
import re

import streamlit as st
from supabase import Client, create_client

from app.config import SUPABASE_URL, SUPABASE_ANON_KEY, APP_URL


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


def _handle_email_verification_callback() -> None:
    """Exchange Supabase PKCE callback codes (email confirmation or OAuth) for an app session."""
    code = st.query_params.get("code")
    if not code:
        return

    client = st.session_state.get("auth_client")
    if client is None:
        st.error(
            "Authentication returned to the app, but its PKCE verifier is missing. "
            "Please retry Google sign-in in the same browser tab. If this persists, check the OAuth callback/session configuration."
        )
        return

    try:
        result = client.auth.exchange_code_for_session(code)
        if not result.user or not result.session:
            raise RuntimeError("Supabase did not return an authenticated session.")
        st.session_state.auth_client = client
        st.session_state.auth_user = {
            "id": str(result.user.id),
            "email": result.user.email,
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
        }
        st.query_params.clear()
        st.success("Signed in successfully.")
        st.rerun()
    except Exception as exc:
        st.error(f"Email verification callback failed — {_safe_auth_error(exc)}")


def require_authentication() -> bool:
    """Render login/signup and return True only for an authenticated session."""
    _handle_email_verification_callback()
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
    st.subheader("Your documents. Your answers.")
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
                # Keep this client in the Streamlit session: Supabase PKCE stores
                # the code verifier in the client's auth storage until callback.
                st.session_state.auth_client = client
                result = client.auth.sign_up({"email": email, "password": password})
                if not result.session:
                    st.success("Account created. Check your email and open the latest confirmation link.")
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
    st.divider()
    st.caption("Or continue with")
    if st.button("Continue with Google", use_container_width=True):
        try:
            client = _client()
            # Supabase PKCE keeps its code verifier in this client's auth storage.
            # Retain the same client for the callback code exchange.
            st.session_state.auth_client = client
            oauth = client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": APP_URL},
            })
            if not oauth.url:
                raise RuntimeError("Supabase did not return a Google authorization URL.")
            st.session_state["google_oauth_url"] = oauth.url
        except Exception as exc:
            st.error(f"Google sign-in could not be started — {_safe_auth_error(exc)}")
    if st.session_state.get("google_oauth_url"):
        # Force the OAuth round-trip in this tab; opening a second tab creates a
        # separate Streamlit session and can lose the in-memory PKCE verifier.
        oauth_url = html.escape(st.session_state["google_oauth_url"], quote=True)
        st.markdown(
            f'<a href="{oauth_url}" target="_self" rel="noopener noreferrer">Continue to Google</a>',
            unsafe_allow_html=True,
        )
    st.caption("Your documents remain isolated to your account.")
    return False
