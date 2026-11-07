import html
import re
import secrets
from urllib.parse import urlencode

import streamlit as st
from supabase import Client, ClientOptions, create_client

from app.config import SUPABASE_URL, SUPABASE_ANON_KEY, APP_URL
from app.redis_store import RedisAuthStorage


def _client(storage=None) -> Client:
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise RuntimeError("Configure SUPABASE_URL and SUPABASE_ANON_KEY.")
    options = ClientOptions(
        storage=storage,
        flow_type="pkce",
        auto_refresh_token=False,
        persist_session=storage is not None,
    )
    return create_client(SUPABASE_URL, SUPABASE_ANON_KEY, options=options)


def _safe_auth_error(exc: Exception) -> str:
    message = str(exc).strip() or "No additional error message was returned."
    for secret in (SUPABASE_ANON_KEY, SUPABASE_URL):
        if secret:
            message = message.replace(secret, "[redacted]")
    message = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", message, flags=re.IGNORECASE)
    return f"{type(exc).__name__}: {message[:400]}"


def current_user():
    return st.session_state.get("auth_user")


def _handle_auth_callback() -> None:
    code = st.query_params.get("code")
    if not code:
        return

    flow_id = st.query_params.get("oauth_flow")
    if flow_id:
        if not re.fullmatch(r"[a-f0-9]{48}", flow_id):
            st.error("The Google sign-in link is invalid or expired. Please start again.")
            return
        storage = RedisAuthStorage(flow_id)
        try:
            client = _client(storage=storage)
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
            storage.clear()
            st.query_params.clear()
            st.rerun()
        except Exception as exc:
            storage.clear()
            st.error(
                "Google sign-in could not be completed. The one-time sign-in link may have expired; "
                f"please try again. Details: {_safe_auth_error(exc)}"
            )
        return

    # Existing email-confirmation links were initiated in the same Streamlit session.
    client = st.session_state.get("auth_client")
    if client is None:
        st.error("The email confirmation session is unavailable. Please request a new confirmation link.")
        return
    try:
        result = client.auth.exchange_code_for_session(code)
        if not result.user or not result.session:
            raise RuntimeError("Supabase did not return an authenticated session.")
        st.session_state.auth_user = {
            "id": str(result.user.id),
            "email": result.user.email,
            "access_token": result.session.access_token,
            "refresh_token": result.session.refresh_token,
        }
        st.query_params.clear()
        st.rerun()
    except Exception as exc:
        st.error(f"Email verification callback failed — {_safe_auth_error(exc)}")


def require_authentication() -> bool:
    _handle_auth_callback()
    if current_user():
        try:
            client = st.session_state.auth_client
            verified = client.auth.get_user(current_user()["access_token"])
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

    st.title("DocuMind")
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
            flow_id = secrets.token_hex(24)
            storage = RedisAuthStorage(flow_id)
            client = _client(storage=storage)
            redirect_to = f"{APP_URL.rstrip('/')}?{urlencode({'oauth_flow': flow_id})}"
            oauth = client.auth.sign_in_with_oauth({
                "provider": "google",
                "options": {"redirect_to": redirect_to},
            })
            if not oauth.url:
                raise RuntimeError("Supabase did not return a Google authorization URL.")
            st.session_state["google_oauth_url"] = oauth.url
        except Exception as exc:
            st.error(f"Google sign-in could not be started — {_safe_auth_error(exc)}")
    if st.session_state.get("google_oauth_url"):
        oauth_url = html.escape(st.session_state["google_oauth_url"], quote=True)
        st.markdown(
            f'<a href="{oauth_url}" target="_self" rel="noopener noreferrer">Continue to Google</a>',
            unsafe_allow_html=True,
        )
    st.caption("Your documents remain isolated to your account.")
    return False
