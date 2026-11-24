# Put Pragya online on Streamlit Community Cloud (free), with the Groq AI model

You need: a GitHub account (code), a Streamlit Community Cloud account (runs it) and a free Groq API key (AI).
Your repository is already set up: https://github.com/Varshini-arsh/kar-drishti

## 1. Upload the new version to GitHub
Double-click **`publish_to_github.bat`** in this folder. It commits everything and pushes it.
(If Git asks you to sign in, choose "Sign in with your browser".)
If you have never deployed: create an EMPTY public repository named `kar-drishti` at https://github.com/new first.

## 2. Get a free Groq API key
1. Go to https://console.groq.com and sign up.
2. **API Keys** > **Create API key**. Copy it (starts with `gsk_`). Keep it private: never put it in a file in this folder.

## 3. Create the app (first time) or open it (already deployed)
1. Go to https://share.streamlit.io and sign in with GitHub.
2. First time: **Create app** > **Deploy a public app from GitHub**:
   - Repository `Varshini-arsh/kar-drishti`, branch `main`, main file `app.py`
   - App URL: for example `pragya-demo` (gives `https://pragya-demo.streamlit.app`)
   - **Advanced settings** > Python version **3.12**
3. In **Advanced settings > Secrets** (or later: your app > **⋮** > **Settings** > **Secrets**) paste:
   ```
   GROQ_API_KEY = "gsk_your_key_here"
   ```
   and click **Save**.
4. Click **Deploy** (first start takes 3–5 minutes). If the app was already running, it restarts with the key.

## 4. Check it
- The sidebar shows the Groq model with a green dot (for example `llama-3.3-70b-versatile`).
- The top line says "Online demo · synthetic data only · ... AI wording by Groq (cloud)".
- **+ Add data** > tick datasets > **Add**, then ask a suggested question.

## Good to know
- Without the secret the app still works; AI wording is just off.
- Groq's free tier has rate limits. If many people use it at once, some answers will say to wait a minute.
- A free app sleeps after some days without visitors; opening the link wakes it in about a minute.
- Never put real taxpayer data in the online version.
