# Put Kar Drishti online on Streamlit Community Cloud (free)

You need two free accounts: GitHub (stores the code) and Streamlit Community Cloud (runs it).

## 1. Put the code on GitHub (about 5 minutes)
1. Create a free account at https://github.com/signup (skip if you have one).
2. Go to https://github.com/new
   - Repository name: `kar-drishti`
   - Choose **Public**
   - Do not add a README (leave the boxes unticked)
   - Click **Create repository**
3. On the next page click the link **"uploading an existing file"**.
4. Open this folder (`streamlit_cloud`) in File Explorer, press **Ctrl+A** to select everything inside it,
   and drag it into the GitHub page. Check that the `.streamlit` folder is included
   (it holds the colours).
5. Wait until all files are listed, then click **Commit changes**.

## 2. Start the app (about 5 minutes)
1. Go to https://share.streamlit.io and click **Continue with GitHub**, then allow access.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Fill in:
   - Repository: `<your-github-name>/kar-drishti`
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: for example `kar-drishti` (gives `https://kar-drishti.streamlit.app`)
   - Advanced settings → Python version: **3.12**
4. Click **Deploy**. The first start takes 3–5 minutes while it installs the packages.

## Good to know
- Everything works except free-form AI questions and AI-written explanations (the free host has
  about 1–2.7 GB of memory, too little for the AI model). Show those live on your laptop.
- A free app sleeps after a few days without visitors; opening the link wakes it in about a minute.
- To update the app later, upload changed files to the same GitHub repository; it restarts by itself.
- Never upload real taxpayer data to the online version.
