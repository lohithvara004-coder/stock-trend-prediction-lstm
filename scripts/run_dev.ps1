# Launch Streamlit Application for Local Development (PowerShell)
Write-Host "Starting TrendPulse-AI Development Server..." -ForegroundColor Green
streamlit run app.py --server.port 8501 --theme.base "dark"
