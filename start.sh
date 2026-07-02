#!/bin/bash
python bot.py &
exec streamlit run dashboard.py --server.port $PORT --server.headless true
