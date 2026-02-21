#!/bin/bash
# Risk Dashboard – Doppelklick zum Starten

cd "/Users/arminrost/Documents/Python_Projects/Testumgebung für Claude Code/risk_dashboard" || exit 1
source .venv/bin/activate
streamlit run app.py

echo ""
echo "Dashboard beendet. Drücke Enter zum Schliessen."
read -r
