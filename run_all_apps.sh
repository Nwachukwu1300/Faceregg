#!/bin/bash

# Run all Streamlit apps for the Facial Recognition System

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting all Facial Recognition System apps...${NC}"

# Start the main app on port 8501 (default)
echo -e "${YELLOW}Starting main app (app.py) on port 8501...${NC}"
streamlit run app.py &
MAIN_PID=$!

# Wait a moment before starting the other apps
sleep 2

# Start the threshold analysis app on port 8503
echo -e "${YELLOW}Starting Threshold Analysis app (view_analysis.py) on port 8503...${NC}"
streamlit run view_analysis.py --server.port 8503 &
ANALYSIS_PID=$!

# Start the fine-tuning app on port 8504
echo -e "${YELLOW}Starting Threshold Fine-Tuning app (fine_tune_threshold.py) on port 8504...${NC}"
streamlit run fine_tune_threshold.py --server.port 8504 &
FINETUNE_PID=$!

echo -e "${GREEN}All apps started successfully!${NC}"
echo -e "Main App: http://localhost:8501"
echo -e "Threshold Analysis: http://localhost:8503"
echo -e "Threshold Fine-Tuning: http://localhost:8504"
echo -e "${YELLOW}Press Ctrl+C to stop all apps${NC}"

# Wait for user to press Ctrl+C
trap "echo -e '${GREEN}Stopping all apps...${NC}'; kill $MAIN_PID $ANALYSIS_PID $FINETUNE_PID; exit" INT

# Keep the script running
wait 