import streamlit as st
import requests
from requests_html import HTMLSession
import time

API_URL = "http://localhost:8000"

st.title("WandB Simulation with Real-Time Updates")

# Sidebar to create a new experiment
st.sidebar.header("Create New Experiment")
experiment_name = st.sidebar.text_input("Experiment Name")
experiment_description = st.sidebar.text_input("Experiment Description")
if st.sidebar.button("Create Experiment"):
    response = requests.post(f"{API_URL}/experiments/", json={
        "name": experiment_name,
        "description": experiment_description
    })
    if response.status_code == 200:
        st.sidebar.success("Experiment created successfully")
        st.sidebar.text(f"Experiment ID: {response.json()['id']}")
    else:
        st.sidebar.error("Error creating experiment.")

if st.sidebar.button("Clear Experiments"):
    response = requests.delete(f"{API_URL}/experiments/")
    if response.status_code == 200:
        st.sidebar.success("Experiments cleared successfully")
    else:
        st.sidebar.error("Error clearing experiments.")

def fetch_experiments():
    response = requests.get(f"{API_URL}/experiments/")
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching experiments.")
        return []

def fetch_and_display_experiments(container):
    experiments = fetch_experiments()
    for experiment in experiments:
        container.empty()
        
        with container:
            loss, acc = st.columns(2)   
        
            with loss:
                st.subheader("Metrics: Loss")
                st.write(f"Experiment: {experiment['name']}")
                for metric_key, metric_values in experiment["metrics"].items():
                    if "loss" in metric_key:
                        st.line_chart(metric_values)
            
            with acc:
                st.subheader("Metrics: Accuracy")
                st.write(f"Experiment: {experiment['name']}")
                for metric_key, metric_values in experiment["metrics"].items():
                    if "accuracy" in metric_key:
                        st.line_chart(metric_values)

session = HTMLSession()
events_url = f"{API_URL}/events/"

def sse_fetch_and_display(container):
    with session.get(events_url, stream=True) as response:
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if "update" in decoded_line:
                    fetch_and_display_experiments(container)
                    break

st.header("Experiments")
container = st.empty()
fetch_and_display_experiments(container)

while True:
    sse_fetch_and_display(container)
    time.sleep(1)
