test: neural_net.c
		gcc neural_net.c -o neural_net -lm
		./neural_net

train-cpu: ./core/cpu/main.c
		gcc -o core/cpu/build/main core/cpu/main.c -lm
		./core/cpu/build/main

install:
		sudo apt-get install libconfig-dev
		rm -r -f .venv || true
		python3 -m venv .venv
		. .venv/bin/activate
		.venv/bin/pip3 install -r requirements.txt

run-api:
		. .venv/bin/activate
		.venv/bin/uvicorn api.main:app

run-gui:
		. .venv/bin/activate
		.venv/bin/streamlit run gui/app.py
