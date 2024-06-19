test: neural_net.c
		gcc neural_net.c -o neural_net -lm
		./neural_net

train-cpu: ./core/cpu/train.c
		gcc -o core/cpu/build/train core/cpu/train.c -lm
		./core/cpu/build/train

train-gpu: ./core/gpu/train.cu
		nvcc -o core/gpu/build/train core/gpu/train.cu -fmad=false
		./core/gpu/build/train

infer-gpu: ./core/gpu/infer.cu
		nvcc -o core/gpu/build/infer core/gpu/infer.cu -fmad=false
		./core/gpu/build/infer

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
