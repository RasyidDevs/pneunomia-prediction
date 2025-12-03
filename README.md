# Pneumonia Prediction App 🫁

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?style=for-the-badge&logo=docker&logoColor=white)

A web-based application designed to detect **Pneumonia** from chest X-Ray images using Deep Learning techniques. This project utilizes transfer learning with **ResNet50** and **MobileNet** architectures to provide accurate classifications. It features an interactive **Streamlit** dashboard and is fully containerized with **Docker** for easy deployment.

## Project Structure

```
├── artifacts/
  ├── models     # Save ResNet50 and MobileNet model
  ├── metrics.json # Saving metrics from ResNet50 and MobileNet model
├── logs/            # Logging 
├── notebooks/       # Jupyter Notebooks for experimentation and training
├── pages/
  ├──1_Model Information.py    # Additional Pages for streamlit
├── src/             # Core source code for data processing and modeling
  ├── image-processing.py # Module for processing image
  ├── model.py       #Module for model inference
├── utils/           # for utility
  ├── config.py      # Storing Variable used for application
  ├── logging.py     # For logging system
  ├── styling.py     # For load css
├── Home.py          # Main entry point for the Streamlit application
├── app.py           # Alternative backend or entry script
├── Dockerfile       # Docker image configuration
├── docker-compose.yml  # Docker orchestration configuration
└── requirements.txt    # Python dependency list
```

## Installation
1. Clone Repository
```
git clone https://github.com/RasyidDevs/pneunomia-prediction.git
cd pneunomia-prediction
```
2. Create enviroment
```
python3 -m venv venv
source venv/bin/activate 
```

3. Install Depedencies
```
pip install -r requirements.txt
```

4. Run FastAPI
```
uvicorn app:app --host 0.0.0.0 --port 8003 --reload
```
5. Run Streamlit APP
```
streamlit run Home.py
```

