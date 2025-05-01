# FAQ
## Setup
Clone the repository:
```
git clone https://github.com/NiazTahi/FAQ.git
cd FAQ
```
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Set up your environment variables:
```
touch .env
```
3. Set your API keys:
```
OPENAI_API_KEY = your-openai-api-key
HF_TOKEN = your-huggingface-api-key
```
## Usage
1. Set up Backend:
```
python backend.py
```
2. Run Frontend:
```
streamlit run frontend.py
```
