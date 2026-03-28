# Agent - AI Financial Analysis Platform

---

# Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

## 2. Environment Setup (Python 3.10+)

It is recommended to use a virtual environment:

Mac/Linux
```bash
python -m venv venv
source venv/bin/activate
```
Windows
```bash
python -m venv venv
.\venv\Scripts\activate
```
## 3. Install Dependencies

```bash
pip install -r requirements.txt
```
## 4. Configure Environment Variables

Add your API keys in the root directory .env file:

```bash
OPENAI_API_KEY=your_openai_api_key
FINNHUB_API_KEY=your_finnhub_api_key
```
## 5. Start the program
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000 
```
