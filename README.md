# Flucture AI

Flucture AI is a GenAI-enhanced posture intelligence system for desk workers, students, and corporate users who spend long hours sitting in front of a screen.

It combines:
- real-time posture tracking with OpenCV and MediaPipe
- seated-friendly posture evaluation
- AI-generated posture explanation and correction reports
- downloadable PDF reports for later review
- background alerts through browser title, favicon, and notifications.

The main idea is simple: keep Flucture AI running in one tab while you work in another. When your posture starts slipping, the app can alert you. When you want a deeper review, it generates a clear report explaining what is wrong, what issues it may cause, and how to improve.

## Features

- Live webcam posture monitoring
- Seated posture support for desk-job use
- Real-time posture quality feedback
- Browser-based attention alerts
- OpenAI-powered report generation
- Fallback rule-based report generation if AI is unavailable
- Human-readable report summary in the UI
- Downloadable PDF posture report
- Saved JSON and PDF reports for session history

## Tech Stack

- Python
- Flask
- OpenCV
- MediaPipe
- NumPy
- OpenAI API
- ReportLab
- HTML, CSS, JavaScript

## Project Structure

```text
Flucture/
├── Freemium_app.py
├── main3.py
├── analytics/
│   ├── session_analyzer.py
│   └── severity_rules.py
├── llm/
│   ├── prompts.py
│   └── retriever.py
├── schemas/
│   └── report_schema.py
├── utils/
│   ├── pdf_report.py
│   └── storage.py
├── data/
│   └── knowledge_base/
├── templates/
│   └── index.html
├── static/
│   ├── app.js
│   └── styles.css
├── requirements.txt
└── PROJECT_PITCH.md
```

## How It Works

1. The app opens the webcam and continuously reads posture landmarks.
2. It estimates posture quality using shoulder, neck, head, and torso alignment.
3. When posture is poor, the interface can signal this visually and through browser notifications.
4. A report can then be generated from live metrics and logged session evidence.
5. The report explains:
   - what is wrong
   - possible consequences
   - how to improve
   - remedies, stretches, and ergonomic corrections
6. The result is saved as both JSON and PDF.

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sandy22Coder/Flucture.git
cd Flucture
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Current PowerShell session:

```powershell
$env:OPENAI_API_KEY="sk-your-key"
```

Permanent on Windows:

```powershell
setx OPENAI_API_KEY "sk-your-key"
```

If the key is not available, the app still works using the fallback report generator.

## Run the App

```bash
python Freemium_app.py
```

Then open:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

## Usage Flow

1. Start tracking
2. Sit naturally at your desk
3. Keep the tab open while working
4. Enable alerts if you want background notifications
5. Generate a report when you want a posture review
6. Download the PDF report

## Report Output

Each report includes:
- overall assessment
- risk level
- what is wrong
- possible consequences
- improvement plan
- stretches
- strengthening exercises
- daily habits
- ergonomic corrections
- progress score

Reports are saved in the local `reports/` folder as:
- `session_report_<timestamp>.json`
- `session_report_<timestamp>.pdf`

## Notes

- Best results come when your head, shoulders, and upper torso are visible.
- The app has been adapted to work better for seated users.
- Browser notifications require permission from the user.
- Generated files and local environment folders are intentionally excluded from git.

## Rubric Alignment

This project is suitable for a Generative AI course because it goes beyond simple prompting:
- combines computer vision with GenAI
- uses structured report generation
- uses a local posture knowledge base
- supports grounded, user-friendly explanations
- includes a polished interactive UI and downloadable outputs

## Author

Built for a Generative AI course project using posture tracking, grounded LLM reporting, and interactive UX design.
