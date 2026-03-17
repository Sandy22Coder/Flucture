FluctureMain

Flucture Fit is a computer-vision workout assistant that:

Tracks posture and counts reps in real time (Freemium).

Generates AI workout analysis and guided plans (Premium).

Deployment not required for evaluation. This repo is fully runnable locally.

🧱 Repository Structure
FluctureMain/
├─ templates/                # HTML/JS/CSS used by the Flask apps
│   └─ ...                   # (no API keys inside)
├─ Freemium_app.py           # Freemium web app (UI server)
├─ Premium_app.py            # Premium web app (requires OpenAI API key)
├─ main3.py                  # Freemium CV/pose engine (runs alongside Freemium_app.py)
├─ requirements.txt          # Python dependencies
└─ README.md                 # You are here


Freemium = posture correction + rep counting (no AI API needed).
Runs as two processes:

main3.py (camera + MediaPipe/OpenCV engine)

Freemium_app.py (Flask UI that connects to the engine)

Premium = guided workouts + AI session summaries.
Runs as a single app Premium_app.py and requires your own OpenAI API key.

🛠 Prerequisites

Python 3.9 – 3.12 (recommended 3.10/3.11)

A webcam

OS: Windows 10/11, macOS, or Linux

(Optional) Virtual environment recommended

If you have issues on Apple Silicon, install Python via pyenv or conda and ensure a recent pip.

⚙️ Setup (once)
1) Clone & enter the project
git clone https://github.com/<your-username>/FluctureMain.git
cd FluctureMain

2) Create & activate a virtual environment

Windows (CMD/PowerShell)

python -m venv venv
venv\Scripts\activate


macOS / Linux

python3 -m venv venv
source venv/bin/activate


You should see (venv) before your terminal prompt.

3) Install dependencies
pip install -r requirements.txt

🚀 Running the Freemium App (No API key needed)

Freemium is split into engine and UI. Keep both terminals open.

Terminal A – start the engine
# (venv) inside the project folder
python main3.py


You should see logs indicating the webcam opened and the pose pipeline is running.

Terminal B – start the UI

Open a new terminal window, activate the same venv again, then:

# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Start the Freemium web app
python Freemium_app.py


Open your browser at:

http://127.0.0.1:5000


You’ll get real-time posture feedback and rep counts.
HTML is served from the templates/ folder.

💎 Running the Premium App (Requires OpenAI)
1) Set your OpenAI API key

Windows (PowerShell)

setx OPENAI_API_KEY "sk-<your-key>"
# Close & reopen the terminal to load the variable


macOS / Linux (temporary for current shell)

export OPENAI_API_KEY="sk-<your-key>"


Alternatively, create a .env file and load it in code (not included here for simplicity).
Do not commit your API key or .env to GitHub.

2) Start the Premium app
# (venv) active
python Premium_app.py


Open your browser at:

http://127.0.0.1:5000


You’ll see guided workouts and AI session summaries. The app reads your key from OPENAI_API_KEY.

🔧 Common Options & Tips

Choose another camera: if your default webcam is busy, edit the camera index in the code (e.g., cv2.VideoCapture(1)).

Port in use: change the Flask port, e.g. app.run(port=5050).

Performance: close extra browser tabs and video apps; ensure only one app uses the camera at a time.

🧰 Troubleshooting

ModuleNotFoundError: Verify the venv is active and run pip install -r requirements.txt again.

Camera not found / black screen: Ensure permissions are granted; try camera index 1 or 2.

MediaPipe errors: Upgrade pip and reinstall:

python -m pip install --upgrade pip
pip install --upgrade mediapipe opencv-python


OpenAI key error (Premium): Make sure OPENAI_API_KEY is set in the same shell where you run Premium_app.py.

🔒 Security Notes

The repository does not contain any API keys.

Keep your OPENAI_API_KEY private. Do not hardcode keys or commit .env files.

📋 What each app demonstrates

Freemium

Real-time posture correction (green/red body lines)

Rep detection & counters

Session basics (no external AI calls)

Premium

All Freemium features plus:

AI workout summaries (sets, common mistakes, cues)

Guided workout modes

Export/report options (as implemented)
