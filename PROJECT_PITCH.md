# Flucture AI Project Pitch

## Title

Flucture AI: A Posture Intelligence System Using Computer Vision, Retrieval, and Structured LLM Reporting

## Problem Definition

Most posture trackers only tell users that their posture is wrong in the moment. They do not explain what the issue means, what health risks may follow, or how to correct it in a personalized and structured way.

Flucture AI addresses this gap by combining real-time posture detection with a grounded Generative AI reporting pipeline. The system converts session logs into a detailed user report that explains:
- what is wrong
- what risks may arise if the posture continues
- how to improve
- what remedies, stretches, and habits to follow

## Novelty

- Uses computer vision data as structured evidence for LLM reasoning
- Retrieves only relevant posture knowledge instead of relying on generic generation
- Produces strict JSON reports for reliable UI rendering
- Tracks progress across sessions instead of giving one-off advice

## Technical Complexity

The project combines four technical layers:

1. Real-time pose tracking with MediaPipe and OpenCV
2. Session analytics over logged posture deviations
3. Retrieval over a curated posture knowledge base
4. Structured LLM generation for grounded posture intervention reports

This is stronger than basic prompting because the LLM is not used as a standalone chatbot. It operates on structured evidence and retrieved knowledge.

## System Flow

1. Webcam feed is processed in real time.
2. Posture metrics and reasons are logged into CSV.
3. Session analytics convert the raw logs into issue frequencies, severity, and confidence.
4. A retriever selects issue-specific posture knowledge.
5. The LLM generates a structured report.
6. The UI presents the report and stores it for later comparison.

## Output Sections

The generated report contains:
- what is wrong
- possible consequences
- improvement plan
- remedies
- ergonomic corrections
- red flags
- progress score compared to the previous session

## Rubric Mapping

### Problem Definition and Scope
- Clear real-world problem
- Personalized intervention rather than basic error detection
- Creative use of LLMs over posture logs

### Technical Complexity
- CV + analytics + retrieval + structured generation
- LLM grounded by local knowledge base
- Report persistence and progress comparison

### Implementation Quality
- Modular architecture
- Strict report schema
- Reusable knowledge base
- Clean separation between tracking, analytics, retrieval, and report generation

### Project Demonstration
- Live tracking
- Session report generation
- Structured UI cards
- Saved session reports for comparison

## Demo Script

1. Start live posture tracking
2. Show real-time metrics changing
3. Let the system collect a short posture session
4. Click `Generate AI Report`
5. Show the issue breakdown, risks, improvements, and remedies
6. Show that the report is saved and can be compared with the previous session
