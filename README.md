# Human Activity Recognition with RNN (LSTM)

## Run
```
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python har_rnn.py              # add --compare for GRU + SimpleRNN, --epochs 10 for a quick run
```
Needs Python 3.9-3.12 and internet on first run (downloads the ~60 MB UCI HAR dataset).
Without internet it falls back to synthetic data (the deck says so).

## Output (./outputs)
Graphs (PNG), `results.json`, trained model `har_lstm.keras`, and `HAR_RNN_Presentation.pptx`
(built from your real results, 9-10 slides).
"# human_acitivity_recognition" 
