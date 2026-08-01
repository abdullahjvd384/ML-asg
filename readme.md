# Freight Rate Prediction Challenge

See `freight-rate-ml-assessment.pdf` for the assessment instructions.

## What to do

1. Train and validate your model using `train-test.csv`.
2. Predict every load in `validation.csv`. Each load has a unique `load_id`.
3. Fill the matching `predicted_rate` values in `validation-predictions-template.csv` and save it as `validation_predictions.csv`.
4. Predict every row in `december-chart-inputs.csv` by filling its `predicted_rate` column.
5. Install the scorer requirements and run:

```bash
python -m pip install -r requirements.txt
python train_and_predict.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

The scorer validates both files and creates `scorer_results/candidate_december.png`.

## Included Baseline Pipeline

- `train_and_predict.py` trains a baseline `RandomForestRegressor`.
- It uses a temporal split by date (first 80% dates train, last 20% dates validation).
- It writes:
	- `validation_predictions.csv`
	- updated `december-chart-inputs.csv` with `predicted_rate`
	- `model_metrics.json` (MAE, RMSE, MAPE on the temporal holdout)

## Submit

- GitHub repository containing your code, dependencies, and run instructions
- `validation_predictions.csv`
- PDF or DOCX report containing your validation, data split approach and `candidate_december.png`
- 2-3 minute Loom link
