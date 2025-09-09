import os
import warnings
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
import argparse

warnings.filterwarnings("ignore")

# ydata-synthetic imports (may be unavailable for some classes)
try:
	from ydata_synthetic.synthesizers.regular.ctgan import CTGANSynthesizer as YDataCTGAN
except Exception:
	YDataCTGAN = None

try:
	from ydata_synthetic.synthesizers.privacy.pategan import PATEGANSynthesizer
except Exception:
	PATEGANSynthesizer = None

try:
	from ydata_synthetic.synthesizers.privacy.dp_ctgan import DPCTGANSynthesizer
except Exception:
	DPCTGANSynthesizer = None

# Fallback CTGAN implementations
try:
	from ctgan import CTGAN as SDVCTGAN
except Exception:
	SDVCTGAN = None

try:
	from sdv.tabular import CTGAN as SDVTabularCTGAN
except Exception:
	SDVTabularCTGAN = None

# SDV Single-Table CTGAN
try:
	from sdv.metadata import SingleTableMetadata
	from sdv.single_table import CTGANSynthesizer as SDVSingleTableCTGAN
except Exception:
	SingleTableMetadata = None
	SDVSingleTableCTGAN = None

RANDOM_STATE = 42
TARGET_COL = "cardio"
ID_COLS = ["id"]
CSV_PATH = os.path.join(os.path.dirname(__file__), "cardio_train_dataset.csv")


def load_dataset(path: str) -> pd.DataFrame:
	sep = ";" if path.endswith(".csv") else ","
	return pd.read_csv(path, sep=sep)


def split_train_test(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_df, test_df = train_test_split(
		df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[target_col]
	)
	return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> ColumnTransformer:
	features = [c for c in df.columns if c not in drop_cols + [target_col]]
	categorical_cols = []
	numeric_cols = []
	for col in features:
		if pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 10:
			categorical_cols.append(col)
		else:
			numeric_cols.append(col)

	numeric_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="median")),
		("scaler", StandardScaler()),
	])
	categorical_transformer = Pipeline(steps=[
		("imputer", SimpleImputer(strategy="most_frequent")),
		("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
	])

	preprocessor = ColumnTransformer(
		transformers=[
			("num", numeric_transformer, numeric_cols),
			("cat", categorical_transformer, categorical_cols),
		]
	)
	return preprocessor


def train_eval_classifier(train_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[float, float]:
	preprocessor = build_preprocessor(train_df, target_col, drop_cols)
	clf = Pipeline(steps=[
		("pre", preprocessor),
		("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
	])
	X_train = train_df.drop(columns=drop_cols + [target_col], errors="ignore")
	y_train = train_df[target_col].astype(int)
	X_test = test_df.drop(columns=drop_cols + [target_col])
	y_test = test_df[target_col].astype(int)
	clf.fit(X_train, y_train)
	probs = clf.predict_proba(X_test)[:, 1]
	return roc_auc_score(y_test, probs), average_precision_score(y_test, probs)


def detect_discrete_columns(df: pd.DataFrame, target_col: str) -> List[str]:
	cols = []
	for c in df.columns:
		if c in ID_COLS:
			continue
		if c == target_col:
			cols.append(c)
			continue
		if pd.api.types.is_integer_dtype(df[c]) and df[c].nunique() <= 20:
			cols.append(c)
	return cols


def train_ctgan(train_df: pd.DataFrame, target_col: str, epochs: int) -> pd.DataFrame:
	train_data = train_df.drop(columns=ID_COLS, errors="ignore").copy()
	discrete_columns = detect_discrete_columns(train_data, target_col)

	# ydata CTGAN
	if YDataCTGAN is not None:
		model = YDataCTGAN(
			epochs=epochs,
			batch_size=256,
			embedding_dim=128,
			generator_dim=(256, 256),
			discriminator_dim=(256, 256),
			discrete_columns=discrete_columns,
			ratios=None,
			random_state=RANDOM_STATE,
		)
		model.fit(train_data)
		return model.sample(len(train_df))

	# ctgan package
	if SDVCTGAN is not None:
		try:
			model = SDVCTGAN(
				epochs=epochs,
				batch_size=256,
				embedding_dim=128,
				generator_dim=(256, 256),
				discriminator_dim=(256, 256),
			)
			model.fit(train_data, discrete_columns=discrete_columns)
			return model.sample(len(train_df))
		except Exception:
			pass

	# sdv.tabular CTGAN
	if SDVTabularCTGAN is not None:
		try:
			model = SDVTabularCTGAN(cuda=False, epochs=epochs, verbose=True)
			model.fit(train_data)
			return model.sample(num_rows=len(train_df))
		except Exception:
			pass

	# SDV Single-Table CTGAN
	if SingleTableMetadata is not None and SDVSingleTableCTGAN is not None:
		metadata = SingleTableMetadata()
		metadata.detect_from_dataframe(data=train_data)
		try:
			model = SDVSingleTableCTGAN(metadata, epochs=epochs, verbose=True)
			model.fit(train_data)
			return model.sample(num_rows=len(train_df))
		except Exception:
			pass

	raise RuntimeError("No CTGAN implementation available.")


def train_pategan(train_df: pd.DataFrame, target_col: str, epochs: int) -> pd.DataFrame:
	if PATEGANSynthesizer is None:
		raise RuntimeError("PATE-GAN not available in installed ydata-synthetic.")
	train_data = train_df.drop(columns=ID_COLS, errors="ignore").copy()
	discrete_columns = detect_discrete_columns(train_data, target_col)
	model = PATEGANSynthesizer(
		epochs=epochs,
		batch_size=256,
		epsilon=2.0,
		delta=1e-5,
		momentum=0.5,
		discrete_columns=discrete_columns,
		random_state=RANDOM_STATE,
	)
	model.fit(train_data)
	return model.sample(len(train_df))


def train_dp_ctgan(train_df: pd.DataFrame, target_col: str, epochs: int) -> pd.DataFrame:
	if DPCTGANSynthesizer is None:
		raise RuntimeError("DP-CTGAN not available in installed ydata-synthetic.")
	train_data = train_df.drop(columns=ID_COLS, errors="ignore").copy()
	discrete_columns = detect_discrete_columns(train_data, target_col)
	model = DPCTGANSynthesizer(
		epochs=epochs,
		batch_size=256,
		epsilon=2.0,
		delta=1e-5,
		discrete_columns=discrete_columns,
		random_state=RANDOM_STATE,
	)
	model.fit(train_data)
	return model.sample(len(train_df))


def main():
	parser = argparse.ArgumentParser(description="Train CTGAN, PATE-GAN, DP-CTGAN and evaluate AUC metrics")
	parser.add_argument("--ctgan-epochs", type=int, default=50)
	parser.add_argument("--pategan-epochs", type=int, default=50)
	parser.add_argument("--dpctgan-epochs", type=int, default=50)
	args = parser.parse_args()

	print("Loading dataset ...")
	df = load_dataset(CSV_PATH)
	assert TARGET_COL in df.columns

	drop_cols = [c for c in ID_COLS if c in df.columns]
	train_df, test_df = split_train_test(df, TARGET_COL)

	results: Dict[str, Tuple[float, float]] = {}

	print("Training baseline classifier (real->real) ...")
	base_auc_roc, base_auc_pr = train_eval_classifier(train_df, test_df, TARGET_COL, drop_cols)
	results["baseline_real"] = (base_auc_roc, base_auc_pr)
	print(f"Baseline AUC-ROC: {base_auc_roc:.4f} | AUC-PR: {base_auc_pr:.4f}")

	# CTGAN
	try:
		print("Training CTGAN ...")
		ctgan_samples = train_ctgan(train_df, TARGET_COL, epochs=args.ctgan_epochs)
		ctgan_samples[TARGET_COL] = ctgan_samples[TARGET_COL].round().clip(0, 1).astype(int)
		ct_auc_roc, ct_auc_pr = train_eval_classifier(ctgan_samples, test_df, TARGET_COL, drop_cols)
		results["ctgan"] = (ct_auc_roc, ct_auc_pr)
		print(f"CTGAN AUC-ROC: {ct_auc_roc:.4f} | AUC-PR: {ct_auc_pr:.4f}")
	except Exception as e:
		print(f"CTGAN failed: {e}")

	# PATE-GAN
	try:
		print("Training PATE-GAN ...")
		pate_samples = train_pategan(train_df, TARGET_COL, epochs=args.pategan_epochs)
		pate_samples[TARGET_COL] = pate_samples[TARGET_COL].round().clip(0, 1).astype(int)
		pg_auc_roc, pg_auc_pr = train_eval_classifier(pate_samples, test_df, TARGET_COL, drop_cols)
		results["pategan"] = (pg_auc_roc, pg_auc_pr)
		print(f"PATE-GAN AUC-ROC: {pg_auc_roc:.4f} | AUC-PR: {pg_auc_pr:.4f}")
	except Exception as e:
		print(f"PATE-GAN failed: {e}")

	# DP-CTGAN
	try:
		print("Training DP-CTGAN ...")
		dpctgan_samples = train_dp_ctgan(train_df, TARGET_COL, epochs=args.dpctgan_epochs)
		dpctgan_samples[TARGET_COL] = dpctgan_samples[TARGET_COL].round().clip(0, 1).astype(int)
		dpct_auc_roc, dpct_auc_pr = train_eval_classifier(dpctgan_samples, test_df, TARGET_COL, drop_cols)
		results["dp_ctgan"] = (dpct_auc_roc, dpct_auc_pr)
		print(f"DP-CTGAN AUC-ROC: {dpct_auc_roc:.4f} | AUC-PR: {dpct_auc_pr:.4f}")
	except Exception as e:
		print(f"DP-CTGAN failed: {e}")

	print("\nResults:")
	for k, (a1, a2) in results.items():
		print(f"{k:>15}: AUC-ROC={a1:.4f} | AUC-PR={a2:.4f}")

	out_path = os.path.join(os.path.dirname(__file__), "gan_eval_results.csv")
	rows = [
		{"model": k, "auc_roc": v[0], "auc_pr": v[1]}
		for k, v in results.items()
	]
	pd.DataFrame(rows).to_csv(out_path, index=False)
	print(f"Saved results to {out_path}")


if __name__ == "__main__":
	main()
