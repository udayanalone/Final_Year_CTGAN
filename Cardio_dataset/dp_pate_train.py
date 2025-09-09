import os
import math
import random
import argparse
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

CSV_PATH = os.path.join(os.path.dirname(__file__), "cardio_train_dataset.csv")
TARGET_COL = "cardio"
ID_COLS = ["id"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset(path: str) -> pd.DataFrame:
	sep = ";" if path.endswith(".csv") else ","
	df = pd.read_csv(path, sep=sep)
	return df


def split_train_test(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	train_df, test_df = train_test_split(
		df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[target_col]
	)
	return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_preprocessor(df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[Pipeline, List[str], List[str]]:
	features = [c for c in df.columns if c not in drop_cols + [target_col]]
	categorical_cols: List[str] = []
	numeric_cols: List[str] = []
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

	pipe = Pipeline(steps=[("pre", preprocessor)])
	return pipe, numeric_cols, categorical_cols


def transform_features(pipe: Pipeline, df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
	X = df.drop(columns=drop_cols + [target_col], errors="ignore")
	y = df[target_col].astype(int).values
	X_t = pipe.fit_transform(X)
	return X_t.astype(np.float32), y.astype(np.int64)


def transform_features_infer(pipe: Pipeline, df: pd.DataFrame, target_col: str, drop_cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
	X = df.drop(columns=drop_cols + [target_col], errors="ignore")
	y = df[target_col].astype(int).values
	X_t = pipe.transform(X)
	return X_t.astype(np.float32), y.astype(np.int64)


class MLPGenerator(nn.Module):
	def __init__(self, noise_dim: int, cond_dim: int, out_dim: int, hidden: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(noise_dim + cond_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, out_dim),
		)
	def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([z, c], dim=1))


class MLPDiscriminator(nn.Module):
	def __init__(self, in_dim: int, cond_dim: int, hidden: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim + cond_dim, hidden), nn.LeakyReLU(0.2),
			nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
			nn.Linear(hidden, 1),
		)
	def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		return self.net(torch.cat([x, c], dim=1))


def train_classifier_auc(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
	clf = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=None, random_state=RANDOM_STATE)
	clf.fit(X_train, y_train)
	probs = clf.predict_proba(X_test)[:, 1]
	return roc_auc_score(y_test, probs), average_precision_score(y_test, probs)


def sample_labels_like(y: np.ndarray, n: int) -> np.ndarray:
	# Sample labels preserving class prior
	p = y.mean()
	return (np.random.rand(n) < p).astype(np.int64)


def train_dp_gan(
		X_train: np.ndarray,
		y_train: np.ndarray,
		noise_dim: int = 64,
		batch_size: int = 256,
		steps: int = 2000,
		lr: float = 1e-4,
		clip_norm: float = 1.0,
		sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Conditional DP-GAN: concatenate label as condition to G and D. Clip D gradients and add Gaussian noise.
	"""
	in_dim = X_train.shape[1]
	cond_dim = 1
	G = MLPGenerator(noise_dim, cond_dim, in_dim).to(DEVICE)
	D = MLPDiscriminator(in_dim, cond_dim).to(DEVICE)
	opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
	opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

	dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1))
	dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
	bce = nn.BCEWithLogitsLoss()

	for step in range(steps):
		for xb, yb in dl:
			xb = xb.to(DEVICE)
			yb = yb.to(DEVICE)
			# Train D with DP noise
			opt_D.zero_grad(set_to_none=True)
			z = torch.randn(xb.size(0), noise_dim, device=DEVICE)
			fake = G(z, yb).detach()
			logits_real = D(xb, yb)
			logits_fake = D(fake, yb)
			loss_D = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
			loss_D.backward()
			_ = torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=clip_norm)
			with torch.no_grad():
				for p in D.parameters():
					if p.grad is not None:
						p.grad.add_(torch.randn_like(p.grad) * (sigma * clip_norm))
			opt_D.step()

			# Train G
			opt_G.zero_grad(set_to_none=True)
			z = torch.randn(xb.size(0), noise_dim, device=DEVICE)
			gen = G(z, yb)
			logits = D(gen, yb)
			loss_G = bce(logits, torch.ones_like(logits))
			loss_G.backward()
			opt_G.step()
			break

	# Sample synthetic data with labels from prior
	n = X_train.shape[0]
	labels = torch.from_numpy(sample_labels_like(y_train, n).astype(np.float32)).unsqueeze(1).to(DEVICE)
	G.eval()
	out = []
	with torch.no_grad():
		for i in range(0, n, batch_size):
			bs = min(batch_size, n - i)
			z = torch.randn(bs, noise_dim, device=DEVICE)
			c = labels[i:i+bs]
			gen = G(z, c)
			out.append(gen.cpu().numpy())
	synth = np.vstack(out)
	return synth, labels.cpu().numpy().squeeze(1).astype(np.int64)


def train_pate_gan(
		X_train: np.ndarray,
		y_train: np.ndarray,
		noise_dim: int = 64,
		batch_size: int = 256,
		teachers: int = 5,
		teacher_steps: int = 800,
		student_steps: int = 1200,
		lr: float = 1e-4,
		sigma_votes: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
	"""
	Conditional PATE-GAN: teachers and generator conditioned on label.
	"""
	in_dim = X_train.shape[1]
	cond_dim = 1
	idx = np.arange(X_train.shape[0])
	np.random.shuffle(idx)
	shards = np.array_split(idx, teachers)

	teachers_D = [MLPDiscriminator(in_dim, cond_dim).to(DEVICE) for _ in range(teachers)]
	G = MLPGenerator(noise_dim, cond_dim, in_dim).to(DEVICE)
	opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
	opt_D = [torch.optim.Adam(t.parameters(), lr=lr, betas=(0.5, 0.9)) for t in teachers_D]
	bce = nn.BCEWithLogitsLoss()

	# Phase 1: Train teachers
	for t_step in range(teacher_steps):
		for k in range(teachers):
			sh = shards[k]
			if len(sh) < batch_size:
				continue
			batch_idx = np.random.choice(sh, batch_size, replace=False)
			xb = torch.from_numpy(X_train[batch_idx]).to(DEVICE)
			yb = torch.from_numpy(y_train[batch_idx].astype(np.float32)).unsqueeze(1).to(DEVICE)
			z = torch.randn(batch_size, noise_dim, device=DEVICE)
			fake = G(z, yb).detach()
			opt_D[k].zero_grad(set_to_none=True)
			logits_real = teachers_D[k](xb, yb)
			logits_fake = teachers_D[k](fake, yb)
			loss_D = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
			loss_D.backward()
			opt_D[k].step()

	# Phase 2: Train generator using noisy aggregated votes
	for s in range(student_steps):
		opt_G.zero_grad(set_to_none=True)
		z = torch.randn(batch_size, noise_dim, device=DEVICE)
		c = torch.from_numpy(sample_labels_like(y_train, batch_size).astype(np.float32)).unsqueeze(1).to(DEVICE)
		gen = G(z, c)
		with torch.no_grad():
			votes = []
			for k in range(teachers):
				logits = teachers_D[k](gen, c)
				pred = (torch.sigmoid(logits) > 0.5).float()
				votes.append(pred)
			votes = torch.stack(votes, dim=0)
			summed = votes.sum(dim=0).squeeze(1)
			noisy_counts = summed + torch.randn_like(summed) * sigma_votes
			targets = (noisy_counts > (teachers / 2.0)).float().unsqueeze(1)
		logits = teachers_D[0](gen, c)
		loss_G = bce(logits, targets)
		loss_G.backward()
		opt_G.step()

	# Sample synthetic
	n = X_train.shape[0]
	labels = torch.from_numpy(sample_labels_like(y_train, n).astype(np.float32)).unsqueeze(1).to(DEVICE)
	G.eval()
	out = []
	with torch.no_grad():
		for i in range(0, n, batch_size):
			bs = min(batch_size, n - i)
			z = torch.randn(bs, noise_dim, device=DEVICE)
			c = labels[i:i+bs]
			gen = G(z, c)
			out.append(gen.cpu().numpy())
	synth = np.vstack(out)
	return synth, labels.cpu().numpy().squeeze(1).astype(np.int64)


def main():
	parser = argparse.ArgumentParser(description="Train custom conditional DP-GAN and PATE-GAN on cardio data and evaluate AUCs")
	parser.add_argument("--dpgan-steps", type=int, default=1500)
	parser.add_argument("--pategan-teacher-steps", type=int, default=600)
	parser.add_argument("--pategan-student-steps", type=int, default=900)
	parser.add_argument("--batch-size", type=int, default=256)
	parser.add_argument("--noise-dim", type=int, default=64)
	parser.add_argument("--dp-clip", type=float, default=1.0)
	parser.add_argument("--dp-sigma", type=float, default=0.5)
	parser.add_argument("--pate-sigma", type=float, default=2.0)
	args = parser.parse_args()

	print("Loading dataset ...")
	df = load_dataset(CSV_PATH)
	assert TARGET_COL in df.columns
	drop_cols = [c for c in ID_COLS if c in df.columns]
	train_df, test_df = split_train_test(df, TARGET_COL)

	# Preprocess
	prep_pipe, _, _ = build_preprocessor(train_df, TARGET_COL, drop_cols)
	X_train, y_train = transform_features(prep_pipe, train_df, TARGET_COL, drop_cols)
	X_test, y_test = transform_features_infer(prep_pipe, test_df, TARGET_COL, drop_cols)

	# Baseline
	base_auc_roc, base_auc_pr = train_classifier_auc(X_train, y_train, X_test, y_test)
	print(f"Baseline AUC-ROC: {base_auc_roc:.4f} | AUC-PR: {base_auc_pr:.4f}")

	results: Dict[str, Tuple[float, float]] = {"baseline_real": (base_auc_roc, base_auc_pr)}

	# DP-GAN
	print("Training DP-GAN ...")
	synth_dp, y_dp = train_dp_gan(
		X_train,
		y_train,
		noise_dim=args.noise_dim,
		batch_size=args.batch_size,
		steps=args.dpgan_steps,
		clip_norm=args.dp_clip,
		sigma=args.dp_sigma,
	)
	dp_auc_roc, dp_auc_pr = train_classifier_auc(synth_dp, y_dp, X_test, y_test)
	results["dp_gan"] = (dp_auc_roc, dp_auc_pr)
	print(f"DP-GAN AUC-ROC: {dp_auc_roc:.4f} | AUC-PR: {dp_auc_pr:.4f}")

	# PATE-GAN
	print("Training PATE-GAN ...")
	synth_pate, y_pate = train_pate_gan(
		X_train,
		y_train,
		noise_dim=args.noise_dim,
		batch_size=args.batch_size,
		teachers=5,
		teacher_steps=args.pategan_teacher_steps,
		student_steps=args.pategan_student_steps,
		sigma_votes=args.pate_sigma,
	)
	pg_auc_roc, pg_auc_pr = train_classifier_auc(synth_pate, y_pate, X_test, y_test)
	results["pate_gan"] = (pg_auc_roc, pg_auc_pr)
	print(f"PATE-GAN AUC-ROC: {pg_auc_roc:.4f} | AUC-PR: {pg_auc_pr:.4f}")

	# Save
	rows = [{"model": k, "auc_roc": v[0], "auc_pr": v[1]} for k, v in results.items()]
	out_path = os.path.join(os.path.dirname(__file__), "custom_gan_eval_results.csv")
	pd.DataFrame(rows).to_csv(out_path, index=False)
	print(f"Saved results to {out_path}")


if __name__ == "__main__":
	main()
