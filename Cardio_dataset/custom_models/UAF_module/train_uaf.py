import os
import json
import argparse
from datetime import datetime
from typing import Tuple

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
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_COL = "cardio"
ID_COLS = ["id"]


def build_preprocessor(df: pd.DataFrame, target_col: str, drop_cols):
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


def load_splits(input_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Pipeline, list]:
	df = pd.read_csv(input_csv, sep=";")
	drop_cols = [c for c in ID_COLS if c in df.columns]
	train_df, val_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df[TARGET_COL])
	pre = build_preprocessor(train_df, TARGET_COL, drop_cols)
	X_train = train_df.drop(columns=drop_cols + [TARGET_COL], errors="ignore")
	y_train = train_df[TARGET_COL].astype(int).values
	X_val = val_df.drop(columns=drop_cols + [TARGET_COL], errors="ignore")
	y_val = val_df[TARGET_COL].astype(int).values
	X_train_t = pre.fit_transform(X_train).astype(np.float32)
	X_val_t = pre.transform(X_val).astype(np.float32)
	feature_names = [f"f_{i}" for i in range(X_train_t.shape[1])]
	return X_train_t, y_train.astype(np.int64), X_val_t, y_val.astype(np.int64), pre, feature_names


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
		self.feature = nn.Sequential(
			nn.Linear(in_dim + cond_dim, hidden), nn.LeakyReLU(0.2),
			nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
		)
		self.out = nn.Linear(hidden, 1)
	def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
		feat = self.feature(torch.cat([x, c], dim=1))
		return self.out(feat)
	def forward_with_features(self, x: torch.Tensor, c: torch.Tensor):
		feat = self.feature(torch.cat([x, c], dim=1))
		return self.out(feat), feat


class MLPPredictor(nn.Module):
	def __init__(self, in_dim: int, hidden: int = 256):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(in_dim, hidden), nn.ReLU(),
			nn.Linear(hidden, hidden), nn.ReLU(),
			nn.Linear(hidden, 2),
		)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)


def sample_noise_and_condition(batch_size: int, noise_dim: int, label_prior: float, balanced: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
	z = torch.randn(batch_size, noise_dim, device=DEVICE)
	if balanced:
		# Half 0s, half 1s (or as close as possible)
		n_pos = batch_size // 2
		c = torch.zeros(batch_size, 1, device=DEVICE)
		c[:n_pos] = 1.0
		perm = torch.randperm(batch_size, device=DEVICE)
		c = c[perm]
	else:
		c = (torch.rand(batch_size, 1, device=DEVICE) < label_prior).float()
	return z, c


def train_uaf(
		input_csv: str,
		output_dir: str,
		lambda_u: float = 1.0,
		steps_G_per_step_D: int = 1,
		predictor_steps_per_round: int = 5,
		batch_size: int = 256,
		epochs: int = 20,
		noise_dim: int = 64,
		smallset_M: int = 512,
		lr: float = 1e-4,
		r1_gamma: float = 10.0,
		feat_match_weight: float = 0.0,
		predictor_minibatches: int = 1,
		predictor_warm_start: bool = True,
		balanced_labels: bool = False,
):
	X_tr, y_tr, X_val, y_val, _, feature_names = load_splits(input_csv)
	in_dim = X_tr.shape[1]
	label_prior = float(y_tr.mean())

	G = MLPGenerator(noise_dim, 1, in_dim).to(DEVICE)
	D = MLPDiscriminator(in_dim, 1).to(DEVICE)
	P = MLPPredictor(in_dim).to(DEVICE)
	opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
	opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))
	opt_P = torch.optim.Adam(P.parameters(), lr=lr)
	bce = nn.BCEWithLogitsLoss()
	ce = nn.CrossEntropyLoss()

	dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True, drop_last=True)
	val_X_t = torch.from_numpy(X_val).to(DEVICE)
	val_y_t = torch.from_numpy(y_val).long().to(DEVICE)

	metrics = []
	for epoch in range(1, epochs + 1):
		epoch_losses = {"L_D": 0.0, "L_G_adv": 0.0, "util_proxy": 0.0, "L_fm": 0.0, "R1": 0.0}
		steps = 0
		for xb_np, yb_np in dl:
			xb = xb_np.to(DEVICE)
			yb = yb_np.float().unsqueeze(1).to(DEVICE)

			# --- Discriminator step(s) with R1 penalty on real
			for _ in range(max(1, steps_G_per_step_D)):
				opt_D.zero_grad(set_to_none=True)
				z, c = sample_noise_and_condition(xb.size(0), noise_dim, label_prior, balanced_labels)
				x_fake = G(z, c).detach()
				logits_real = D(xb.requires_grad_(True), yb)
				logits_fake = D(x_fake, c)
				loss_D = bce(logits_real, torch.ones_like(logits_real)) + bce(logits_fake, torch.zeros_like(logits_fake))
				# R1 gradient penalty: (γ/2)*||∇x D(x)||^2
				if r1_gamma > 0:
					grad_real = torch.autograd.grad(
						outputs=logits_real.sum(), inputs=xb, create_graph=True, retain_graph=True, only_inputs=True
					)[0]
					r1_pen = (r1_gamma / 2.0) * (grad_real.view(grad_real.size(0), -1).pow(2).sum(1).mean())
					loss_D = loss_D + r1_pen
					_epoch_r1 = float(r1_pen.detach().cpu())
				else:
					_epoch_r1 = 0.0
				loss_D.backward()
				opt_D.step()

			# --- Generator adversarial and utility-aware update
			opt_G.zero_grad(set_to_none=True)
			z, c = sample_noise_and_condition(xb.size(0), noise_dim, label_prior, balanced_labels)
			x_fake = G(z, c)
			logits_fake, feat_fake = D.forward_with_features(x_fake, c)
			loss_G_adv = bce(logits_fake, torch.ones_like(logits_fake))
			loss_fm = torch.tensor(0.0, device=DEVICE)
			if feat_match_weight > 0:
				with torch.no_grad():
					_, feat_real = D.forward_with_features(xb, yb)
				loss_fm = F.mse_loss(feat_fake.mean(dim=0), feat_real.mean(dim=0))

			# --- Utility-aware update (approximate)
			with torch.no_grad():
				S_x = []
				S_y = []
				for _ in range(smallset_M // batch_size + 1):
					bs = min(batch_size, smallset_M - len(S_x))
					if bs <= 0:
						break
					z_s, c_s = sample_noise_and_condition(bs, noise_dim, label_prior, balanced_labels)
					xk = G(z_s, c_s).detach()
					S_x.append(xk.cpu())
					S_y.append(c_s.squeeze(1).long().cpu())
				S_x = torch.cat(S_x, dim=0)
				S_y = torch.cat(S_y, dim=0)

			# Train predictor θ_p' for few steps on S (warm start optional)
			P_temp = MLPPredictor(in_dim).to(DEVICE)
			if predictor_warm_start:
				P_temp.load_state_dict(P.state_dict())
			opt_P_temp = torch.optim.Adam(P_temp.parameters(), lr=lr)
			S_dl = DataLoader(TensorDataset(S_x.to(DEVICE), S_y.to(DEVICE)), batch_size=min(256, len(S_x)), shuffle=True)
			used_minibatches = 0
			for _ in range(predictor_steps_per_round):
				for sx, sy in S_dl:
					logits = P_temp(sx)
					loss_p = ce(logits, sy)
					opt_P_temp.zero_grad(set_to_none=True)
					loss_p.backward()
					opt_P_temp.step()
					used_minibatches += 1
					if used_minibatches >= predictor_minibatches:
						break
				if used_minibatches >= predictor_minibatches:
					break

			# Validation loss on real val set
			with torch.no_grad():
				val_logits = P_temp(val_X_t)
				val_loss = ce(val_logits, val_y_t)

			# Total generator loss: L_G = L_G_adv + w_fm*L_fm + λ_u * util_proxy
			loss_G = loss_G_adv + feat_match_weight * loss_fm + lambda_u * val_loss
			loss_G.backward()
			opt_G.step()

			epoch_losses["L_D"] += float((loss_D.detach() - (_epoch_r1 if r1_gamma > 0 else 0)).cpu())
			epoch_losses["L_G_adv"] += float(loss_G_adv.detach().cpu())
			epoch_losses["util_proxy"] += float(val_loss.detach().cpu())
			epoch_losses["L_fm"] += float(loss_fm.detach().cpu())
			epoch_losses["R1"] += float(_epoch_r1)
			steps += 1

		# Optionally sync predictor
		P.load_state_dict(P_temp.state_dict())

		# Log epoch summary
		for k in epoch_losses:
			epoch_losses[k] /= max(1, steps)
		print(f"[UAF] Epoch {epoch}/{epochs} - L_D={epoch_losses['L_D']:.4f} | L_G_adv={epoch_losses['L_G_adv']:.4f} | util_proxy={epoch_losses['util_proxy']:.4f} | L_fm={epoch_losses['L_fm']:.4f} | R1={epoch_losses['R1']:.4f}")
		metrics.append({"epoch": epoch, **epoch_losses})

	# Final: sample synthetic matching train size
	G.eval()
	n = X_tr.shape[0]
	out = []
	labels = []
	with torch.no_grad():
		for i in range(0, n, batch_size):
			bs = min(batch_size, n - i)
			z, c = sample_noise_and_condition(bs, noise_dim, label_prior, balanced_labels)
			x = G(z, c)
			out.append(x.cpu().numpy())
			labels.append(c.cpu().numpy())
	synth = np.vstack(out)
	ys = np.vstack(labels).squeeze(1).astype(np.int64)

	# Evaluate utility: train classifier on synthetic, test on real validation
	clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
	clf.fit(synth, ys)
	val_probs = clf.predict_proba(X_val)[:, 1]
	auc_roc = float(roc_auc_score(y_val, val_probs))
	auc_pr = float(average_precision_score(y_val, val_probs))
	print(f"[UAF] Utility metrics -> AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")

	# Save outputs
	run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
	run_dir = os.path.join(output_dir, run_id)
	os.makedirs(run_dir, exist_ok=True)
	pd.DataFrame(synth, columns=feature_names).assign(**{TARGET_COL: ys}).to_csv(os.path.join(run_dir, 'synthetic.csv'), index=False)
	with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
		json.dump({
			"lambda_u": lambda_u,
			"steps_G_per_step_D": steps_G_per_step_D,
			"predictor_steps_per_round": predictor_steps_per_round,
			"batch_size": batch_size,
			"epochs": epochs,
			"noise_dim": noise_dim,
			"smallset_M": smallset_M,
			"lr": lr,
			"r1_gamma": r1_gamma,
			"feat_match_weight": feat_match_weight,
			"predictor_minibatches": predictor_minibatches,
			"predictor_warm_start": predictor_warm_start,
			"balanced_labels": balanced_labels,
			"input_csv": input_csv,
			"run_id": run_id
		}, f, indent=2)
	pd.DataFrame(metrics).to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
	with open(os.path.join(run_dir, 'utility.json'), 'w', encoding='utf-8') as f:
		json.dump({"auc_roc": auc_roc, "auc_pr": auc_pr}, f, indent=2)
	print(f"[UAF] Completed. Outputs saved to {run_dir}")


def parse_args():
	p = argparse.ArgumentParser(description="Utility-Aware GAN training (UAF)")
	p.add_argument('--input', type=str, default=os.path.join('custom_models', 'datasets', 'input', 'cardio_train_dataset.csv'))
	p.add_argument('--output', type=str, default=os.path.join('custom_models', 'datasets', 'generated', 'uaf'))
	p.add_argument('--lambda_u', type=float, default=1.0)
	p.add_argument('--steps_G_per_step_D', type=int, default=1)
	p.add_argument('--predictor_steps_per_round', type=int, default=5)
	p.add_argument('--predictor_minibatches', type=int, default=4)
	p.add_argument('--predictor_warm_start', action='store_true')
	p.add_argument('--batch_size', type=int, default=256)
	p.add_argument('--epochs', type=int, default=10)
	p.add_argument('--noise_dim', type=int, default=64)
	p.add_argument('--smallset_M', type=int, default=512)
	p.add_argument('--lr', type=float, default=1e-4)
	p.add_argument('--r1_gamma', type=float, default=10.0)
	p.add_argument('--feat_match_weight', type=float, default=0.0)
	p.add_argument('--balanced_labels', action='store_true')
	return p.parse_args()


if __name__ == '__main__':
	args = parse_args()
	train_uaf(
		input_csv=args.input,
		output_dir=args.output,
		lambda_u=args.lambda_u,
		steps_G_per_step_D=args.steps_G_per_step_D,
		predictor_steps_per_round=args.predictor_steps_per_round,
		batch_size=args.batch_size,
		epochs=args.epochs,
		noise_dim=args.noise_dim,
		smallset_M=args.smallset_M,
		lr=args.lr,
		r1_gamma=args.r1_gamma,
		feat_match_weight=args.feat_match_weight,
		predictor_minibatches=args.predictor_minibatches,
		predictor_warm_start=args.predictor_warm_start,
		balanced_labels=args.balanced_labels,
	)
