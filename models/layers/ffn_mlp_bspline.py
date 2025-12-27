"""
A5: MLP using learned b-spline activations
"""

import torch
import torch.nn as nn
import math


class BSplineActivation(nn.Module):
	"""
	Learned B-Spline activation functions

	Formally:
	-------
	σ(x) = Σ c_i · B_i(x) + SiLU(x)

	with:
	- c_i = learned coefficients
	- B_i(x) = B-Spline basis
	- SiLU(x) = Residual for stability
	"""

	def __init__(
			self,
			n_features: int,
			grid_size: int = 5,
			grid_range: tuple[int, int] = (-1,1),
			spline_order: int = 3
	):
		super().__init__()


		self.n_features = n_features
		self.grid_size = grid_size
		self.grid_range = grid_range
		self.spline_order = spline_order

		self.n_bases = grid_size + spline_order

		# Grid
		h = (self.grid_range[1] - self.grid_range[0]) / grid_size  # distance between grid points
		grid = torch.arange(-spline_order, grid_size + spline_order + 1) * h + self.grid_range[0]
		self.register_buffer('grid', grid)

		self.spline_weight = nn.Parameter(torch.empty(self.n_features, self.n_bases))

		# Spline weight init
		# Noise scaled with 1/grid_size (as in efficient-kan).
		noise_scale = 0.1
		std_dev = noise_scale / self.grid_size

		# trunc_normal_ to prevent extreme values in the beginning of training
		nn.init.trunc_normal_(self.spline_weight, mean=0.0, std=std_dev)

		self.spline_scale = nn.Parameter(torch.ones(self.n_features))
		self.base_scale = nn.Parameter(torch.ones(self.n_features))

	def compute_bspline_bases(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Compute B-Spline basis functions
		"""
		# x: (..., n_features)
		x = x.unsqueeze(-1)  # (..., n_features, 1)
		grid = self.grid  # (n_basis + spline_order + 1,)

		# Cox-de boor algorithm
		# B_i,0(x) = 1 if grid[i] <= x < grid[i+1], else 0
		bases = ((x >= grid[:-1]) & (x < grid[1:])).float()

		# solve recursively for higher degrees
		for k in range(1, self.spline_order + 1):
			# left term: (x - t_i) / (t_{i+k} - t_i)
			left_num = x - grid[:-(k + 1)]
			left_denom = grid[k:-1] - grid[:-(k + 1)]
			left_denom = torch.where(left_denom == 0, torch.ones_like(left_denom), left_denom)
			left = left_num / left_denom

			# right term: (t_{i+k+1} - x) / (t_{i+k+1} - t_{i+1})
			right_num = grid[k + 1:] - x
			right_denom = grid[k + 1:] - grid[1:-k]
			right_denom = torch.where(right_denom == 0, torch.ones_like(right_denom), right_denom)
			right = right_num / right_denom

			# combination
			bases = left * bases[..., :-1] + right * bases[..., 1:]

		return bases  # (..., n_features, n_bases)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Forward Pass using B-Spline activation.
		"""
		# B-Spline bases
		bases = self.compute_bspline_bases(x)  # (..., n_features, n_bases)

		# Weighted sum: Σ c_i · B_i(x)
		# spline_weight: (n_features, n_bases)
		spline_out = torch.einsum('...fb,fb->...f', bases, self.spline_weight)  # (..., n_features)

		# Residual term
		base_out = torch.nn.functional.silu(x)

		output = self.spline_scale * spline_out + self.base_scale * base_out

		return output


class BSplineFFN(nn.Module):
	"""
	A5: MLP Feed-Forward Network using B-Spline activation.
	"""

	def __init__(
			self,
			d_model: int,
			d_ff: int,
			n_hidden: int = 0,
			dropout: float = 0.1,
			grid_size: int = 5,
			grid_range: tuple[int, int] = (-1,1),
			spline_order: int = 3
	):
		super().__init__()

		layers = []

		# 1. Input Projection
		layers.append(nn.Linear(d_model, d_ff))
		layers.append(BSplineActivation(d_ff, grid_size, grid_range, spline_order))
		layers.append(nn.Dropout(dropout))

		# 2. Hidden Layers (optional)
		for _ in range(n_hidden):
			layers.append(nn.Linear(d_ff, d_ff))
			layers.append(nn.LayerNorm(d_ff))
			layers.append(BSplineActivation(d_ff, grid_size, grid_range, spline_order))
			layers.append(nn.Dropout(dropout))

		# 3. Output Projection
		layers.append(nn.Linear(d_ff, d_model))

		self.net = nn.Sequential(*layers)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
