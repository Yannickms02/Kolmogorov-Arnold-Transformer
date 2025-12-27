"""
GR-KAN FFN Module based on the KAT/FlashKAT Paper.

References:
- Yang & Wang (2024): Kolmogorov-Arnold Transformer
- Raffel & Chen (2025): FlashKAT - Understanding and Addressing Performance Bottlenecks

Key equations from the paper:
- GR-KAN(x) = W · F(x)  where F is group-wise rational activation
- F(x) = P(x)/Q(x) = (a0 + a1*x + ... + am*x^m) / (1 + |b1*x + ... + bn*x^n|)
"""

import torch
import torch.nn as nn
import math


class GroupRationalActivation(nn.Module):
	"""
	Group-wise Rational Activation (Padé Activation Unit) as used in KAT.

	Each group shares the same learned rational function coefficients,
	reducing parameters from din*dout to (m + n*g + 1) per layer.

	Args:
		dim: Input dimension (must be divisible by num_groups)
		n_groups: Number of coefficient groups (default: 8 as in KAT)
		num_degree: Degree of numerator polynomial m (default: 5 → 6 coefficients)
		denom_degree: Degree of denominator polynomial n (default: 4)
		init_fn: Initialization target ('identity', 'swish', 'gelu')
		use_layernorm: Whether to use layernorm (default: False)
		clamp_coef: Whether to clamp coefficient (default: True)
	"""

	def __init__(
			self,
			dim: int,
			n_groups: int = 8,
			num_degree: int = 5,
			denom_degree: int = 4,
			init_fn: str = 'swish',
			use_layernorm: bool = True,
			clamp_coef: bool = True,
	):
		super().__init__()

		# Adjust group count as needed
		if dim % n_groups != 0:
			for g in range(n_groups, 0, -1):
				if dim % g == 0:
					n_groups = g
					break

		self.dim = dim
		self.n_groups = n_groups
		self.group_size = dim // n_groups
		self.num_degree = num_degree  # m
		self.denom_degree = denom_degree  # n
		self.clamp_coef = clamp_coef # clamping for stability
		self.use_layernorm = use_layernorm

		# Numerator coefficients: a0, a1, ..., am per group
		# Shape: (num_groups, num_degree + 1)
		self.numerator = nn.Parameter(torch.zeros(n_groups, num_degree + 1))

		# Denominator coefficients: b1, b2, ..., bn per group
		# Shape: (n_groups, denom_degree)
		self.denominator = nn.Parameter(torch.zeros(n_groups, denom_degree))

		self._init_coefficients(init_fn)


	def _init_coefficients(self, init_fn: str):
		"""
		Initialize coefficients to approximate known activation functions.
		This is crucial for training stability (variance-preserving init).
		"""
		with torch.no_grad():
			if init_fn == 'identity':
				# F(x) ≈ x: set a1=1, rest=0
				self.numerator.zero_()
				self.numerator[:, 1] = 1.0
				self.denominator.zero_()

			elif init_fn == 'swish':
				# Approximate Swish
				self.numerator[:, 0] = 0.0
				self.numerator[:, 1] = 0.5
				self.numerator[:, 2] = 0.25
				if self.num_degree >= 3:
					self.numerator[:, 3] = 0.03
				if self.num_degree >= 4:
					self.numerator[:, 4] = 0.004
				if self.num_degree >= 5:
					self.numerator[:, 5] = 0.0005

				self.denominator[:, 0] = 0.5
				if self.denom_degree >= 2:
					self.denominator[:, 1] = 0.08
				if self.denom_degree >= 3:
					self.denominator[:, 2] = 0.008
				if self.denom_degree >= 4:
					self.denominator[:, 3] = 0.0008

			elif init_fn == 'gelu':
				self.numerator[:, 0] = 0.0
				self.numerator[:, 1] = 0.5
				self.numerator[:, 2] = 0.35
				if self.num_degree >= 3:
					self.numerator[:, 3] = 0.04

				self.denominator[:, 0] = 0.7
				if self.denom_degree >= 2:
					self.denominator[:, 1] = 0.08
			else:
				raise ValueError(f"Unknown init_fn: {init_fn}")

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Apply group-wise rational activation.

		Args:
			x: Input tensor of shape (..., dim)
		Returns:
			Output tensor of same shape
		"""

		orig_shape = x.shape

		# Reshape: (..., dim) → (..., n_groups, group_size)
		x = x.view(*orig_shape[:-1], self.n_groups, self.group_size)

		# Compute polynomial P(x) = a0 + a1*x + a2*x² + ... + am*x^m
		# Use Horner's method for numerical stability: P(x) = a0 + x*(a1 + x*(a2 + ...))
		num_coef = self.numerator

		P = num_coef[:, -1].view(1, self.n_groups, 1).expand_as(x)
		for i in range(self.num_degree - 1, -1, -1):
			coef = num_coef[:, i].view(1, self.n_groups, 1)
			P = coef + x * P

		# Compute A(x) = b1*x + b2*x² + ... + bn*x^n (inner part of denominator)
		# Horner: A(x) = x * (b1 + x * (b2 + x * (...)))
		if self.denom_degree > 0:
			den_coef = self.denominator

			A = den_coef[:, -1].view(1, self.n_groups, 1).expand_as(x)
			for i in range(self.denom_degree - 2, -1, -1):
				coef = den_coef[:, i].view(1, self.n_groups, 1)
				A = coef + x * A
			A = x * A

			# Safe PAU: Q = 1 + |A| + eps
			Q = 1.0 + torch.abs(A) + 1e-6
		else:
			Q = torch.ones_like(x)

		# F(x) = P(x) / Q(x)
		out = P / Q

		# Reshape back: (..., n_groups, group_size) → (..., dim)
		return out.view(orig_shape)

	def extra_repr(self) -> str:
		return (f'dim={self.dim}, n_groups={self.n_groups}, '
		        f'num_degree={self.num_degree}, denom_degree={self.denom_degree}')


class GRKANFFN(nn.Module):
	"""
	GR-KAN Feed-Forward Network as described in the KAT paper.

	Replaces standard MLP FFN in Transformer with:
		GR-KAN(x) = W · F(x)

	where F is the group-wise rational activation and W is a learnable weight matrix.

	Two-layer structure (like standard FFN):
		Layer 1: x → F₁(x) → Linear(d_model → d_ff)
		Layer 2: → F₂(·) → Linear(d_ff → d_model)

	Args:
		d_model: Model dimension
		d_ff: Feed-forward dimension (default: 4 * d_model)
		n_groups: Number of coefficient groups (default: 8)
		num_degree: Numerator polynomial degree (default: 5)
		denom_degree: Denominator polynomial degree (default: 4)
		dropout: Dropout probability
		n_hidden: Number of hidden layers in FFN (default: 0)
		use_layernorm: Use layernorm for deeper networks (default: False)
	"""

	def __init__(
			self,
			d_model: int,
			d_ff: int = None,
			n_groups: int = 8,
			num_degree: int = 5,
			denom_degree: int = 4,
			dropout: float = 0.0,
			n_hidden: int = 0,
			use_layernorm: bool = True
	):
		super().__init__()

		if d_ff is None:
			d_ff = 4 * d_model

		self.d_model = d_model
		self.d_ff = d_ff
		self.n_hidden = n_hidden
		self.use_layernorm = use_layernorm

		dims = [d_model, d_ff] + [d_ff] * n_hidden + [d_model]

		# Build layers
		self.activations = nn.ModuleList()
		self.linears = nn.ModuleList()
		self.norms = nn.ModuleList() if use_layernorm else None
		self.dropouts = nn.ModuleList()

		self.group_sizes = []

		for i in range(len(dims) - 1):
			if i == 0:
				init_type = 'identity'  # First layer identity
			else:
				init_type = 'swish'  # all other layers non-linear

			# Ensure dimensions are divisible by num_groups
			# Adjust num_groups if necessary
			self.group_sizes.append(self._adjust_groups(dims[i], n_groups))

			self.activations.append(GroupRationalActivation(
				dim=dims[i],
				n_groups=self.group_sizes[i],
				num_degree=num_degree,
				denom_degree=denom_degree,
				init_fn=init_type,
				use_layernorm=use_layernorm
			))
			self.linears.append(nn.Linear(dims[i], dims[i + 1]))

			# LayerNorm
			if use_layernorm:
				self.norms.append(nn.LayerNorm(dims[i + 1]))

			self.dropouts.append(nn.Dropout(dropout))

		# Apply variance-preserving initialization to linear layers
		self._init_linear_weights()


	def _adjust_groups(self, dim: int, target_groups: int) -> int:
		"""Find largest divisor of dim that is <= target_groups."""
		for g in range(target_groups, 0, -1):
			if dim % g == 0:
				return g
		return 1

	def _init_linear_weights(self):
		"""
		Variance-preserving initialization as described in KAT paper.

		W ~ N(0, α/d_in) where α = E[F(x)²]/Var[x] assuming x ~ N(0, 1)

		For identity init: α ≈ 1 (F(x) ≈ x)
		For Swish init: α ≈ 0.5 (empirical)
		"""

		for i, linear in enumerate(self.linears):
			is_boundary = (i == 0 or i == len(self.linears) - 1)

			if is_boundary:
				# Input-Layer: identity init → α ≈ 1.0
				fan_in = linear.in_features
				nn.init.normal_(linear.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))
			else:
				# Hidden layers: Swish init → α ≈ 0.5
				fan_in = linear.in_features
				nn.init.normal_(linear.weight, mean=0.0, std=math.sqrt(0.5 / fan_in))

			if linear.bias is not None:
				nn.init.zeros_(linear.bias)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		Args:
			x: Input tensor of shape (batch, seq, d_model)
		Returns:
			Output tensor of shape (batch, seq, d_model)
		"""
		for i in range(len(self.linears)):
			# Rational Activation
			x = self.activations[i](x)

			# Linear Layer
			x = self.linears[i](x)

			# LayerNorm
			is_last_layer = (i == len(self.linears) - 1)
			if self.use_layernorm and not is_last_layer:
				x = self.norms[i](x)

			# Dropout
			x = self.dropouts[i](x)

		return x

	def extra_repr(self) -> str:
		return (f'd_model={self.d_model}, d_ff={self.d_ff}, '
		        f'n_hidden={self.n_hidden}, groups={self.group_sizes}, '
		        f'layernorm={self.use_layernorm}')