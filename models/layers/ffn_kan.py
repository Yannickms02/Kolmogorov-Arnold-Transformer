from efficient_kan import KAN as EfficientKAN
from torch import nn


class KANFFN(nn.Module):
	"""
	A2 + A3: KAN-FFN based on efficient-kan
	Option for different aggregation method on nodes (as per Altarabichi (2024): arXiv:2407.20667)
	"""

	def __init__(self, d_model, d_ff, grid_size=5, spline_order=3, aggregation='sum', n_hidden=0):
		super().__init__()

		layer_dims = [d_model, d_ff] + [d_ff] * n_hidden + [d_model]
		self.kan = EfficientKAN(
			layers_hidden=layer_dims,
			grid_size=grid_size,
			spline_order=spline_order,
		)
		self.aggregation = aggregation

	def forward(self, x):
		# Transformer expects shape (batch, seq, dim) whereas efficient-kan (batch, dim)
		b, s, d = x.shape
		x_flat = x.view(-1, d)

		if self.aggregation == 'sum':
			# Standard efficient-kan forward
			out = self.kan(x_flat)

		elif self.aggregation == 'mean':
			# Manual forward with mean aggregation per layer
			out = x_flat
			for layer in self.kan.layers:
				out = layer(out)
				out = out / layer.in_features

		else:
			raise ValueError(f"Unknown aggregation: {self.aggregation}")

		# Reshape to initial shape
		out = out.view(b, s, d)

		return out