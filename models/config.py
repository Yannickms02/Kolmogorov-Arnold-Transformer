# Model sizes
MODEL_CONFIGS = {
    "tiny": {
        "d_model": 192,
        "n_heads": 3,
        "n_layers": 12,
	    "n_hidden": 0,
        "d_ff": {
	        "kan_bspline": 77,
	        "kan_mean": 77,
	        "kan_grkan": 768,
	        "mlp": 768,
	        "mlp_bspline": 768
        },
        "dropout": 0.1
    },
    "small": {
        "d_model": 384,
        "n_heads": 6,
        "n_layers": 12,
	    "n_hidden": 0,
        "d_ff": {
	        "kan_bspline": 154,
	        "kan_mean": 154,
	        "kan_grkan": 1536,
	        "mlp": 1536,
	        "mlp_bspline": 1536
        },
        "dropout": 0.1
    },
    "base": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
	    "n_hidden": 0,
        "d_ff": {
	        "kan_bspline": 307,
	        "kan_mean": 307,
	        "kan_grkan": 3072,
	        "mlp": 3072,
	        "mlp_bspline": 3072
        },
        "dropout": 0.1
    }
}

# Training Hyperparameters
TRAINING_CONFIGS = {
	"classification": {
		"batch_size": 64,
		"mini_batch_size": 16,
		"max_epochs": 10,
	    "lr_mlp": 3e-4,
	    "lr_kan_bspline": 3e-4,
		"lr_kan_mean": 3e-4,
	    "lr_kan_grkan": 3e-4,
		"lr_mlp_bspline": 3e-4,
	    "weight_decay": 0.1,
	    "warmup_steps": 500,
	    "early_stopping_patience": 3,
	    "early_stopping_min_delta": 0.001
	},
	"language_modelling": {
		"batch_size": 64,
		"mini_batch_size": 32,
		"max_epochs": 100,
	    "lr_mlp": 3e-4,
	    "lr_kan_bspline": 3e-4,
		"lr_kan_mean": 3e-4,
	    "lr_kan_grkan": 3e-4,
		"lr_mlp_bspline": 3e-4,
	    "weight_decay": 0.1,
	    "warmup_steps": 500,
	    "early_stopping_patience": 5,
	    "early_stopping_min_delta": 0.001
	}
}

SEEDS = [42, 1337, 2024]

# Architecture choice
ARCHITECTURES = [
    "mlp",           # A1
    "kan_bspline",   # A2
    "kan_mean",      # A3
	"kan_grkan",  # A4
	"mlp_bspline"    # A5
]

LM_DATASETS = ["fineweb", "wikitext"]