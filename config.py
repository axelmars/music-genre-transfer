import numpy as np
default_config = dict(

	# pose_std=1,
	# pose_decay=0.001,
	pose_std=0.4,
	pose_decay=0.0004,
	# pose_std=0,
	# pose_decay=0,

	n_adain_layers=4,
	adain_dim=256,

	min_level_db=-100.0,
	max_level_db=41.0,
	min_phase=-0.1,
	max_phase=0.1,

	perceptual_loss=dict(
		layers=[2, 5, 8, 13, 18],
		weights=[1, 1, 1, 1, 1],
		scales=[128, ]
	),

	train=dict(
		batch_size=32,
		n_epochs=1000
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=200
	)
)
