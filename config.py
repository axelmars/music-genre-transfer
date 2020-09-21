default_config = dict(
	# pose_std=1,
	# pose_decay=0.001,
	pose_std=0,
	pose_decay=0,

	n_adain_layers=5,
	adain_dim=512,

	min_level_db=-80.0,
	max_level_db=43,

	perceptual_loss=dict(
		layers=[2, 5, 8, 13, 18],
		weights=[1, 1, 1, 1, 1],
		scales=[64, ]
	),

	train=dict(
		batch_size=26,
		n_epochs=1000
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=200
	)
)
