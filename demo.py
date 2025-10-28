lr_scheduler.ReduceLROnPlateau(
    optimizer=optimizer,
    mode=scheduler_mode,
    verbose=False,
    threshold_mode="abs",
    factor=config.get("decrease_factor", 0.1),
    patience=config.get("patience", 60),
)