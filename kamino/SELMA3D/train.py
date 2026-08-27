import os
import logging
import torch
from monai.bundle import ConfigParser
from monai.inferers import sliding_window_inference
import argparse

from pygments.lexer import default


def setup_logger(log_file: str):
    """Configures logging to output to both console and a specified log file."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True
    )

def main(config_file):
    # 1. Parse configuration
    parser = ConfigParser()
    parser.read_config(config_file)

    crop_size = parser.get("crop_size")
    epochs = parser.get("epochs")
    val_interval = parser.get("val_interval")
    model_path = parser.get("model_path")
    new_model_path = parser.get("new_model_path", default=model_path)
    log_file = parser.get("log_path", default="./trained_models/train.log")
    scheduler_config = parser.get("scheduler", default={"type": "ReduceLROnPlateau", "params": {"mode": "min", "patience": 3, "factor": 0.5}})

    # 2. Initialize Logger
    setup_logger(log_file)
    logging.info("=" * 40)
    logging.info(f"Starting execution | Log file: {log_file}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using compute device: {device}")

    # 3. Instantiate objects via ConfigParser
    model = parser.get_parsed_content("model").to(device)
    train_loader = parser.get_parsed_content("train_loader")
    val_loader = parser.get_parsed_content("val_loader")
    optimizer = parser.get_parsed_content("optimizer")
    loss_function = parser.get_parsed_content("loss_function")
    metric = parser.get_parsed_content("metric", default=None)
    scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_config["type"])
    scheduler = scheduler_class(optimizer, **scheduler_config["params"])

    # 4. Resume Checkpoint handling
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        logging.info(f"Loaded existing model weights from: {model_path}")
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        open(log_file, 'w').close()
        logging.info("No saved checkpoint found. Starting training from scratch.")

    if new_model_path != model_path:
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)

    scaler = torch.amp.GradScaler('cuda') if device.type == "cuda" else None
    best_val_loss = float("inf")

    # 5. Training & Validation Loop
    for epoch in range(epochs):
        logging.info(f"--- Epoch [{epoch + 1}/{epochs}] ---")
        model.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            steps += 1
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / steps
        logging.info(f"Train Loss: {avg_train_loss:.5f}")

        # Validation Step
        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0.0
            val_steps = 0
            metric_score = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_steps += 1
                    val_inputs = val_batch["image"].to(device)
                    val_targets = val_batch["label"].to(device)

                    if scaler:
                        with torch.amp.autocast('cuda'):
                            val_outputs = sliding_window_inference(
                                val_inputs,
                                roi_size=(crop_size, crop_size, crop_size),
                                sw_batch_size=4,
                                predictor=model
                            )
                    else:
                        val_outputs = sliding_window_inference(
                            val_inputs,
                            roi_size=(crop_size, crop_size, crop_size),
                            sw_batch_size=4,
                            predictor=model
                        )

                    if metric:
                        metric_score += metric(val_outputs, val_targets).item()
                    else:
                        v_loss = loss_function(val_outputs, val_targets)
                        val_loss += v_loss.item()

            if metric:
                avg_metric_score = metric_score / val_steps
                scheduler.step(avg_metric_score)
                logging.info(f"Metric Score: {avg_metric_score:.5f}")
            else:
                avg_val_loss = val_loss / val_steps
                scheduler.step(avg_val_loss)
                logging.info(f"Val Loss: {avg_val_loss:.5f} (Best: {best_val_loss:.5f})")

            if not metric and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), new_model_path)
                logging.info(f"  -> Saved new best checkpoint to {new_model_path}")
            elif metric and avg_metric_score > best_val_loss:
                best_val_loss = avg_metric_score
                torch.save(model.state_dict(), new_model_path)
                logging.info(f"  -> Saved new best checkpoint to {new_model_path}")

    logging.info("=" * 40)
    logging.info(f"Training completed. Best Validation Loss: {best_val_loss:.5f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 3D model with MONAI.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config)