import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import argparse

from model3 import PedalNet


def main(args):
    model = PedalNet(
        num_channels=args.num_channels,
        dilation_depth=args.dilation_depth,
        num_repeat=args.num_repeat,
        kernel_size=args.kernel_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data=args.data,
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs, 
        accelerator="gpu" if args.gpus else "cpu",
        devices=int(args.gpus) if args.gpus else 1,
        log_every_n_steps=10,
        callbacks=[early_stop_callback]
    )
    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_channels", type=int, default=16)
    parser.add_argument("--dilation_depth", type=int, default=8)
    parser.add_argument("--num_repeat", type=int, default=3)
    parser.add_argument("--kernel_size", type=int, default=3)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=3e-3)

    parser.add_argument("--max_epochs", type=int, default=1_500)
    parser.add_argument("--gpus", default=None)

    parser.add_argument("--data", default="data.pickle")
    args = parser.parse_args()
    main(args)
