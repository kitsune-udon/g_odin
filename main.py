from odin_datamodule import ODINDataModule
from train_utils import main
from g_odin import GODIN


if __name__ == '__main__':
    main(ODINDataModule, GODIN, "g_odin")
