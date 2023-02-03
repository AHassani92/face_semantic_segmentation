# basic utilities
import os

# perception repository libraries
from Configs.config import config
from Src.trainer import main

if __name__ == '__main__':
    main(config)
    # ledger_csv(config)