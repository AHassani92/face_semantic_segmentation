from Config.config import config
from Src.module import main
from Src.utils import ledger_csv
import os

if __name__ == '__main__':
    main(config)
    # ledger_csv(config)