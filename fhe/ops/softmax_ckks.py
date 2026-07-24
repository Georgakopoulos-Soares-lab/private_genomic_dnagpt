"""Run the isolated CKKS softmax self-check."""

import sys

from op_matrix_ckks import main


if __name__ == "__main__":
    sys.exit(main(default_only="softmax"))
