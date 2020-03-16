#!/bin/bash

# Help text
usage="
$(basename "$0") [-hpdw] [-v VERSION(S)]

Run pytest for emg3d locally in an isolated venv before submitting to
GitHub/Travis-CI; by default for all supported python versions of emg3d.

where:
    -h : Show this help text.
    -v : Python 3.x version, e.g. '-v 7' for Python 3.7. Default: '7 8'.
    -p : Print output of conda.
    -d : Delete environments after tests.
    -w : Disable pytest warnings.

"

# Set default values
PYTHON3VERSION="7 8"
PRINT="/dev/null"
PCKGS="numpy scipy pytest pytest-cov numba"
PROPS="--flake8"
INST="pytest-flake8 scooby"
WARN=""

# Get Optional Input
while getopts "hv:pdw" opt; do

  case $opt in
    h) echo "$usage"
       exit
       ;;
    v) PYTHON3VERSION=$OPTARG
       ;;
    p) PRINT="/dev/tty"
       ;;
    d) DELETE=true
       ;;
    w) WARN="--disable-warnings"
       ;;
    :) printf "missing argument for -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done


# Loop over Python versions
for i in ${PYTHON3VERSION[@]}; do

  # Environment name
  NAME=test_emg3d_3${i}

  # Print info
  STR="  PYTHON 3."${i}"  "
  LENGTH=$(( ($(tput cols) - ${#STR}) / 2 - 2 ))
  printf "\n  "
  printf '\e[1m\e[34m%*s' "${LENGTH}" '' | tr ' ' -
  if [ $((${#STR}%2)) -ne 0 ];
  then
      printf "-"
  fi
  printf "${STR}"
  printf '%*s\n' "${LENGTH}" '' | tr ' ' -
  printf "\e[0m\n"

  # Create venv
  if [ ! -d "$HOME/anaconda3/envs/$NAME" ]; then
    conda create -y -n $NAME python=3.${i} $PCKGS &> $PRINT
  fi

  # Activate venv
  source activate $NAME

  # Install flake8
  if [ ! -d "$HOME/anaconda3/envs"+$NAME ]; then
    conda install -y -c conda-forge empymod &> $PRINT
    pip install $INST &> $PRINT
  fi

  # Run tests
  pytest --cov=emg3d $PROPS $WARN
  coverage html

  # De-activate venv
  conda deactivate

  # Remove venv
  if [ "$DELETE" = true ] ; then
    conda remove -y -n $NAME --all &> $PRINT
  fi

done
