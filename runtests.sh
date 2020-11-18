#!/bin/bash

# Help text
usage="
$(basename "$0") [-hpdnw] [-v VERSION(S)]

Run pytest for emg3d locally in an isolated venv before submitting to
GitHub/Travis-CI; by default for all supported python versions of emg3d.

where:
    -h : Show this help text.
    -v : Python 3.x version, e.g. '-v 7' for Python 3.7. Default: '7 8'.
    -p : Print output of conda.
    -d : Delete environments after tests.
    -n : Run tests without soft dependencies.
    -w : Disable pytest warnings.

"

# Set default values
PYTHON3VERSION="7 8"
PRINT="/dev/null"
PCKGS="scipy numba pytest pytest-cov pytest-flake8 pytest-console-scripts"
SOFT="xarray empymod h5py scooby 'discretize<0.6.0' matplotlib"
PROPS="--flake8"
WARN=""
SD="_soft"

# Get Optional Input
while getopts "hv:pdnw" opt; do

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
    n) SOFT=""
       SD="_no-soft"
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
  NAME=test_emg3d_3${i}${SD}

  # Print info
  STR="  PYTHON 3."${i}${SD}"  "
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

  # Create virtual environment
  if [ ! -d "$HOME/anaconda3/envs/$NAME" ]; then
    conda create -y -c conda-forge -n $NAME python=3.${i} $PCKGS $SOFT &> $PRINT
  fi

  # Activate virtual environment
  source activate $NAME

  # Install emg3d
  python setup.py install &> $PRINT

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
