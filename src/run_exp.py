import sys

if len(sys.argv) > 1:

  if sys.argv[1] == '-1':
    from experiments.round1 import experiment_1_2
  elif sys.argv[1] == '-2':
    from experiments.round1 import experiment_3_4
  elif sys.argv[1] == '-3':
    from experiments.round1 import experiment_5_6
  elif sys.argv[1] == '-4':
    from experiments.round1 import experiment_7_8

else:

  from experiments.round1 import experiment_1_2
  # from experiments.round1 import experiment_3_4
  # from experiments.round1 import experiment_5_6
  # from experiments.round1 import experiment_7_8
