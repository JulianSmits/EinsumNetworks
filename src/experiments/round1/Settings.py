from EinsumNetwork import EinsumNetwork

class Settings():
    """
    Settings class for experiments in round 1
    """
    def __init__(self):
        """ network size options """
        self.size_mixture_down = False
        # self.size = 'extra-small'
        self.size = 'small'
        # self.size = 'medium'
        # self.size = 'large'
        # self.size = 'extra-large'

        """ choose dataset, default is mnist """
        self.fashion_mnist = False
        self.svhn = False

        """ choose exponential family """
        self.exponential_family = EinsumNetwork.BinomialArray
        # self.exponential_family = EinsumNetwork.CategoricalArray
        # self.exponential_family = EinsumNetwork.NormalArray

        """ choose classes """
        self.classes = [2, 7]
        # self.classes = [2, 3, 5, 7]
        # self.classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # self.classes = None

        self.K = 10

        """ choose structure """
        # self.tructure = 'poon-domingos'
        self.structure = 'binary-trees'

        """ poon-domingos settings """
        self.pd_num_pieces = [4]
        # self.pd_num_pieces = [7]
        # self.pd_num_pieces = [7, 28]

        """ binary-trees (RAT-SPN) settings """
        if self.size == 'extra-small':
            self.depth = 1
        elif self.size == 'small':
            self.depth = 3
        elif self.size == 'medium':
            self.depth = 5
        elif self.size == 'large':
            self.depth = 7
        elif self.size == 'extra-large':
            self.depth = 9
        self.num_repetitions = 20 if self.size_mixture_down else 20 * len(self.classes)
        self.num_repetitions_mixture = int(20 / len(self.classes)) if self.size_mixture_down else 20

        """ dimensions """
        self.width = 28
        self.height = 28
        if self.svhn:
            self.width = 32
            self.height = 32

        """ learning settings """ 
        self.num_epochs = 2
        self.batch_size = 100
        self.online_em_frequency = 1
        self.online_em_stepsize = 0.05
        self.SGD_learning_rate = 0.1
