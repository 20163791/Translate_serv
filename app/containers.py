from dependency_injector import providers, containers


BATCH_SIZE = 5
MODEL_SIZE = 128
H = 2
NUM_LAYERS = 2
NUM_EPOCHS = 1

class Configs(containers.DeclarativeContainer):
    config = providers.Configuration('config')




