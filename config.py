class Config(object):

    DEBUG = False
    TESTING = False
    ASSET_PATH = 'static/'
    MODEL_PATH = 'model/'
    CONTROLLERS = 'controllers/'
    UPLOAD_FOLDER = 'static/'
    VIEWS = 'views/'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    SECRET_KEY = "secret key"


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True
    SECRET_KEY = "secret key"


class TestingConfig(Config):
    TESTING = True

