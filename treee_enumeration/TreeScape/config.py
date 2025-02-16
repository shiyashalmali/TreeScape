import os

class Config:
    """
    Base configuration class with default settings.
    Extend this class for environment-specific configurations.
    """
    # Secret key for securing sessions and other cryptographic operations
    SECRET_KEY = os.environ.get('SECRET_KEY', 'you-will-never-guess')

    # Folder to store uploaded files
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'app', 'static', 'UPLOADS')

    # Allowed image file extensions for uploads
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

    # Other global configurations
    # Example: Database URI (replace with actual connection string if needed)
    DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///default_db.sqlite3')

    # Placeholder for adding global configurations like email, logging, etc.
    # MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.example.com')


class DevelopmentConfig(Config):
    """
    Configuration class for development environment.
    """
    DEBUG = True
    ENV = 'development'
    # Use a local SQLite database or other development-specific settings
    DATABASE_URI = 'sqlite:///development_db.sqlite3'


class TestingConfig(Config):
    """
    Configuration class for testing environment.
    """
    TESTING = True
    DEBUG = False  # Testing should not have debug mode enabled
    WTF_CSRF_ENABLED = False  # Disable CSRF for testing purposes
    ENV = 'testing'
    # Use an in-memory SQLite database for tests
    DATABASE_URI = 'sqlite:///:memory:'


class ProductionConfig(Config):
    """
    Configuration class for production environment.
    """
    DEBUG = False
    ENV = 'production'
    # Fetch production database URI from environment
    DATABASE_URI = os.environ.get('DATABASE_URI', 'sqlite:///production_db.sqlite3')

    # Add any production-specific configurations, such as logging levels, monitoring, etc.


# Configuration dictionary for environment-based selection
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig,  # Default to development configuration
}

