from flask import Flask
from flask_cors import CORS
from config import config
import os

def create_app(config_name='default'):
    """
    Factory function to create a Flask app instance with the specified configuration.
    :param config_name: The name of the configuration to use ('default', 'development', etc.)
    :return: Configured Flask app instance
    """
    # Initialize the Flask app
    app = Flask(__name__)
    
    # Enable CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
    
    # Load the configuration from config.py based on the provided config_name
    app.config.from_object(config[config_name])
    
    # Set the upload folder dynamically, ensuring compatibility across environments
    app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'app', 'static', 'UPLOADS')
    
    # Ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Register the main Blueprint
    from .routes import main
    app.register_blueprint(main)
    
    return app