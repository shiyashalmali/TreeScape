from app import create_app

# Create the app instance using the factory function
app = create_app()

# Only one if __name__ == '__main__' block is needed
if __name__ == '__main__':
    # Run the app with debug mode
    app.run(debug=True)
