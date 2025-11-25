# Copilot Instructions for SQL Server ML Project

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Project Context
This is a Python-based machine learning project that connects to SQL Server databases for data analysis and model development.

## Key Technologies
- **Database**: Microsoft SQL Server
- **Language**: Python 3.x
- **ML Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **Database Connectivity**: pyodbc, SQLAlchemy
- **Development Environment**: Jupyter notebooks for interactive analysis

## Code Guidelines
- Use pyodbc or SQLAlchemy for database connections
- Follow PEP 8 Python style guidelines
- Use type hints where appropriate
- Include proper error handling for database operations
- Use environment variables for database credentials
- Structure SQL queries for readability and maintainability

## Security Considerations
- Never hardcode database credentials
- Use environment variables or secure configuration files
- Implement proper connection pooling
- Use parameterized queries to prevent SQL injection

## ML Model Development
- Start with exploratory data analysis (EDA)
- Use appropriate data preprocessing techniques
- Implement proper train/validation/test splits
- Focus on feature engineering from SQL data
- Include model evaluation metrics and visualization
