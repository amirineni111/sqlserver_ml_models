# SQL Server ML Project

A Python-based machine learning project for analyzing data from SQL Server databases and building predictive models.

## Features

- **Database Connectivity**: Connect to SQL Server databases using pyodbc and SQLAlchemy
- **Data Exploration**: Interactive Jupyter notebooks for data analysis
- **Machine Learning**: Build and evaluate ML models using scikit-learn
- **Visualization**: Create charts and graphs with matplotlib and seaborn
- **Security**: Environment-based configuration for database credentials

## Project Structure

```
├── .github/
│   └── copilot-instructions.md    # GitHub Copilot custom instructions
├── .vscode/
│   └── tasks.json                 # VS Code tasks configuration
├── data/
│   ├── raw/                       # Raw data exports (if needed)
│   └── processed/                 # Processed datasets
├── notebooks/
│   ├── 01_database_connection.ipynb    # Database setup and connection testing
│   ├── 02_data_exploration.ipynb       # Exploratory data analysis
│   └── 03_model_development.ipynb      # ML model building and evaluation
├── src/
│   ├── __init__.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── connection.py          # Database connection utilities
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessing.py       # Data cleaning and preprocessing
│   └── models/
│       ├── __init__.py
│       └── ml_models.py          # Machine learning model definitions
├── .env.example                   # Example environment variables
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Quick Start

### 1. Environment Setup

1. **Create a virtual environment:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure database connection:**
   ```powershell
   Copy-Item .env.example .env
   # Edit .env file with your SQL Server credentials
   ```

### 2. SQL Server Configuration

Configure your `.env` file with your SQL Server details:

```
SQL_SERVER=your-server-name
SQL_DATABASE=your-database-name
SQL_USERNAME=your-username
SQL_PASSWORD=your-password
SQL_DRIVER=ODBC Driver 17 for SQL Server
```

### 3. Start Development

1. **Test database connection:**
   Open `notebooks/01_database_connection.ipynb` and run the cells to verify connectivity.

2. **Explore your data:**
   Use `notebooks/02_data_exploration.ipynb` to analyze your tables and views.

3. **Build ML models:**
   Develop and evaluate models in `notebooks/03_model_development.ipynb`.

## Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning library
- **pyodbc**: SQL Server database connectivity
- **SQLAlchemy**: SQL toolkit and ORM
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **jupyter**: Interactive notebook environment
- **python-dotenv**: Environment variable management

## VS Code Extensions

This project works best with these VS Code extensions:
- Python
- SQL Server (mssql)
- Jupyter
- Python Environment Manager

## Security Notes

- Never commit `.env` files to version control
- Use Windows Authentication when possible
- Implement proper connection pooling for production use
- Use parameterized queries to prevent SQL injection

## Getting Help

- Check the notebooks for examples and documentation
- Use GitHub Copilot for code suggestions and explanations
- Refer to the `.github/copilot-instructions.md` for project-specific guidelines
