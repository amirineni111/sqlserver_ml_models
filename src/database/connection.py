"""
Database connection utilities for SQL Server.

This module provides classes and functions for connecting to SQL Server databases
using both pyodbc and SQLAlchemy, with support for environment-based configuration.
"""

import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import quote_plus

import pyodbc
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SQLServerConnection:
    """
    A class to handle SQL Server database connections using pyodbc and SQLAlchemy.
    
    Supports both SQL Server authentication and Windows authentication.
    """
    
    def __init__(
        self,
        server: Optional[str] = None,
        database: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        driver: Optional[str] = None,
        trusted_connection: bool = False
    ):
        """
        Initialize the SQL Server connection.
        
        Args:
            server: SQL Server instance name or IP address
            database: Database name
            username: SQL Server username (not needed for Windows auth)
            password: SQL Server password (not needed for Windows auth)
            driver: ODBC driver name
            trusted_connection: Use Windows authentication if True
        """
        self.server = server or os.getenv('SQL_SERVER')
        self.database = database or os.getenv('SQL_DATABASE')
        self.username = username or os.getenv('SQL_USERNAME')
        self.password = password or os.getenv('SQL_PASSWORD')
        self.driver = driver or os.getenv('SQL_DRIVER', 'ODBC Driver 17 for SQL Server')
        self.trusted_connection = trusted_connection or os.getenv('SQL_TRUSTED_CONNECTION', '').lower() == 'yes'
        
        self._engine: Optional[Engine] = None
        self._connection_string: Optional[str] = None
        
    def get_connection_string(self) -> str:
        """
        Generate the connection string for SQL Server.
        
        Returns:
            Connection string for pyodbc
        """
        if self._connection_string:
            return self._connection_string
            
        if self.trusted_connection:
            self._connection_string = (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"Trusted_Connection=yes;"
            )
        else:
            self._connection_string = (
                f"DRIVER={{{self.driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.username};"
                f"PWD={self.password};"
            )
        
        return self._connection_string
    
    def get_sqlalchemy_engine(self) -> Engine:
        """
        Create and return a SQLAlchemy engine.
        
        Returns:
            SQLAlchemy Engine object
        """
        if self._engine:
            return self._engine
            
        connection_string = self.get_connection_string()
        # URL encode the connection string for SQLAlchemy
        connection_url = f"mssql+pyodbc:///?odbc_connect={quote_plus(connection_string)}"
        
        self._engine = create_engine(
            connection_url,
            pool_size=int(os.getenv('SQL_POOL_SIZE', 5)),
            max_overflow=int(os.getenv('SQL_MAX_OVERFLOW', 10)),
            pool_timeout=int(os.getenv('SQL_TIMEOUT', 30)),
            echo=os.getenv('DEBUG', 'False').lower() == 'true'
        )
        
        return self._engine
    
    def test_connection(self) -> bool:
        """
        Test the database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with pyodbc.connect(self.get_connection_string()) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                logger.info("Database connection successful")
                return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            pandas DataFrame with query results
        """
        try:
            engine = self.get_sqlalchemy_engine()
            
            if params:
                # Use SQLAlchemy's text() for parameterized queries
                df = pd.read_sql(text(query), engine, params=params)
            else:
                df = pd.read_sql(query, engine)
                
            logger.info(f"Query executed successfully, returned {len(df)} rows")
            return df
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def get_table_list(self) -> pd.DataFrame:
        """
        Get a list of all tables in the database.
        
        Returns:
            DataFrame with table information
        """
        query = """
        SELECT 
            TABLE_SCHEMA as schema_name,
            TABLE_NAME as table_name,
            TABLE_TYPE as table_type
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        return self.execute_query(query)
    
    def get_view_list(self) -> pd.DataFrame:
        """
        Get a list of all views in the database.
        
        Returns:
            DataFrame with view information
        """
        query = """
        SELECT 
            TABLE_SCHEMA as schema_name,
            TABLE_NAME as view_name,
            TABLE_TYPE as table_type
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_TYPE = 'VIEW'
        ORDER BY TABLE_SCHEMA, TABLE_NAME
        """
        return self.execute_query(query)
    
    def get_table_schema(self, table_name: str, schema_name: str = 'dbo') -> pd.DataFrame:
        """
        Get the schema information for a specific table.
        
        Args:
            table_name: Name of the table
            schema_name: Schema name (default: 'dbo')
            
        Returns:
            DataFrame with column information
        """
        query = """
        SELECT 
            COLUMN_NAME as column_name,
            DATA_TYPE as data_type,
            IS_NULLABLE as is_nullable,
            COLUMN_DEFAULT as column_default,
            CHARACTER_MAXIMUM_LENGTH as max_length,
            NUMERIC_PRECISION as numeric_precision,
            NUMERIC_SCALE as numeric_scale
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = ? AND TABLE_SCHEMA = ?
        ORDER BY ORDINAL_POSITION
        """
        return self.execute_query(query, {'table_name': table_name, 'schema_name': schema_name})
    
    def get_sample_data(self, table_name: str, schema_name: str = 'dbo', limit: int = 100) -> pd.DataFrame:
        """
        Get a sample of data from a table.
        
        Args:
            table_name: Name of the table
            schema_name: Schema name (default: 'dbo')
            limit: Number of rows to return (default: 100)
            
        Returns:
            DataFrame with sample data
        """
        query = f"SELECT TOP {limit} * FROM [{schema_name}].[{table_name}]"
        return self.execute_query(query)


def create_connection() -> SQLServerConnection:
    """
    Create a database connection using environment variables.
    
    Returns:
        SQLServerConnection instance
    """
    return SQLServerConnection()


def get_available_drivers() -> list:
    """
    Get a list of available ODBC drivers for SQL Server.
    
    Returns:
        List of available driver names
    """
    drivers = [driver for driver in pyodbc.drivers() if 'SQL Server' in driver]
    return drivers
