#!/bin/bash
# Install Microsoft ODBC Driver 18 for SQL Server on Render (Ubuntu)
curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add -
curl https://packages.microsoft.com/config/ubuntu/20.04/prod.list > /etc/apt/sources.list.d/mssql-release.list
apt-get update
ACCEPT_EULA=Y apt-get install -y msodbcsql18
apt-get install -y unixodbc-dev

# Install Python dependencies
pip install -r requirements.txt
