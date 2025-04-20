# Download the package to configure the Microsoft repo
curl -sSL -O https://packages.microsoft.com/config/ubuntu/$(grep VERSION_ID /etc/os-release | cut -d '"' -f 2)/packages-microsoft-prod.deb
# Install the package
sudo dpkg -i packages-microsoft-prod.deb
# Delete the file
rm packages-microsoft-prod.deb

# Install the driver
sudo apt-get update
sudo ACCEPT_EULA=Y apt-get install -y msodbcsql18
# optional: for bcp and sqlcmd
sudo ACCEPT_EULA=Y apt-get install -y mssql-tools18
echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc
source ~/.bashrc
# optional: for unixODBC development headers
sudo apt-get install -y unixodbc-dev


if [ -f .env ]; then
    source .env
fi

curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login --identity --username $MANAGED_ID_CLIENT_ID

mkdir data
mkdir data/processed
mkdir data/processed/MINDsmall_train

az storage blob download --account-name bucketlist --container-name mycontainer --file data/processed/MINDsmall_train/behaviors.parquet --name behaviors.parquet --auth-mode login
az storage blob download --account-name bucketlist --container-name mycontainer --file data/processed/MINDsmall_train/news_text.parquet --name news_text.parquet --auth-mode login