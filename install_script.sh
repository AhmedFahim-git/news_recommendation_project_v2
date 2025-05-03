# if [ -f .env ]; then
#     source .env
# fi

# curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# az login --identity --username $MANAGED_ID_CLIENT_ID

mkdir data
mkdir data/processed
mkdir data/processed/MINDsmall_train
mkdir data/processed/MINDsmall_dev

mv ../MINDsmall_train/behaviors.parquet data/processed/MINDsmall_train/
mv ../MINDsmall_train/news_text.parquet data/processed/MINDsmall_train/

mv ../MINDsmall_dev/behaviors.parquet data/processed/MINDsmall_dev/
mv ../MINDsmall_dev/news_text.parquet data/processed/MINDsmall_dev/

mv ../embeddings ./

export TOKENIZERS_PARALLELISM=false

# az storage blob download --container-name $CONTAINER_NAME --account-name $BUCKET_NAME --file data/processed/MINDsmall_train/behaviors.parquet --name behaviors.parquet --auth-mode key --sas-token $BLOB_SAS_TOKEN
# az storage blob download --container-name $CONTAINER_NAME --account-name $BUCKET_NAME --file data/processed/MINDsmall_train/news_text.parquet --name news_text.parquet --auth-mode key --sas-token $BLOB_SAS_TOKEN
# az storage blob download --container-name $CONTAINER_NAME --account-name $BUCKET_NAME --file mydb_train.sqlite --name mydb_train.sqlite --auth-mode key --sas-token $BLOB_SAS_TOKEN