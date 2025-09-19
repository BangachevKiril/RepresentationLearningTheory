pip install img2dataset
img2dataset \
  --url_list laion400m_parquets/*.parquet \
  --input_format parquet \
  --url_col URL --caption_col TEXT \
  --output_folder data/laion_100k \
  --output_format webdataset \
  --count 100000 \
  --number_sample_per_shard 1000
