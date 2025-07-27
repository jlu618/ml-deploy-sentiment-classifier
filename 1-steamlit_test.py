# import streamlit as st
# import time
# from PIL import Image
# import os
# import boto3
# from transformers import pipeline
# import torch

# # s3 download
# local_path = "../s3_downloaded/tinybert_sentiment_model/"
# s3_bucket = "udemy-sentiment-analysis"
# s3_prefix = "model_folder/tinybert_sentiment_model/"

# s3 = boto3.client('s3')
# def download_directory_from_bucket(bucket_name, s3_prefix, local_directory):
#     s3 = boto3.client('s3')
#     if not os.path.exists(local_directory):
#         os.makedirs(local_directory)
    
#     try:
#         paginator = s3.get_paginator('list_objects_v2')
#         for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
#             if 'Contents' in result:
#                 for obj in result['Contents']:
#                     object_name = obj['Key']
#                     file_path = os.path.join(local_directory, os.path.relpath(object_name, s3_prefix)).replace("\\", "/")
#                     if not os.path.exists(os.path.dirname(file_path)):
#                         os.makedirs(os.path.dirname(file_path))
#                     s3.download_file(bucket_name, object_name, file_path)
#                     print(f"Downloaded {object_name} to {file_path}")
#             else:
#                 print("No objects found with the specified prefix.")
#     except Exception as e:
#         print(f"Error downloading directory from bucket {bucket_name}: {e}")

# # streamlit app
# st.title("Quick Sentiment Analysis & Human Motion Detection App")
# st.markdown("Analyze text sentiment using TinyBERT model", unsafe_allow_html=True)
# st.divider()

# button = st.button("Download Model")
# if button:
#     with st.spinner("Downloading...Please wait!"):
#         download_directory_from_bucket(s3_bucket, s3_prefix, local_path)
#         st.success("Model downloaded successfully!")

# text = st.text_area("Enter your text here")
# prediction_button = st.button("Predict Sentiment")

# # -----------------------------------------
# device =torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# if prediction_button:
#     classifier = pipeline("text-classification",
#                         model=local_path,
#                         device=device)
#     output = classifier(text)
#     st.write(output)
#     st.info(output)
