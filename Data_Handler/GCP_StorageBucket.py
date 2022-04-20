#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 11 13:57:47 2021

@author: marcus
"""
from google.cloud import storage


def create_bucket(bucket_name):
    from google.cloud.storage.constants import PUBLIC_ACCESS_PREVENTION_ENFORCED    
    
    storage_client = storage.Client()
    '''bucket setting'''
    bucket = storage_client.bucket(bucket_name)   
    bucket.storage_class = "STANDARD"
    bucket.iam_configuration.uniform_bucket_level_access_enabled = True
    bucket.iam_configuration.public_access_prevention = (
                                              PUBLIC_ACCESS_PREVENTION_ENFORCED
                                            )

    '''creating bucket & location'''
    new_bucket = storage_client.create_bucket(bucket, location = 'AUSTRALIA-SOUTHEAST1')
    print(
        '''
        Created bucket - {}
        location - {}
        storage class - {}
        access control - UNIFORM
        access  - NOT PUBLIC
        '''.format(
                   new_bucket.name, new_bucket.location, new_bucket.storage_class
                    )
        )



def download_blob(bucket_name, source_blob_name, destination_file_name):
    '''
    bucket_name = "workupload"
    source_blob_name = "filename"
    destination_file_name = "/path/to/folder/file.name"
    
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    blob.download_to_filename(destination_file_name)

    print(
        "Blob {} downloaded to {}.".format(
            source_blob_name, destination_file_name
        )
    )

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    '''
    bucket_name = "workupload"
    source_blob_name = "/path/to/folder/file.format"
    destination_file_name = "file.format"
    
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    blob.upload_from_filename(source_file_name)
    print(
        "File {} uploaded to {}.".format(
            source_file_name, destination_blob_name
            )
        )
    
def delete_blob(bucket_name, blob_name):

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.delete()

    print("Blob {} deleted.".format(blob_name))
