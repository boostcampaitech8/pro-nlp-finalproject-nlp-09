# gcloud Storage Upload Manual
참고 문서: https://docs.cloud.google.com/storage/docs/uploading-objects#upload-object-cli

## 파일 업로드 명령어

가격 데이터를 gcs 버킷에 업로드 하는 cli 명령어 예시
```bash
gcloud storage cp {file1}.csv {file2}.csv gs://{root}/raw/prices
```

## 디렉토리 업로드 명렁어 예시
```bash
gcloud storage rsync --recursive LOCAL_DIRECTORY gs://DESTINATION_BUCKET_NAME/FOLDER_NAME
```

